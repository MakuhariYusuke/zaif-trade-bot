# 452# Micro-timeouts (TIF Emulation) 詳細設計案 (ph2)

**種別**: plan  
**日付**: 2026-03-16  
**関連**: 447# (提案B), 450#

---

## 1. 概要と目的

**Micro-timeouts（TIF方式の板監視）**の主な目的は、「Adverse Selection（逆選択 / 毒性のあるフローへの被弾）の大幅な削減」です。
現在の `v460` アーキテクチャでは、Maker注文を出した後、最大90秒（売りは75秒）程度、板を固定で放置（ポーズ）しています。大口のTakerやHFT業者はこの「長時間存在し続ける固定板」を格好のターゲット（カモ）にします。
これを「**例えば15秒以内に埋まらなければ、無条件にキャンセルして相場から逃げる（Time-In-Force: FOK/IOCに近い挙動のシミュレート）**」ことで、強力な生存バイアスを持たせる設計です。

## 2. 現状 (v460) の制約と課題

現在の基本パラメーター（`configs/v460/fill_test.yaml`）は以下のようになっています。
- `cycle_interval_sec: 120.0`
- `order_timeout_sec: 90.0`
- `order_timeout_sec_sell: 75.0`

**【最大の設計課題】**
もし単に `order_timeout_sec = 15.0` に変更した場合、15秒で注文がキャンセルされた後、残りの `105秒間`（120 - 15）何もせずに待機する（あるいは無駄なサイクルスリープに入る）ことになり、資金効率（Time-in-Market）が極端に低下する懸念があります。

---

## 3. アーキテクチャ設計 (3つのアプローチ)

この15秒の「短命・逃げ足重視」注文を、120秒のサイクルの中にどう組み込むか。3つのアプローチを比較します。

### 案1: 同期・ショートサイクル化 (最もシンプル)
- **仕組み**: `cycle_interval_sec` 自体を `15.0` や `30.0` に短縮し、`order_timeout_sec = 10.0` 等にする。
- **Pros**: 既存のロジックをいじらず、YAMLの変更だけで済む。
- **Cons**: APIのRate Limit（Coincheckの呼び出し制限）に確実に抵触する。1サイクルごとのオーバーヘッド（状態保存、ML特徴量計算など）が重すぎてシステムが破綻する可能性が高い。

### 案2: サブサイクル・ポーリング型 (★推奨)
- **仕組み**: メインの `cycle_interval_sec: 120.0` は据え置き、注文監視のフェーズ（Wait・ポーリング部分）だけを**小間切れのループ（サブサイクル）**に分割する。
  - 例: 注文を出す → 15秒待つ → キャンセルする → 未約定なら**再度最新の価格を計算（Re-quote）**して注文し直す、を1サイクル内で最大N回繰り返す。
- **Pros**: Rate Limitをケアしつつ、常に「最新の価格で15秒だけ存在し、消える」板を出し直せる。Adverse Selectionを弾きつつ、Time-in-Marketも高水準で維持できる。
- **Cons**: `run_fill_test.py` 内部の `wait_full_cycle` 周辺のポーリングロジックの改修が中規模になる。

### 案3: 非同期エグゼキューター型 (Cancel-and-Forget)
- **仕組み**: 注文管理をメインサイクルから切り離し、独立したスレッド/タスクが「15秒ごとに生存確認しキャンセル」を行う。
- **Pros**: サイクル時間は完全に正確に維持される。
- **Cons**: マルチスレッド/非同期制御のバグ（スレッド間の状態の不整合やゾンビオーダー）が起きやすく、既存の逐次処理アーキテクチャ（`v460`）と相性が悪い。

---

## 4. サブサイクル型（案2）の具体化

案2を採用した場合のロジックフローです。

1. **Cycle開始**: 通常通り、各種特徴量やシグナル（Cross-Venue含む）を評価し、Target Offsetを決定。
2. **Sub-cycle ループ (例: 最大4回 / 1回15秒)**:
   - **Step A**: 現在の最適価格でMaker注文をPost。
   - **Step B**: `micro_timeout_sec` (例: 15秒) だけ待機（ポーリング監視）。
   - **Step C**: もし15秒経過して未約定（`filled < order_quantity`）なら、**能動的にCancel（キャンセル）**。
   - **Step D**: キャンセル成功後、スプレッドやCross-Venueの重大なアラート（提案CのSpread急拡大など）がなければ、**即座にその時点の最新価格（Mid等）を取り直し、新しい注文をPost（再評価・Re-quote）**。"Step B" に戻る。
3. **Cycle終了**: 規定回数のサブサイクル（あるいは全体の `cycle_interval_sec` 到達）で、最終的な残存注文をキャンセルし、評価や状態保存を行って次のサイクルへ。

### 新規追加・変更パラメーター群（`configs/v460/fill_test.yaml` 向け）
```yaml
# --- Micro-timeouts (TIF Emulation) ---
micro_timeout:
  enabled: true
  wait_sec: 15.0                # 1回あたりの最大配置時間。これを超えればキャンセル
  wait_sec_sell: 10.0           # 売りはさらに短く逃げる（既存の sell timeout 優遇を踏襲）
  max_requote_per_cycle: 4      # 1サイクル(120s)内で、指値を置き直す最大回数
  requote_cooloff_sec: 5.0      # キャンセル後、再突入するまでの微小な冷却期間（HFTのノイズ回避）
  cancel_on_cross_venue_flip: true  # 15秒待たなくとも、Cross-Venueが反転した瞬間にCancelするか
```

---

## 5. 実装に向けた影響範囲 (ph2 実装方針)

この機能は、指値の生存期間に直接手を入れるため、既存の実装 (ph2 maker執行可能性検証 / G1.1-exec) に対して以下の改修を求めます。

1. **ill_cycle_executor.py (メインループのサブサイクル化)**
   現在の execute_cycle は以下のような単一パスのフローです。
   特徴量抽出 → 価格計算 → 注文(_execute_order) → 待機(_monitor_fill_polling, 最大90秒)
   
   これを**「方策(Policy)」と「執行(Execution)」で分離**し、執行部分をループ化します。
   `python
   # 1. Policy Phase (120秒に1回だけ計算)
   features = await self._extract_features(...)
   directional_bias = self._sidecar.get_signal()
   
   # 2. Execution sub-cycle Phase (最大 N 回ループ)
   for attempt in range(max_requote_per_cycle):
       # 毎回最新の中値(Mid)を取り直す
       current_mid = await _fetch_latest_mid(...)
       
       # 最新Midから価格を計算 (Offsetは固定でよい)
       order_price = _calculate_target_price(current_mid, offset, ...)
       
       # 注文
       order = await self._execute_order(...)
       
       # マイクロタイムアウトで待機 (15秒)
       monitor = await self._monitor_fill_polling(..., effective_timeout=15.0)
       
       if monitor.filled >= order_quantity:
           break  # 刺さったら終了
       # 15秒でタイムアウトしたら次ループへ (Re-quote)
   `

2. **部分約定 (Partial Fill) のハンドリング**
   15秒経過時に「ロットの半分だけ約定している」場合、残りをキャンセルし、order_quantity を減算して Re-quote ループへ継続する実装がクリーンです。

3. **ロギングとA/Bテスト**
   1サイクル中に何度も注文・キャンセルを繰り返すため、ill_records に 
equote_attempts （何度目の Re-quote で刺さったか）を記録し、b_offset_comparison.py で分析できるようにします。

## 6. Next Action (提案)
000番ドキュメントの定義に従い、本件は **ph2 (maker執行可能性検証 / G1.1-exec)** のスコープで扱います。
450# で指摘されたような「実装したつもりが適用されていなかったギャップ」を防ぐため、まずは **ドライラン環境 (ill_test / paper trading)** 下で、この「15秒出して引っ込める」サブサイクルの挙動を組み込み、ログを検証してからの投入（ph2.1等）を提案します。
