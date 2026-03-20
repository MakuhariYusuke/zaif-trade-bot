# 494# [Phase G] 493# 検証と市場理論に基づくアクションプラン：Continuous Participation の回復

> 種別: verify / action_plan
> 対象: 493# `docs/v460/493_phg_rev_490_492_profit_first_fill_test_review.md`

---

## 1. 493# 評価：総括への完全な同意と市場理論の裏付け

493# の結論である**「Composite Risk等の新機構を入れるより先に、Runtimeのクリーン化・TTL短縮・在庫制約の短絡化を優先すべき」**という方針は、実運用と市場構造の観点から完全に支持できる。

特に、ログ分析で指摘された **「Slow Fill（待機30秒以上）の逆選択によるマイナス（`queue_wait_sec >= 30s` で avg -3.750）」** は、マーケットメイキングにおける古典的な「Glosten-Milgrom モデル（情報非対称性市場での情報優位者からの逆選択）」そのものである。
長く板に残る注文は、安全な相場（Ranging）では約定せず、相場が突き抜ける（Trending/Adverse）時にだけ食い破られる為、ここを放置して複雑なリスク判定（Composite Risk等）を前段に入れても意味がない。

これを踏まえ、以下に即座に実行すべき具体的な Tactical Fix（コード／設定変更）を定義する。

---

## 2. 具体的なアクションプラン（Tactical Fix）

### 2.1 Micro-timeout の完全有効化とTTLの短縮（即時対応要）
現状の `configs/v460/fill_test.yaml` を確認した結果、以下の事実が判明した。

```yaml
# configs/v460/fill_test.yaml
micro_timeout:
    enabled: true                    # 454# Step 1: micro-timeout 有効化        
    wait_sec: 30.0                   # 454# Step 1: 15→30s (保守的)
    wait_sec_sell: 20.0              # 454# Step 1: 10→20s (sell 側も保守的) 
```

`micro_timeout` 自体は有効化されているものの、`wait_sec: 30.0` という極めて保守的（長い）設定のままである。
493# のデータが示す通り、「30秒以上の待機」は致命的なアドバースセレクションに繋がる。

- **Action**: `fill_test.yaml` の `micro_timeout` 設定を即座に以下へ切り下げる（Step 2/3 のアグレッシブ化）。
  - `wait_sec`: `30.0` → **`15.0`**
  - `wait_sec_sell`: `20.0` → **`10.0`**
  - `max_requote_per_cycle`: `2` → **`4`**
これにより、古い価格を放置して被弾する負け筋を物理的に断つ。

### 2.2 Route-to-Kill Deadlock に対する "Inventory Recovery Skew" の導入
現在、`scripts/v460/lib/orchestrator_balance.py` において、`buy insufficient`（資金枯渇）かつ切替先の `sell` が `kill-gated` (Toxicity 等によりVeto) の場合、`both-side blocked` として即 Skip (休止) されている（421# 制約）。

- **市場理論的課題**: これが 492#/493# の指標に現れている**「参加しなさすぎる（Participation Collapse）」の直接原因（デッドロック）**である。在庫が極端に偏っている際に、Gate制限を絶対視して休むのは、Makerとしての流動性提供の放棄である。
- **Action**: `route_to_kill_deadlock` 時の完全 Skip 処理を廃止し、**「Inventory Recovery Skew (在庫修復スキュー)」メカニズム** を導入する。
  - *具体策*: 枯渇していない側（kill-gated な側）に対し、Veto を Bypass する代わりに、クランプの天井（`0.20`等）を越えた非常に広いスプレッド（`forced_offset_mult = 2.0` など、例:`0.35`）で指値を入れる。
  - これにより、「不利な相場で食われても十分なプレミアムが確保でき、同時に在庫の偏りが修復される」非対称な期待値状態を獲得できる。引っ込む（Skip）のではなく、奥に控える（Skewing）のが本来のMakerbotの振る舞いである。

### 2.3 Runtime Drift の完全除去（P0 止血）
489# および 493# で指摘されている `NameError: name '_sidecar_signal' is not defined` の多発は、現行の Git HEAD のコード（`orchestrator_mid_cycle.py:141`）には変数の定義が存在するため、**完全に本番プロセスのキャッシュ残留・Deploy不整合**である。

- **Action**: いかなるパラメータ調整やスクリプト分析を行う前に、**直ちに本番プロセスの完全終了と Cold Restart（仮想環境、プロセスマネージャ、メモリ上のレコードキャッシュの破棄）を実行**する。
  この不整合状態でのロギングを用いた分析は因果関係を完全に見誤らせる。

---

## 3. セルフレビューと Next Steps

**セルフレビュー:**
- 新たに野良の分析スクリプトを増やすことなく静的なソース・Config追跡により裏付けを取った。指示に合致している。
- 単に「493の言う通り」とするだけでなく、「なぜなら `wait_sec: 30.0` のままになっているから」「なぜなら route_to_kill_deadlock が完全に Veto として働いているから」という具現化された修正ポイント（Tactical Fix）を提示できた。
- 市場理論（Inventory Skewing vs Adverse Selection）による「参加の復元方法」の提言が妥当な形でまとまった。

**Next Steps**:
まずは上記の即時対応（1. Runtime Restart, 2. TTLの半減, 3. Deadlock Bypass Skewing）を適用し、12〜24時間の稼働でログを監視。「Same-SHA, Same-Run」の環境下でFill Rateが有意に復元し、Profit Factorが安定するかを観測する。
その検証の完了後に初めて、490# や 491# で提案されたアーキテクチャの抜本的変更（Composite Riskの導入やSACへの状態移管）に移行すべきである。