# 086# time_filter 片側蓄積バグ修正 + 085# セッション考察

**日付**: 2026-02-17  
**前提**: 085# 実装後の fill_test 再開（run_id: `1771258270_75e34201`）で発生した SAFE_STOP の原因分析と修正  
**コミット**: `85a160d5d` (fix: 086# time_filter 片側蓄積バグ修正)

---

## エグゼクティブサマリ

085# の改善を投入して fill_test を再開したところ、**11 サイクル・約 2.1 時間で SAFE_STOP** が発生した。
根本原因は time_filter の side 切り替えロジックに潜在していたバグで、**片側ポジション蓄積**を引き起こしていた。
加えて、セッション中にモデル（SkipGate）が一度も取引を拒否しなかった理由を値動き・スプレッド・P(AS) の観点から考察する。

---

## 1. バグ: time_filter 片側蓄積 (086#)

### 1.1 発生現象

| 項目 | 値 |
|------|-----|
| 発生サイクル | 512 (最後のサイクル) |
| 発生時刻 | 2026-02-16 07:26 JST (UTC 22h) |
| 症状 | BTC double buy → JPY 残高枯渇 → SAFE_STOP |
| 残高推移 | JPY: 十分 → 2,794 / BTC: 0.001 → 0.002 |

### 1.2 根本原因

time_filter には side 別のスキップ時間が設定されている:

```yaml
skip_utc_hours_buy:  [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23]  # 16時間
skip_utc_hours_sell: [3, 4, 5, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]    # 16時間
```

| カテゴリ | 該当 UTC 時間帯 | 時間数 |
|----------|------------------|--------|
| **両方アクティブ** | 0, 6, 7, 20 | **4h** |
| **buy のみ** | 3, 4, 5, 22 | 4h |
| **sell のみ** | 1, 2, 10, 11 | 4h |
| **両方ブロック** | 8, 9, 12-19, 21, 23 | **12h** |

**バグの発火条件**: 片側のみアクティブな 8 時間帯で、`_last_side` が既にアクティブ側と一致している場合、交代ロジックが同じ side を連続実行する。

#### 具体的な発火シーケンス (UTC 22h)

```
状態: _last_side = buy (cycle 511 で buy FILL)
      UTC 22h → sell はブロック (skip_utc_hours_sell に 22 含む)
               buy はアクティブ

1. next_side = sell (_last_side=buy の反転)
2. sell がブロック → alt_side = buy
3. alt_side == _last_side → 🐛 buy を再実行
4. cycle 511: buy FILL (+5.49bps)
5. cycle 512: buy FILL (-7.37bps) ← double buy
6. BTC = 0.002, JPY = 2,794 → SAFE_STOP
```

### 1.3 影響範囲

8 つの脆弱時間帯すべてで発火可能:

| UTC | アクティブ side | _last_side が同じなら | リスク |
|-----|-----------------|------------------------|--------|
| 1 | sell のみ | sell → double sell | BTC 枯渇 |
| 2 | sell のみ | sell → double sell | BTC 枯渇 |
| 3 | buy のみ | buy → double buy | JPY 枯渇 |
| 4 | buy のみ | buy → double buy | JPY 枯渇 |
| 5 | buy のみ | buy → double buy | JPY 枯渇 |
| 10 | sell のみ | sell → double sell | BTC 枯渇 |
| 11 | sell のみ | sell → double sell | BTC 枯渇 |
| **22** | **buy のみ** | **buy → double buy** | **JPY 枯渇 ← 今回** |

### 1.4 修正内容

`scripts/v460/run_fill_test.py` L1607-1620:

```python
# 086# Bug: alt_side が _last_side と同じ場合、片側蓄積が発生する
if alt_side == self._last_side:
    logger.info(
        f"[time_filter] {next_side} filtered at UTC {utc_h}h, "
        f"alt={alt_side} would repeat last side → treating as both-filtered "
        f"(086# 片側蓄積防止)"
    )
    if not self._in_time_filter:
        self._in_time_filter = True
        self._last_heartbeat_time = time.time()
    await asyncio.sleep(self.config.cycle_interval_sec)
    continue
```

**方針**: `alt_side == _last_side` の場合は **両方ブロックと同じ扱い** にして待機。  
side の交互実行 (buy ↔ sell) という基本契約を破らない。

### 1.5 テスト

`tests/unit/v460/test_regime_detector.py` に `TestBug086TimeFilterPositionAccumulation` を追加。  
ソースコード内にガード条件 `alt_side == self._last_side` が存在することを検査。

### 1.6 修正後の動作確認

再起動後のログで正常動作を確認:

```
[time_filter] buy filtered at UTC 2h, alt=sell would repeat last side
  → treating as both-filtered (086# 片側蓄積防止)
```

---

## 2. 085# セッション分析

### 2.1 全サイクルデータ

run_id `1771258270_75e34201` の 11 サイクル (cycle 502-512):

| Cycle | Side | Result | PnL (bps) | Wait (s) | Spread (JPY) | P(AS) | 備考 |
|-------|------|--------|-----------|----------|---------------|-------|------|
| 502 | buy | FILL | -0.16 | 272.7 | 2,723 | 0.545 | |
| 503 | sell | FILL | +0.35 | 89.6 | 1,493 | 0.545 | |
| 504 | buy | MISS | — | 303.4 | 1,627 | 0.499 | timeout |
| 505 | buy | MISS | — | 304.2 | 1,576 | 0.510 | BTC 不足で sell スキップ |
| 506 | buy | FILL | -0.62 | 88.8 | 1,903 | 0.496 | |
| 507 | sell | FILL | -5.65 | 263.9 | 4,002 | 0.510 | early_exit |
| 508 | buy | FILL | -3.13 | 24.7 | 4,328 | 0.513 | rapid fill |
| 509 | sell | FILL | +2.31 | 132.6 | 4,002 | 0.510 | |
| 510 | buy | MISS | — | 300.1 | 3,246 | 0.492 | timeout |
| 511 | buy | FILL | +5.49 | 5.6 | 3,007 | 0.523 | time_filter 離脱直後 |
| 512 | buy | FILL | -7.37 | 18.0 | 4,677 | 0.527 | 🐛 double buy |

### 2.2 統計サマリ

| 指標 | 値 |
|------|-----|
| FILL 率 | 8/11 (72.7%) |
| 平均 PnL (FILL) | **-1.10 bps** |
| P(AS) 平均 | 0.515 (range: 0.492 – 0.545) |
| SkipGate SKIP 回数 | **0 / 11** |
| 取引時間帯 | UTC 20h (9 cycles), UTC 22h (2 cycles) |
| 価格推移 | 10,428,008 → 10,568,253 JPY (**+1.34%**, 上昇トレンド) |
| スプレッド推移 | 1,493 → 4,677 JPY (**3.1× 拡大**) |

---

## 3. 考察: なぜ SkipGate は一度も拒否しなかったか

### 3.1 P(AS) の絶対値が低い

全サイクルで P(AS) ∈ [0.492, 0.545] であり、閾値 (buy: 0.65, sell: 0.60) を大きく下回っている。  
P(AS) = 0.515 は「AS 発生確率 51.5%」を意味し、コインフリップに近い。  
これは SkipGate モデルが**この時間帯・市場条件下で明確な AS シグナルを検知していない**ことを示す。

### 3.2 SkipGate は「行くべきでない時」を拒否するツール

SkipGate の設計意図は「AS が高確率で発生する条件を検知して SKIP」することであり、  
「利益が出る方向を予測して GO」するものではない。  
P(AS) ≈ 0.5 は「AS リスクは通常水準」という判定であり、PASS は正常な挙動。

### 3.3 では何が損失を生んだか

SkipGate が PASS したにもかかわらず平均 PnL = -1.10 bps の理由:

| 要因 | 説明 |
|------|------|
| **スプレッド急拡大** | 1,493 → 4,677 JPY (cycle 503→512)。スプレッド拡大局面では maker の edge が減少 |
| **上昇トレンド (+1.34%)** | 一方向のトレンドは buy FILL → 直後に price 上昇で利益に見えるが、sell 側で逆方向に振られる。cycle 507 sell の -5.65bps が典型 |
| **rapid fill (queue_wait < 25s)** | cycle 508 (24.7s, -3.13bps), 512 (18.0s, -7.37bps) — 速い約定は AS の兆候 |
| **time_filter 窓の狭さ** | 実効取引可能時間 = 4h/day(両方アクティブ) → 市場選択の余地が極めて小さい |

### 3.4 time_filter が支配的なボトルネック

time_filter の coverage 分析:

```
24時間中:
  ├── 両方アクティブ:  4h (16.7%) ← ここでしか sell/buy 交互不可
  ├── 片側のみ:       8h (33.3%) ← 086# バグの温床
  └── 両方ブロック:   12h (50.0%) ← 取引不可
```

**日に 4 時間しか安全に取引できない** のは、統計的検出力を速やかに蓄積する上で深刻な制約。  
現在のペースだと G1.1 gate に必要な n≈700 clean records に到達するまでに顕著な時間がかかる。

---

## 4. 値動き考察

### 4.1 セッション中の価格挙動

- **期間**: 2026-02-16 05:17 JST → 07:26 JST (約 2.1 時間)
- **始値**: 10,428,008 JPY / **終値**: 10,568,253 JPY
- **変動**: **+140,245 JPY (+1.34%)**
- **性質**: 一方向の上昇トレンド

### 4.2 トレンドと maker 戦略の相性

maker 戦略 (buy/sell 交互) はレンジ相場で最も機能する。  
+1.34% の一方向トレンドでは:

- **buy 側**: 板に指値→価格上昇→FILL→含み益（正のスリッページ）
  - 実際: cycle 511 で +5.49bps (time_filter 離脱直後、買い有利)
- **sell 側**: 板に指値→価格上昇→FILL しにくい or FILL 後に逆行
  - 実際: cycle 507 で -5.65bps (sell FILL 後の上昇トレンドで損)

しかし **double buy (086# バグ)** がこのトレンド相場で発生したことで、  
本来有利なはずの buy 側でも -7.37bps の大損が出た。  
buy が FILL → 価格が高い位置でさらに buy → spread が拡大した状態で不利な価格を掴む、  
という最悪のシナリオが実現した。

### 4.3 スプレッド拡大の影響

| 時期 | スプレッド | 状態 |
|------|-----------|------|
| 初期 (cycle 502-505) | 1,493-2,723 JPY | 通常 |
| 中盤 (cycle 506-509) | 1,903-4,328 JPY | 拡大中 |
| 終盤 (cycle 510-512) | 3,007-4,677 JPY | 高止まり |

スプレッドが 3,000 JPY を超えると、0.05 (5%) の offset では maker edge がほぼ消失する。  
`spread_offset_ratio = 0.05` は spread ≈ 2,000 JPY 前提で設計されており、  
4,000 JPY 超のスプレッドでは不適切。

---

## 5. Codex レビュー論点

### 5.1 086# 修正の妥当性

- [x] ガード条件 `alt_side == self._last_side` は必要十分か？
- [x] 「両方ブロック扱い」は保守的だが、片側だけでも利益が出る場合を見逃さないか？
  - → **意図的にトレードオフ**: side 交互実行の保証が、片側連続実行のリスクに優先
- [ ] テストがソースコード検査のみで、実際の動作テスト（モック・統合）がない
  - → 非同期ループのモックが困難なため源流検査を採用。改善の余地あり

### 5.2 time_filter 設計の再考

```
有効取引時間: 4h/24h = 16.7%
```

この extreme な制約は以下の問題を引き起こす:

1. **データ収集速度**: G1.1 gate の統計的有意性に到達するまでの日数が増大
2. **過学習リスク**: 4 時間の市場条件に特化した time_filter は、条件変化に脆弱
3. **SkipGate との冗長性**: time_filter が 83.3% をブロックしている状態で SkipGate の貢献が測定困難

**検討すべき方向性**:
- time_filter を緩和し、SkipGate に AS 判断を委ねる (ML ベースの柔軟なフィルタリング)
- time_filter を「明らかに危険な時間帯」のみに限定 (現 16 時間 → 8-10 時間)
- spread 条件を追加して、wide spread 時のみブロック（時間帯固定ではなく動的判断）

### 5.3 スプレッド adaptive な制御

現在は `spread_offset_ratio` が固定 (0.05) だが、  
spread が 2× 以上に拡大した場合の動的調整が未実装。  
`param_adapter` が一部この役割を担うが、スプレッド変動への直接的な応答はない。

### 5.4 収益性の現状評価

| 指標 | 085# セッション | 全体 (clean n=363) |
|------|-------------------|---------------------|
| 平均 PnL | -1.10 bps | (要確認) |
| SkipGate SKIP 率 | 0% | (要確認) |
| FILL 率 | 72.7% | (要確認) |

085# セッション単体では**赤字**。ただし n=8 (FILL のみ) は統計的に有意とは言えない。  
086# バグの cycle 512 (-7.37bps) を除外すると平均 PnL = -0.99 bps で依然赤字。

---

## 6. 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | time_filter 片側蓄積ガード追加 (L1607-1620) |
| `tests/unit/v460/test_regime_detector.py` | `TestBug086TimeFilterPositionAccumulation` 追加 |

---

## 7. 次のアクション

| 優先度 | 項目 | 根拠 |
|--------|------|------|
| **HIGH** | データ蓄積継続 | n=512 → 700 目標、086# 修正済みで安全に継続可能 |
| **MID** | time_filter 緩和の検討 | 4h/day active は厳しすぎる。SkipGate との役割分担再整理 |
| **MID** | spread 動的制御の導入 | 4,000 JPY 超の spread で offset 0.05 は edge 消失 |
| **LOW** | 086# の統合テスト追加 | 現在はソース検査のみ、モックベースの動作テスト追加 |
