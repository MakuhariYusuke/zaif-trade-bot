# 094# stale order 検出 & cancel-replace

| 項目 | 内容 |
|------|------|
| 日付 | 2025-06-17 |
| 根拠 | 084# 盲点C 代替案 "cancel-replace (stale 化した注文のみ再見積)", timeout 300s 中の機会損失削減 |
| 対象 | `run_fill_test.py`, `fill_test.yaml`, `fill_quality.py`, `test_094_stale_order.py` |
| 方針 | 既存ポーリングループ内への最小侵襲追加。新規モジュール不要。|

---

## §1 背景・課題

### 1.1 問題: 価格乖離した注文の滞留
- 注文を板に載せた後、市場価格が離れると成立見込みがなくなる
- 現行: `order_timeout_sec=300s` まで無条件で待ち続ける → 最大 5 分の機会損失
- 特に BTC のトレンド発生時に顕著 (買い注文 → 価格上昇 → 取り残される)

### 1.2 084# での言及
> 代替案: **短 wait fill の AS 防衛** (fast_fill_defense の閾値調整) + **cancel-replace** (stale 化した注文のみ再見積)

- timeout 短縮 (083# §4.1-4) は Q5 (long wait, best PnL) を殺すため **不採用**
- cancel-replace なら有望な位置の注文はそのまま維持し、乖離した注文だけ再発注

---

## §2 実装内容

### 2.1 Config フィールド

```python
stale_order_enabled: bool = False       # 有効/無効
stale_check_after_sec: float = 30.0     # チェック開始までの猶予
stale_drift_bps: float = 5.0            # 乖離閾値 (bps)
stale_max_reprice: int = 2              # 1 サイクル内の最大再発注回数
stale_cooldown_sec: float = 10.0        # 再発注後のチェック猶予
```

### 2.2 YAML セクション

```yaml
stale_order:
  enabled: true
  check_after_sec: 30.0
  drift_bps: 5.0
  max_reprice: 2
  cooldown_sec: 10.0
```

### 2.3 FillRecord 追加フィールド

```python
reprice_count: int = 0   # 1 サイクル内で再発注した回数
```

### 2.4 ポーリングループ内ロジック

既存のステータス確認ループの **末尾** (各 poll イテレーション後) に検出ロジックを追加:

```
条件:
  - stale_order_enabled
  - not filled (まだ約定していない)
  - elapsed >= check_after_sec (十分な時間が経過)
  - reprice_count < max_reprice (上限未達)
  - cooldown 経過済み

検出:
  1. 現在の mid price を取得
  2. 発注時 mid からの乖離 (bps) を算出
  3. 乖離方向チェック:
     - buy: mid 上昇 → 注文が取り残される → stale
     - sell: mid 下降 → 注文が取り残される → stale
     - 逆方向 (注文に近づく) は stale とみなさない

cancel-replace:
  1. cancel_order(既存注文)
     - cancel 失敗 → 約定済みの可能性をチェック
  2. _compute_maker_price() で新価格算出
  3. place_order(新価格)
  4. mid_at_order / order / reprice_count を更新
```

---

## §3 設計判断

### Q1: check_after_sec=30s は早すぎないか?
- 短すぎると微小なノイズで不要な reprice が発生
- 30s は poll_interval(5s) の 6 倍で十分な観測窓
- drift_bps=5.0 と併用で誤発動リスクは限定的

### Q2: 方向チェック (is_drifting_away) は必要か?
- **必須**。price が注文に近づいている場合は約定チャンスが上がっている
- buy で mid 下降 → best_bid に近づく → 約定しやすくなっている → reprice 不要
- 方向無視だと "もうすぐ約定する注文" をキャンセルしてしまう

### Q3: max_reprice=2 の根拠は?
- 強いトレンド時に無限 reprice すると追従し続けて AS リスク増大
- 2 回 reprice してもダメなら、市場状況自体が不利 → timeout まで待つのが安全
- 2 回 × cooldown 10s = 20s の追加使用で、元サイクルの大半はそのまま

### Q4: cancel 成功後 place 失敗した場合は?
- `cancel_reason_poll = "stale_reprice_failed"` を記録してループ脱出
- 注文なしで timeout を待つよりも、次サイクルに進む方が合理的

---

## §4 安全設計

| リスク | 対策 |
|--------|------|
| 連続 reprice でレート制限 | `max_reprice=2` + `cooldown_sec=10` |
| 約定直前に cancel | cancel 失敗 → get_order_status で filled 確認 |
| reprice 後に AS 悪化 | `_compute_maker_price` で spread_adaptive/sell_guard 等の既存防御が適用 |
| API 障害で mid 取得失敗 | try/except で非致命的処理、check スキップ |

---

## §5 テスト

[test_094_stale_order.py](../../tests/unit/v460/test_094_stale_order.py): **30 テスト**
- A. Config フィールド (6)
- B. YAML パース (4)
- C. FillRecord.reprice_count (4)
- D. ロジック構造 (6)
- E. 発動条件の整合 (6)
- F. buy/sell 方向性 (4)

全 781 テスト PASS (v460 スコープ)

---

## §6 期待効果

| 指標 | 期待 | メモ |
|------|------|------|
| timeout キャンセル率 | ↓ | 取り残された注文を再発注 → fill 率向上 |
| 無駄な待機時間 | ↓ | 5 分 timeout を消費する前に再配置 |
| AS 率 | → or ↓ | reprice 時も既存の offset 防御が適用される |
| PnL | ↑ | 市場追従 + 機会損失削減 |
