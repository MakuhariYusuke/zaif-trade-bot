# 542# Ceiling 0.25→0.30 + 残存 Identity 段分析 + メモリリーク監査

- **日付**: 2026-03-22
- **前提**: 540# §2 spread_adapt 主犯特定、541# §4 narrow boost 全サイクル発火確認
- **コミット**: `db918a75c`

---

## §1 ceiling 0.25→0.30 の根拠

540# §2.1 で spread_adapt が offset を ~0.15→0.30 に倍増させ、ceiling 0.25 が 76.7% のサイクルで情報を切り捨てていることを発見。541# §4 で `narrow_spread_bps=2.5` がほぼ毎サイクル発火していることを確認。

- `max_offset_ratio=0.30`（中間段の探索上限）と `offset_ceiling_ratio=0.25`（最終 clamp）の 5% ギャップが情報損失の直接原因
- ceiling を 0.30 に引き上げることで、中間段の出力がそのまま executor に渡る
- `hour_ceiling_mult` による危険時間帯の追加保護は引き続き機能（0.30×2.0=0.60）

### YAML 変更

```yaml
offset_ceiling_ratio_buy: 0.30    # 旧 0.25
offset_ceiling_ratio_sell: 0.30   # 旧 0.25
```

---

## §2 残存 identity 段（6 段）の分析結果

541# で 5 段（kyle, amihud, imb_risk, buy_as_guard, ffd）のスキップ最適化を実施。残り 6 段を精査した結果、**全てスキップ不可**:

| Stage | 理由 |
|-------|------|
| regime | 実データで +0.007 中央値寄与。常に発火する正当なステージ |
| vol_guard | `volatility_guard_enabled=True`。velocity/VPIN に基づく保護で実際に発火する |
| cross_venue | sidecar 稼働時に BitFlyer lead-lag hint を使用。veto 機能を含む重要な防御 |
| sell_hour | **空 dict ではなかった**: YAML に 9 時間帯分の設定あり。実際に発火する |
| loss_boost | 損失イベント後 10-15 分間指数減衰で保護。正当な防御メカニズム |
| final | 終端マーカー。必須 |

Pre-order pipeline 14 段の最終整理:
- **活性 3 段**: base, as_shift, spread_adapt（主要寄与）
- **条件的活性 6 段**: regime, vol_guard, cross_venue, sell_hour, loss_boost, final
- **disabled スキップ 5 段**: kyle, amihud, imb_risk, buy_as_guard, ffd（541# 最適化済み）

---

## §3 メモリリーク・重複計算の監査

- `_last_offset_stages`: 毎 cycle 上書き（append ではない）→ リークなし
- `_inv_fill_history`: `deque(maxlen=N)` 制限あり → 安全
- offset_pipeline: `_exec_stages` はローカルスコープ → リークなし
- VG/VG-supplement 重複: **意図的な設計**（VG supplement は sell-only fallback、`not last_vg_triggered` ガードで二重適用を防止）
- `_emit_vg_event()`: 同期 JSONL 書き込み。VG 発火頻度が低い（velocity > 50bps/s 条件）ため実害なし

---

## §4 陳腐化テスト修正

- `test_405_offset_ceiling_pipeline.py::test_buy_capped_by_buy_ceiling`: 523# で maker_price 中間 ceiling を撤廃し offset_pipeline に一本化した変更にテストが追従していなかった。assertion を `<= 0.20` → `<= 0.30`（intermediate cap）に修正
