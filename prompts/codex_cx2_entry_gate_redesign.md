# Codex Prompt CX2: entry_gate selectivity 再設計

## 背景
entry_gate は 100% suppressed の dead layer (706# §3, 708# V2 で確認済)。

原因の連鎖:
1. `spread_as_guard` が 99.6% の fills に `-0.747` bps penalty を適用
2. regime_guard_overrides が追加で EV premium 0.3 bps を加算
3. CalibrationMap の base EV が既に `-1.91` 付近に集中
4. 結果: entry_gate_ev は常時 `-1.8` 以下 → threshold `-0.5` に永遠に到達不可
5. `entry_gate_guard.py` の `should_suppress_block()` で auto-disable が side-aware 判定より先に発火 (dead code)

## タスク

### Phase 1: コード修正 (dead code 解消)

1. `scripts/v460/lib/entry_gate_guard.py` の `should_suppress_block()`:

```python
# BEFORE (dead code):
def should_suppress_block(self, *, ev, regime, side):
    if self._state.auto_disabled:  # ← L48: これが先に True を返す
        return True
    ...
    if side == "buy" and ev >= threshold:  # ← L68: 到達しない
        return True

# AFTER:
def should_suppress_block(self, *, ev, regime, side):
    if side == "buy" and ev >= threshold:  # side-aware を先に評価
        return True
    if side == "sell":
        return False  # sell は entry_gate 判定に委ねる
    if self._state.auto_disabled:  # 上記に該当しない場合のみ
        return True
```

2. 対応するテスト `tests/unit/v460/test_690_entry_gate_guard.py` を更新

### Phase 2: EV 到達可能性の改善

3. `entry_gate_buy_suppress_ev_threshold` の値を分析:
   - 現行: `-0.5` (到達率 0%)
   - `-1.5` なら 12.3% が到達 (705# deep dive)
   - `-2.0` なら 100% が到達
   - **提案**: `-1.8` をテスト候補に (CalibrationMap median -1.91 付近)

4. SAG penalty の影響分離:
   - `spread_as_guard.ev_penalty_bps` が 0.5 で固定
   - `regime_guard_overrides` の `ev_threshold_premium_bps` が全 regime で 0.3
   - **合計 -0.8 bps の定数ペナルティ**が CalibrationMap base EV に加算
   - ペナルティなしの EV 分布 vs ありの EV 分布を比較

### Phase 3: CalibrationMap 診断

5. `configs/v460/` 配下の calibration_map ファイルを特定
6. map の EV 分布を分析:
   - spread bucket 別 EV
   - regime 別 EV
   - side 別 EV
7. EV が全域で負になる構造的原因を特定
8. map 更新の現実的方法を提案

## 成果物
- Phase 1: コード修正 + テスト更新（即コミット可能）
- Phase 2-3: `docs/v460/708_entry_gate_redesign.md` に分析レポート
- YAML 変更案 (threshold 値の候補を 2-3 パターン)

## 制約
- `git commit --no-verify` を使用
- テストは `python -m pytest tests/unit/v460/ -x --tb=short` で確認
- Phase 1 のコード修正は必ず行う。Phase 2-3 は分析のみ（YAML 変更は提案まで）
