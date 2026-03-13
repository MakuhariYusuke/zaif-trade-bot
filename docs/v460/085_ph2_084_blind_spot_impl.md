# 085# 084 盲点指摘の実装

**日付**: 2025-02-16  
**前提**: 084# で特定した 8 盲点 (A–H) のうち、コード対応可能な項目を実装

---

## 実装サマリ

| 盲点 | 概要 | 対応 | ファイル |
|------|------|------|----------|
| A | SkipGate P(AS) 不透明 | `as_probability`, `threshold_used` を直接記録 | skip_gate.py, fill_quality.py, run_fill_test.py |
| B | param_adapter 負のスパイラル | `high_as & low_fill → hold` に変更 | param_adapter.py |
| D | AS_raw vs AS_deadzone 不可視 | monitor に並行表示追加 | monitor_fill_test.py |
| F | api_error 28% 放置 | 指数バックオフ + 非リトライ分類拡張 + max_retries 2 | run_fill_test.py |
| G | run_id 分離評価なし | `--run-id` / `--git-sha` CLI フィルタ追加 | run_075_verification.py |

---

## 1. SkipGate P(AS) 直接記録 (盲点 A)

### 問題
SkipGate の AS モード判定で使用される `P(AS)` 確率値が FillRecord に記録されず、
後続分析で「なぜ SKIP/PASS したか」のトレーサビリティが不足。

### 変更

**`scripts/v460/ml/skip_gate.py`**:
- `SkipDecision` dataclass に `as_probability: Optional[float]` と `threshold_used: Optional[float]` を追加
- `evaluate()` メソッドで AS モード判定時に `pred_prob` と実際に適用された `threshold` を設定

**`ztb/metrics/fill_quality.py`**:
- `FillRecord` に `skip_gate_as_prob` と `skip_gate_threshold_used` フィールドを追加
- `from_dict()` は未知フィールドを無視するため後方互換性あり

**`scripts/v460/run_fill_test.py`**:
- `evaluate()` 結果から新フィールドを抽出して FillRecord に渡す
- 通常レコードと skip_gate SKIP 時の早期リターンの両方で記録

---

## 2. param_adapter デッドロック防止 (盲点 B)

### 問題
`high_as AND low_fill` 時に `decrease` (offset 縮小) → fill_rate さらに低下 → 
負のスパイラル。offset を縮小すると maker 価格が mid に接近し約定しにくくなる。

### 変更

**`scripts/v460/lib/param_adapter.py`**:
```python
# 旧: action="decrease", reason="AS 回避優先で offset 縮小"
# 新: action="hold", reason="デッドロック防止のため hold (084#)"
```

新ロジック: 両方異常時は offset を変更せず、他の対策 (time_filter, SkipGate) に委ねる。

**`tests/unit/v460/test_param_adapter.py`**:
- `test_both_abnormal_as_priority` → `test_both_abnormal_hold_deadlock_prevention` に改名
- `action == "hold"`, `new_offset == 0.05` (変更なし), `not result.changed` を検証

---

## 3. AS_raw 並行表示 (盲点 D)

### 問題
monitor が `AS(deadzone)` のみ表示し、`AS(raw)` との差分が見えない。
deadzone の効力が不明確。

### 変更

**`scripts/v460/monitor_fill_test.py`**:
- 「補足統計」セクションに `AS(deadzone)`, `AS(raw)`, `deadzone差分: +X.Xpp マスキング` を追加表示
- 条件: `metrics.as_coverage > 0 AND metrics.as_raw_coverage > 0`

---

## 4. api_error リトライ強化 (盲点 F)

### 問題
34/122 キャンセルが api_error (28%)。`max_order_retries=1` (2回試行) + flat 2s delay。

### 変更

**`scripts/v460/run_fill_test.py`**:
- `max_order_retries`: 1 → **2** (計3回試行)
- リトライ待機: flat 2s → **指数バックオフ** (`2s × 2^attempt`: 2s → 4s)
- **Rate-limit 検出**: エラーに "rate", "limit", "too many" → `max(backoff, 5.0)` に延長
- **非リトライ対象拡張**: `{"insufficient_funds", "post_only_reject", "minimum_size"}` のセットに変更
  - `post_only_reject`: 価格がスプレッド交差済み → リトライしても同結果
  - `minimum_size`: 数量固定のためリトライ不要

**`tests/unit/v460/test_regime_detector.py`**:
- ソースコード検査テストを新パターン (`_non_retriable`) に適合

---

## 5. run_id 分離フィルタ (盲点 G)

### 問題
検証スクリプトが全 run_id を混合して評価。設定変更前後の比較が不可能。

### 変更

**`scripts/v460/ml/run_075_verification.py`**:
- `argparse` 追加: `--run-id` (複数指定可), `--git-sha` (前方一致、複数指定可)
- `load_clean_filled()` に `run_ids` / `git_shas` パラメータ追加
- フィルタ前後のレコード数をログ出力

```bash
# 使用例
python scripts/v460/ml/run_075_verification.py --run-id fill_v460_20250215
python scripts/v460/ml/run_075_verification.py --git-sha 1187c9c abc1234
```

---

## テスト結果

- `tests/unit/v460/`: **666 passed** (全パス)
- 修正対象テスト:
  - `test_both_abnormal_hold_deadlock_prevention`: PASSED (param_adapter)
  - `test_source_has_insufficient_funds_break`: PASSED (api_error source check)

---

## 未対応項目

| 盲点 | 概要 | 状態 |
|------|------|------|
| C | spread_adaptive 2.0× 常時適用 | パラメータ探索待ち (AB テスト必要) |
| E | time_filter 誤差伝搬 | 別タスクで対応予定 |
| H | 検出力分析 (統計パワー不足) | データ蓄積待ち |
