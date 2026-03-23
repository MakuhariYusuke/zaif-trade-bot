# Codex Prompt: 587# Telemetry Parity & Dead Config Cleanup

## Context

585# / 586# レビューで発見された構造的負債を解消する。
いずれも「実害はまだないが、放置すると Phase 6 以降の A/B 評価や eDRC 有効化で必ず踏む地雷」。

**共通ルール**:
- テスト: `python -m pytest tests/unit/v460/ -x --tb=short` で既存テスト全PASS維持
- 型検査: `mypy --config-file mypy.ini` でエラー増加なし
- コミット: `git commit --no-verify -m "..."` (pre-commit フック回避)
- `git add` は対象ファイル個別指定、`git add .` 禁止
- ドキュメント番号は `587#` を使用

---

## Task A: `execution_additive_enabled` 三重断線の修復

### 問題
`execution_additive_enabled` は三箇所で断線している:
1. `fill_cycle_executor.py` L1356-1410 の `_build_fill_record()` 呼び出しで引数を渡していない
2. `FillRecord` (ztb/metrics/fill_quality.py L42-248) にフィールドが存在しない → `_sanitize_fill_record_fields` でサイレントにドロップ
3. 結果として JSONL に記録されず、`analyze_fill_logs.py` の A/B 分類が常に None

### 修正方針
`execution_additive_enabled` を `experimental_additive_pipeline` に統一して廃止する方が望ましいが、後方互換のため以下のアプローチで修復する:

1. **`ztb/metrics/fill_quality.py`**: `FillRecord` に `execution_additive_enabled: bool | None = None` フィールドを追加
2. **`scripts/v460/lib/fill_cycle_executor.py`**: `_build_fill_record()` 呼び出しに `execution_additive_enabled=self.config.experimental_additive_pipeline` を追加

これにより JSONL に additive/multiplicative のラベルが記録され、A/B 分析が可能になる。

### 検証
- `python -m pytest tests/unit/v460/test_582_additive_pipeline.py tests/unit/v460/test_585_multiplicative_pipeline.py -x --tb=short`
- `FillRecord(execution_additive_enabled=True)` がシリアライズ/デシリアライズで保持されることを確認するテストを追加

---

## Task B: `additive_base_bps` dead config 削除

### 問題
`additive_base_bps` は以下に定義されるが、ランタイムで一切使用されていない:
- `scripts/v460/lib/fill_config.py` L364: dataclass フィールド
- `scripts/v460/lib/fill_config_parser.py` L193: YAML パース対象
- `configs/v460/fill_test.yaml` L702: YAML 定義

offset_pipeline.py, multiplicative_pipeline.py のどちらにも参照がない。

### 修正
3 箇所とも **削除** する。将来必要になったら再追加すれば良い。YAGNI 原則。

### 検証
- `python -m pytest tests/unit/v460/ -x --tb=short` (全テストPASS)
- `grep -r "additive_base_bps" scripts/ ztb/ configs/ tests/` で参照が残っていないことを確認

---

## Task C: eDRC 入力値の硬クリップ (Winsorization) 追加

### 問題
`resolve_offset_ceiling()` 内の eDRC 計算 `C_base * exp(alpha * sigma + beta * OFI)` において、
`sigma` と `adverse_ofi` は `get_robust_inputs(side)` (asymmetric EMA + median filter) で平滑化済みだが、
**入力値の硬クリップ（Winsorization）が無い**。Fat Tail スパイク時に平滑化だけでは不十分な場合があり、
`exp()` が暴走して `hard_cap` に頻繁に到達するリスクがある。

現状: 呼び出し元（offset_pipeline.py L268, multiplicative_pipeline.py L234）は既に `get_robust_inputs()` を使用しており、平滑化は適用済み。

### 修正
`scripts/v460/lib/fill_config.py` の `resolve_offset_ceiling()` メソッド内、`exp()` 呼び出しの前に入力クリップを追加:

```python
# Before:
ceiling_dynamic = self.edrc_c_base * exp(
    self.edrc_alpha * sigma + self.edrc_beta * adverse_ofi
)

# After:
_sigma_cap = 5.0   # 1-min vol の合理的上限
_ofi_cap = 50.0    # 片側 OFI の合理的上限
sigma = min(sigma, _sigma_cap)
adverse_ofi = min(adverse_ofi, _ofi_cap)
ceiling_dynamic = self.edrc_c_base * exp(
    self.edrc_alpha * sigma + self.edrc_beta * adverse_ofi
)
```

将来的には `_sigma_cap` / `_ofi_cap` を FillConfig のフィールドに昇格させてもよいが、
現段階では固定値で十分（edrc_alpha=0.0, edrc_beta=0.0 で実質無効のため）。

### 検証
- 既存テスト全PASS
- 新テスト追加: `resolve_offset_ceiling()` に極端な sigma/adverse_ofi を渡しても ceiling が hard_cap を超えないことを確認

---

## Task D: `config_hot_reload.py` — Phase 6 フィールドの hot-reload 対応

### 問題
以下のフィールドはすべて hot-reload スコープ外:
- `experimental_additive_pipeline`
- `edrc_alpha`, `edrc_beta`, `edrc_c_base`, `edrc_hard_cap`
- `entry_gate_enabled`, `entry_gate_*` (10フィールド)

584# Phase 6 で A/B 切替を行う際、プロセス再起動が必要になる。

### 修正
`scripts/v460/lib/config_hot_reload.py` の `_HOT_RELOADABLE_FIELDS` に以下を追加:

```python
# Phase 6 A/B control
"experimental_additive_pipeline",
# eDRC tuning
"edrc_alpha",
"edrc_beta",
"edrc_c_base",
"edrc_hard_cap",
# Entry Gate
"entry_gate_enabled",
```

eDRC のパラメータ変更は next cycle から反映されるため、hot-reload で安全に切替可能。
`entry_gate_*` のうちパス系 (`entry_gate_calibration_map_path`) はファイルロードを伴うため hot-reload 対象には含めない。

### 検証
- hot-reload のユニットテストがあれば、新フィールドが対象に含まれることを確認
- なければ `_HOT_RELOADABLE_FIELDS` に存在することを assert するテストを追加

---

## Task E: `execution_additive_enabled` / `experimental_additive_pipeline` 二枚看板の整理

### 問題
- `experimental_additive_pipeline` が **実際のロジック分岐** を駆動 (offset_pipeline.py, fill_config.py)
- `execution_additive_enabled` は **テレメトリラベル** の意図だが Task A までは死んでいた
- 両方が config に存在し、どちらが何をするのか混乱する

### 修正
1. `fill_config.py` の `execution_additive_enabled` フィールドのコメントを更新:
   ```python
   execution_additive_enabled: bool = False  # DEPRECATED: telemetry 記録用。ロジック分岐は experimental_additive_pipeline を使用
   ```
2. `fill_config_parser.py` のパース箇所で、`execution_additive_enabled` が `experimental_additive_pipeline` と異なる値に設定された場合に warning ログを出す:
   ```python
   if cfg.execution_additive_enabled != cfg.experimental_additive_pipeline:
       logger.warning(
           "execution_additive_enabled=%s != experimental_additive_pipeline=%s — "
           "ロジック分岐は experimental_additive_pipeline が優先されます",
           cfg.execution_additive_enabled,
           cfg.experimental_additive_pipeline,
       )
   ```

### 検証
- 既存テスト全PASS

---

## 実行順序

**依存関係**: Task A → Task E (A で FillRecord に追加した後に E の deprecation 注記)

推奨順: B → C → D → A → E

## コミット

```
git add ztb/metrics/fill_quality.py scripts/v460/lib/fill_cycle_executor.py \
  scripts/v460/lib/fill_config.py scripts/v460/lib/fill_config_parser.py \
  scripts/v460/lib/offset_pipeline.py scripts/v460/lib/multiplicative_pipeline.py \
  scripts/v460/lib/config_hot_reload.py configs/v460/fill_test.yaml \
  tests/unit/v460/
git diff --cached --stat
git commit --no-verify -m "refactor: 587# telemetry parity, dead config cleanup, eDRC winsorization, hot-reload scope"
```
