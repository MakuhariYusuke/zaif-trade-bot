# 612# テスト実行時間最適化 Wave 1

## 背景

- `python -m pytest tests/ -x --tb=short` が重く、VS Code 側のメモリ圧迫と相性が悪い。
- 現行設定では `pytest.ini` / `pyproject.toml` の両方で coverage がデフォルト有効になっており、ローカル開発時の実行コストを押し上げていた。
- `tests/unit/v460/` では `FillTestConfig.from_yaml()` を同種 YAML に対して何度も繰り返すテストがあり、shared cache の横展開余地があった。

## 今回の方針

1. **coverage を dev デフォルトから外す**
   - `pytest.ini` と `pyproject.toml` の pytest addopts から `--cov*` を除去
   - coverage は CI / 明示コマンドでのみ有効化する前提へ寄せる
2. **shared config cache を追加する**
   - `tests/unit/v460/_yaml_test_helpers.py` に `FillTestConfig` の path/text ベース cache を追加
   - 既存の `load_yaml_mapping()` / `@lru_cache` パターンを `FillTestConfig` まで拡張
3. **高頻度 YAML test に横展開する**
   - `test_fill_test_config.py`
   - `test_183_log_analysis_improvements.py`
   に対して shared helper を適用
4. **xdist 準備**
   - `serial` marker を pytest 設定へ追加
   - 現環境では `pytest-xdist` 未導入なので、まずは marker と shared-state ラベル付けを先行

## 変更点

### 1. coverage の dev デフォルト解除

- `pytest.ini`
  - `--cov=ztb`
  - `--cov-report=term-missing`
  - `--cov-fail-under=20`
  を削除
- `pyproject.toml`
  - `[tool.pytest.ini_options].addopts` を `-ra -q` に整理

### 2. shared YAML / config helper

- `tests/unit/v460/_yaml_test_helpers.py`
  - `load_fill_test_config_from_text(...)`
  - `load_fill_test_config_from_path(...)`
  - `clone_fill_test_config(...)`
  を追加
- `tests/unit/v460/conftest.py`
  - `v460_fill_test_config_base` (`session`)
  - `v460_fill_test_config_yaml`
  を追加

### 3. 横展開したテスト

- `tests/unit/v460/test_fill_test_config.py`
  - production YAML roundtrip 系を session cache fixture に寄せた
- `tests/unit/v460/test_183_log_analysis_improvements.py`
  - inline YAML を `load_fill_test_config_from_text(...)` ベースの cached config fixture に寄せた
- `tests/unit/v460/test_yaml_test_helpers.py`
  - helper の cache / clone 契約を追加
- `tests/unit/v460/test_micro_timeout.py`
  - inline mapping を `load_fill_test_config_from_mapping(...)` ベースへ寄せた
  - production YAML 読み込みも `v460_fill_test_config_base` を再利用
- `tests/unit/v460/test_151_confidence_lot.py`
  - fixed mapping の `from_yaml(...)` を cached helper に寄せた
- `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - preflight / score calibration の固定 mapping を cached helper に寄せた
- `tests/unit/v460/test_157_regime_features.py`
  - regime feature の固定 mapping を cached helper に寄せた
- `tests/unit/v460/test_166_hotfixes.py`
  - production YAML 由来 config を `v460_fill_test_config_base` から clone する形へ寄せた
- `tests/unit/v460/test_202_log_improvements.py`
  - inline YAML text を `load_fill_test_config_from_text(...)` ベースへ寄せた

### 4. xdist 準備

- `pytest.ini`
- `pyproject.toml`
  - `serial` marker を追加
- `tests/unit/v460/test_retrain_hot_reload.py`
  - module-level で `pytest.mark.serial` を付与

### 5. mapping ベース cache helper の追加

- `tests/unit/v460/_yaml_test_helpers.py`
  - `load_fill_test_config_from_mapping(...)`
  - JSON canonicalization を使った cache path
  を追加
- dict literal / fixed nested mapping を直接 `FillTestConfig.from_yaml(...)` していたテストを、
  検出力を変えずに cached helper ベースへ移行した

### 6. fixed YAML wiring sweep

- 追加で以下を cached helper に横展開:
  - `tests/unit/v460/test_093_side_params.py`
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_243_yaml_wiring.py`
  - `tests/unit/v460/test_154_deadlock_prevention.py`
  - `tests/unit/v460/test_137_p1_features.py`
- いずれも fixed dict / fixed YAML text を対象にしており、
  assert の緩和や production 挙動変更は行わず、初期化重複だけを削減した

### 7. 本体コード側の確認

- `scripts/v460/ml/retrain_scheduler.py::load_retrain_config(...)` を確認
- YAML 読み込み自体は
  [config_loader.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/config_loader.py)
  の `_read_config_section()` で
  - file signature (`mtime_ns`, `size`)
  - `@lru_cache`
  を使ってすでにキャッシュされていた
- そのため現時点では、本体コードにさらに low-risk な大きい短縮余地は小さく、
  先に test 側の repeated `FillTestConfig.from_yaml(...)` を潰す方が効率的と判断
- `tests/unit/v460/test_retrain_hot_reload.py` 単体も確認:
  - `86 passed in 4.47s`
  - slowest でも `0.05s` 台
  - ここは「見た目ほど本丸ではない」

## 重量テストの初期観測

- baseline 実行:
  - `.venv/Scripts/python.exe -m pytest tests/ --durations=30 --tb=short`
- coverage 付き baseline は 31% 到達時点でも継続中で、ローカル開発の初期応答性がかなり悪かった。
- まずは coverage デフォルト解除を優先するのが妥当と判断。

## 部分計測 (before / after)

比較対象:

```bash
.venv/Scripts/python.exe -m pytest \
  tests/unit/v460/test_fill_test_config.py \
  tests/unit/v460/test_183_log_analysis_improvements.py \
  tests/unit/v460/test_yaml_test_helpers.py
```

- with coverage:
  - `101 passed in 15.53s`
  - `--cov-fail-under=20` により exit code 1
  - coverage report / warning 出力も大きい
- without coverage:
  - `10 passed, 177 deselected in 3.56s`

少なくとも focused subset では、coverage デフォルト解除だけで **4x 以上** の差が出ている。

## xdist 観点

- `pytest-xdist` は現環境に未導入
  - `.venv/Scripts/python.exe -m pip show pytest-xdist`
  - `Package(s) not found`
- そのため今回は
  - marker 定義
  - shared-state が疑われる hot-reload test の `serial` 化
 までを先行

## 次の一手

1. `--durations=30` を `--no-cov` 相当の新設定で再測定し、top 30 を分類する
2. `test_retrain_hot_reload.py` など autouse fixture の重いファイルを個別に軽量化する
3. `FillTestConfig.from_yaml()` の高頻度呼び出しファイルをさらに横展開する
4. before / after 比較をこの文書に追記していく
5. `tests/ --durations=30 --no-cov` を改めて取り切り、top 30 を分類する

## 検証

- `python3 -m py_compile`
  - `tests/unit/v460/_yaml_test_helpers.py`
  - `tests/unit/v460/conftest.py`
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`
- focused pytest:
  - `tests/unit/v460/test_fill_test_config.py`
  - `tests/unit/v460/test_183_log_analysis_improvements.py`
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_retrain_hot_reload.py`
  - 結果: `187 passed in 4.97s`
- focused pytest:
  - `tests/unit/v460/test_yaml_test_helpers.py`
  - `tests/unit/v460/test_micro_timeout.py`
  - `tests/unit/v460/test_151_confidence_lot.py`
  - `tests/unit/v460/test_138_p1_preflight_calibration.py`
  - `tests/unit/v460/test_157_regime_features.py`
  - `tests/unit/v460/test_166_hotfixes.py`
  - `tests/unit/v460/test_202_log_improvements.py`
  - 結果: `134 passed in 5.25s`
- focused pytest:
  - `tests/unit/v460/test_093_side_params.py`
  - `tests/unit/v460/test_094_stale_order.py`
  - `tests/unit/v460/test_243_yaml_wiring.py`
  - `tests/unit/v460/test_154_deadlock_prevention.py`
  - `tests/unit/v460/test_137_p1_features.py`
  - 結果: `120 passed in 4.36s`
- targeted hotspot check:
  - `tests/unit/v460/test_retrain_hot_reload.py --durations=20 --no-cov`
  - 結果: `86 passed in 4.47s`
  - 最遅ケースも `0.05s` 台で、現時点では top offender ではなかった
