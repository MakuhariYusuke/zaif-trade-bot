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

### 4. xdist 準備

- `pytest.ini`
- `pyproject.toml`
  - `serial` marker を追加
- `tests/unit/v460/test_retrain_hot_reload.py`
  - module-level で `pytest.mark.serial` を付与

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
