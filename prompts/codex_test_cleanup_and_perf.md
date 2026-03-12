# Codex タスク: テスト整理 + 計算量削減 + 実行時間短縮

## 概要

zaif-trade-bot プロジェクトのテストスイート整理、不要資産の計算量削減、テスト実行時間の短縮を行う。
コードの変更は `--no-verify` でコミットすること。

---

## 1. テスト構造の現状

### ディレクトリ構成
```
tests/
├── unit/           → 533 test files (4571+ test cases in v460/ alone)
│   ├── v460/       → 162 files, 4538 passed, 33 skipped, 0 failed (69s)
│   ├── v459/       → 7 files
│   ├── trading/    → 80 files
│   ├── training/   → 36 files
│   ├── utils/      → 34 files
│   ├── reward/     → 26 files
│   ├── features/   → 26 files
│   ├── environment/→ 22 files
│   ├── analysis/   → 19 files
│   ├── core/       → 17 files
│   ├── algorithms/ → 15 files
│   ├── action_validation/ → 12 files
│   ├── experiments/→ 10 files
│   ├── tools/      → 10 files
│   ├── evaluation/ → 9 files
│   ├── risk/       → 9 files
│   └── (他多数)
├── integration/    → 25 files (5 FAILED — PPO integration tests)
├── training/       → 29 files (1 ERROR — test_v430_1000_steps.py import failure)
├── legacy_tests/   → 9 files
├── multimodal/     → 3 files
├── performance/    → 2 files
├── verification/   → 2 files
├── scripts/        → 1 file
├── tools/          → 2 files
├── trading/        → 4 files
├── utils/          → 1 file
├── v456/           → 1 file
├── v460/           → 1 file
└── benchmark/      → 0 files (empty)
```

### 現在の失敗テスト一覧

#### unit/ (5 failures — 全て既存のまま放置されている)
```
FAILED tests/unit/action_validation/test_signal_guidance_system.py::TestSignalGuidanceSystem::test_full_guidance_pipeline
FAILED tests/unit/action_validation/test_signal_performance_analyzer.py::TestSignalPerformanceAnalyzer::test_history_size_management
FAILED tests/unit/algorithms/test_ab_test_framework.py::TestABTestTypes::test_ab_test_configuration_creation
FAILED tests/unit/algorithms/test_ab_test_framework.py::TestABTestTypes::test_statistical_test_enum
FAILED tests/unit/algorithms/test_ab_test_framework.py::TestABTestConfig::test_performance_config
```

#### integration/ (5 failures)
```
FAILED tests/integration/test_custom_ppo_integration.py::TestCustomPPOInstantiation::test_create_with_all_mitigations
FAILED tests/integration/test_custom_ppo_integration.py::TestCustomPPOInstantiation::test_create_with_pan_only
FAILED tests/integration/test_custom_ppo_integration.py::TestCustomPPOInstantiation::test_create_with_target_entropy_only
FAILED tests/integration/test_custom_ppo_integration.py::TestCustomPPOTraining::test_short_training_run
FAILED tests/integration/test_custom_ppo_integration.py::TestCustomPPOTraining::test_pan_statistics_logging
```

#### training/ (1 collection error)
```
ERROR tests/training/test_v430_1000_steps.py — ModuleNotFoundError: No module named 'sac'
```

### pytest.ini 設定
```ini
[pytest]
testpaths = tests
addopts =
    --tb=short
    --strict-markers
    --disable-warnings
    --ignore=archived
    --ignore=scripts
    --ignore-glob=**/archived/**
    --ignore-glob=**/scripts/**
    --cov=ztb
    --cov-report=term-missing
    --cov-fail-under=80     ← これが毎回 FAIL を出す (実際のカバレッジ ~25%)
    --maxfail=5
```

---

## 2. 実施タスク

### タスク A: 壊れたテストの修正 or 除去

1. **unit/ の 5 failures を調査し修正**:
   - テストコード自体がプロダクションコード変更に追随していないなら、テストを修正
   - テスト対象のモジュールが deprecated (archived/ に相当) なら、テストを `tests/legacy_tests/` に移動
   - 修正不可能なら `@pytest.mark.skip(reason="...")` で明示的にスキップし、理由をコメント

2. **integration/ の 5 PPO failures**:
   - `test_custom_ppo_integration.py` の失敗原因を調査
   - 修正可能なら修正。不可能なら `@pytest.mark.skip` + 理由

3. **training/test_v430_1000_steps.py**:
   - `from sac import SACSuite` で `ModuleNotFoundError`
   - このモジュールが現在のコードベースに存在しないなら `tests/legacy_tests/` に移動

### タスク B: pytest.ini の修正

1. **`--cov-fail-under=80`** を削除するか、現実的な値 (例: 20) に変更
   - 現在のカバレッジは ~25%。80% 閾値は全テスト実行を FAIL にしており、CI 上も意味をなしていない
   - **推奨**: `--cov-fail-under=20` に変更し、将来的にカバレッジ向上を目指す
   - または `--no-cov-on-fail` を追加して、カバレッジ不足を warning に留める

2. **`--maxfail=5`** を確認:
   - 現在 5 件で停止する設定。unit/ と integration/ で異なる失敗があるため、非 v460 テスト実行時にv460テストが実行されない
   - v460 テストは独立して `python -m pytest tests/unit/v460/ -q` で 4538 passed, 0 failed なので、maxfail がマスクしている

### タスク C: テスト実行時間の短縮

1. **v460 テスト (4538 cases, 69s)** の中で遅いテストを特定:
   ```bash
   python -m pytest tests/unit/v460/ --durations=20 -q
   ```
   - 1 秒以上かかるテストを `@pytest.mark.slow` でマーク
   - テスト内で不要な sleep() や大量の fixture 生成があれば最適化

2. **unit/ 全体 (68+ cases non-v460, ~68s)** の中で遅いテストを特定:
   ```bash
   python -m pytest tests/unit/ --ignore=tests/unit/v460/ --durations=20 -q
   ```

3. **integration/ テスト (31 passed, 307s!)** — 非常に遅い:
   - PPO training テストが大半の時間を占めていないか確認
   - 不要な integration テストを `@pytest.mark.slow` でマーク

### タスク D: 空ディレクトリ・dead test ファイルの整理

1. `tests/benchmark/` — 0 files → 空ディレクトリなら削除
2. `tests/unit/data_processing/` — 0 files → 空ディレクトリなら削除
3. `tests/unit/feature_engineering/` — 0 files → 空ディレクトリなら削除
4. `tests/unit/model_evaluation/` — 0 files → 空ディレクトリなら削除
5. 各テストディレクトリに `__init__.py` が存在するか確認し、不足分を追加

### タスク E: conftest.py の強化

1. 現在の `conftest.py` (root) は archived/scripts の ignore のみ
2. 共通 fixture があれば `tests/conftest.py` にまとめる (重複 fixture の解消)
3. `tests/unit/v460/conftest.py` が存在するか確認。なければ作成不要 (テストが動いているため)

---

## 3. 制約・注意事項

### 絶対に守ること
- **`tests/unit/v460/` の 4538 テストは全て pass を維持すること** (これが現在のベースライン)
- テスト修正後、`python -m pytest tests/unit/v460/ -q --no-header --tb=no` で 4538 passed, 0 failed を確認
- プロダクションコード (`ztb/`, `scripts/`) は変更しないこと — テストコードとpytest設定のみ変更
- `--no-verify` オプションを付けてコミット

### コミットメッセージ
```
chore(tests): テスト整理・壊れたテスト修正・pytest.ini最適化

- 壊れた unit/integration テスト N 件を修正/skip/移動
- pytest.ini: --cov-fail-under を現実的な値に修正
- 空テストディレクトリの整理
- テスト実行時間の短縮 (具体的な改善を記載)

ベースライン: tests/unit/v460/ 4538 passed, 0 failed 維持
```

### 確認コマンド
```bash
# ベースライン確認 (必須)
python -m pytest tests/unit/v460/ -q --no-header --tb=no

# 全テスト確認
python -m pytest tests/ -q --no-header --tb=no --ignore=tests/training/test_v430_1000_steps.py

# 遅いテスト特定
python -m pytest tests/unit/ --durations=30 -q --no-header --tb=no
```

---

## 4. 成果基準

| 指標 | Before | After (目標) |
|---|---|---|
| unit/ failures | 5 | 0 |
| integration/ failures | 5 | 0 (修正 or marked skip) |
| training/ errors | 1 | 0 (移動 or 修正) |
| pytest.ini cov-fail-under | 80% (常に FAIL) | 20% (PASS) or 削除 |
| v460 test baseline | 4538 passed | 4538 passed (変更なし) |
| 全テスト実行時の exit code | 1 (常に FAIL) | 0 |
