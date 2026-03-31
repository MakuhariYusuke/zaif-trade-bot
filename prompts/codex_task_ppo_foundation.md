# Codex Task: PPO Sidecar 基盤整備 (v461 準備)

## 目的
v461 で PPO sidecar を SAC と並行運用するための基盤整備。
既存 PPO コード（120+ ファイル）の棚卸し、テスト整備、不要コードの整理を行う。

## 背景
- 現行 SAC sidecar は `total_trades: 0` で実質機能停止 (675# 分析)
- PPO は離散行動 (BUY/SELL/SKIP) に適しており、side selection 改善が期待される
- 既存コードベースには PPO 関連コードが広範に存在するが、v460 では未使用
- テストの多くが壊れている可能性がある

## タスク

### Task 1: PPO コードの棚卸し・分類

以下のファイルを調査し、各ファイルの状態を分類:

**コア実装 (動作確認要):**
- `ztb/training/algorithms/ppo/ppo_algorithm.py`
- `ztb/training/unified_trainer/algorithms/ppo_trainer.py`
- `ztb/training/core/ppo_trainer.py`
- `ztb/training/custom_ppo.py`
- `ztb/training/ppo_trainer.py`
- `ztb/training/trainers/ppo_trainer.py`
- `sb3_contrib/__init__.py` (MaskablePPO)

**設定:**
- `ztb/training/config/ppo_config.py`
- `ztb/training/constants.py` (PPO 関連定数)

**テスト:**
- `tests/training/test_ppo_trainer.py`
- `tests/unit/algorithms/test_ppo_algorithm.py`
- `tests/unit/training/test_ppo_trainer.py`
- `tests/integration/test_custom_ppo_integration.py`

**archive 候補:**
- `ztb/training/archive/ppo_trainer_old.py`
- `ztb/training/experiments/sell_mitigation_ppo_trainer.py`
- `experiments/train_sac_v443_2_*.py`
- `scripts/training/train_v445_*.py`

分類基準:
- **ACTIVE**: v461 で使用予定。テスト整備対象
- **REFACTOR**: 構造は使えるが修正が必要
- **ARCHIVE**: 過去バージョン用。archived/ に移動候補
- **DELETE**: 完全に不要

### Task 2: PPO テストの整備

PPO 関連テストを実行し、壊れているテストを修正:

```bash
python -m pytest tests/training/test_ppo_trainer.py -x --tb=short
python -m pytest tests/unit/algorithms/test_ppo_algorithm.py -x --tb=short
python -m pytest tests/unit/training/test_ppo_trainer.py -x --tb=short
python -m pytest tests/integration/test_custom_ppo_integration.py -x --tb=short
```

修正方針:
- import エラー → パス/モジュール名の修正
- API 変更 → 現行 SB3 API に合わせる
- stub 依存 → `_sb3_test_stub/` の PPO stub が正しく動作するか確認
- 不要テスト → skip mark 付与

### Task 3: HeavyTradingEnv の離散行動モード確認

`ztb/trading/environment/heavy_env/core.py` で:
1. `use_continuous_actions: false` 設定時の action_space が正しく `Discrete(3)` になるか確認
2. 離散モードでの step() が BUY/SELL/HOLD を正しく処理するか確認
3. テストがあれば実行、なければ作成

### Task 4: PPO 関連の過去バージョンコード整理

`experiments/`, `scripts/training/`, `archived/` に散在する過去 PPO コードを整理:
- v443, v444, v445 の実験コードを `archived/` 配下に移動
- 移動時は `git mv` ではなく手動コピー + git add (安全のため)
- binary_search/ 配下の PPO optimizer ファイル群の状態確認

## 制約
- `git commit --no-verify -m "..."` でコミット
- `git add .` 禁止。対象ファイルを個別指定
- テスト実行: `python -m pytest tests/ -x --tb=short`
- 既存の SAC テスト (2248 passing) を壊さないこと
- 型安全: Any 型回避, mypy 準拠

## 成果物
1. PPO コード分類レポート (棚卸し結果)
2. PPO テスト修正 (passing 状態)
3. HeavyTradingEnv 離散モードの動作確認テスト
4. 過去コードの整理 (archived/ 移動)
