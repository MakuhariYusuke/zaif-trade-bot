# 557# [phg] [plan] 報酬計算ロジックの一元化と RewardCalculator の機能分解 (深掘り版)

> **ステータス**: 計画・設計 (Gemini 担当)  
> **更新日**: 2026-03-23  
> **作成者**: Gemini CLI Agent

---

## 1. 核心的な目的：ロジックの「ztb 委譲」

`scripts/v460` 以下の実験用コード（特に `LiteTradingEnv`）における独自実装を廃止し、すべての報酬計算ロジックを `ztb` パッケージ側に委譲する。これにより、実験環境と本番環境の報酬定義が「100% 同一」であることを保証する。

### 1.1 独自実装の排除
- `LiteTradingEnv.step()` 内での PnL 計算やペナルティ計算を、すべて `RewardKernel` への呼び出しに置き換える（Phase 2 で実施済み、さらに深掘りする）。
- `RewardKernel` は `ztb/trading/environment/components/rewards/utils.py` (`RewardUtils`) などの軽量ヘルパーを積極的に活用する。

### 1.2 ヘルパー関数の再利用
`RewardUtils` に含まれる以下のロジックを `RewardKernel` および `scripts/v460` で共通利用する。
- `calculate_balance_penalty`: アクション比率の偏りに対するペナルティ。
- `calculate_activity_bonus`: 直近の取引頻度に対するボーナス。
- `calculate_position_size_bonus`: 適切なポジション量維持へのインセンティブ。

---

## 2. アーキテクチャ設計

### 2.1 RewardKernel の拡張 (Phase 1 強化)
`RewardKernel` を単なる PnL 計算機から、**「環境に依存しない報酬エンジンの核」** に昇格させる。

- **Input**: `RewardContext` (PnL, Actions, Positions, etc.)
- **Params**: `RewardParams` (Scaling, Multipliers, etc.)
- **Logic**: `RewardUtils` を内部で呼び出し、ステートレスに報酬を算出。

### 2.2 設定同期の自動化
`LiteEnvConfig` が `ztb` 側の `RewardSettings` の一部を透過的に扱えるようにし、YAML 設定値がそのまま `RewardKernel` に流れる構造を構築する。

---

## 3. 実行ロードマップ (更新)

### Phase 1: RewardKernel の強化 (実施中)
- `RewardUtils` のロジックを `RewardKernel` に統合。
- `pytest` による `RewardCalculator.calculate_reward_simple` との完全な整合性検証。

### Phase 2: scripts/v460 への完全委譲
- `LiteTradingEnv` だけでなく、分析スクリプト等で報酬を再計算している箇所があれば、すべて `RewardKernel` に移行。

### Phase 3: テスト駆動の品質保証
- `tests/unit/v460/test_558_reward_unification.py` を新設。
- 数値的な一致（浮動小数点誤差の許容範囲内）を自動テストで監視。

---

## 4. 期待される効果

- **評価の信頼性向上**: `LiteTradingEnv` で学習したモデルを `HeavyTradingEnv` に持っていった際の「報酬の不連続性」を排除。
- **コード削減**: `RewardCalculator.py` の 2,200行という巨大さを、コンポーネントの切り出しによって劇的に削減する道筋を作る。
- **保守性**: 報酬ロジックの修正が必要な際、`RewardKernel` または `RewardUtils` を直すだけで全環境に反映される。

---
