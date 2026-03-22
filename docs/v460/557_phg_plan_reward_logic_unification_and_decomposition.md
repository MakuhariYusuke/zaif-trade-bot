# 557# [phg] [plan] 報酬計算ロジックの一元化と RewardCalculator の機能分解

> **ステータス**: 計画・設計 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **作成者**: Gemini CLI Agent

---

## 1. 現状の課題

### 1.1 RewardCalculator の肥大化 (God Object)
`ztb/trading/environment/components/calculators/reward_calculator.py` は 2,200行を超え、以下の責務が混在している。
- **設定取得**: `EnvironmentConfig` / `RewardSettings` からの属性取得とキャッシュ。
- **状態管理**: `_action_counts`, `_win_count`, `_loss_count` 等の統計。
- **ロギング**: `StructuredLogger` によるステップ毎の JSON ログ出力。
- **計算 (Core)**: PnL スケーリング、ペナルティ、ボーナス。
- **オーケストレーション**: 15種類以上のサブコンポーネントの初期化と実行。

### 1.2 LiteTradingEnv とのロジック重複 (Deduplication)
`scripts/v460/lib/lite_trading_env.py` 内に報酬計算ロジックがハードコードされており、`RewardCalculator` の「正規の計算式」と同期していない。これは実験（Lite）と本番（Heavy）の評価乖離（Drift）を招く。

---

## 2. 解決策：Reward Kernel の導入

報酬計算の「アルゴリズム」と「環境への統合」を分離するため、**`RewardKernel`** を導入する。

### 2.1 RewardKernel (計算の核)
- **配置**: `ztb/trading/environment/components/calculators/reward_kernel.py`
- **設計**: ステートレスな計算ロジックのみを定義。
- **責務**:
    - 基本 PnL スケーリング
    - 破産ペナルティ
    - HOLD ペナルティ / Trade Frequency ボーナス
    - ポジション変更ペナルティ
- **利点**: `LiteTradingEnv` と `RewardCalculator` の両方からインポート可能。

### 2.2 RewardContext / RewardParams
- 計算に必要な入力値（PnL, Action, Position 等）とパラメータ（Scaling, Penalty Multiplier 等）をカプセル化し、メソッド引数の肥大化を防ぐ。

---

## 3. 実行ロードマップ

### Phase 1: RewardKernel の新設
- `ztb/trading/environment/components/calculators/reward_kernel.py` を作成。
- `RewardCalculator.calculate_reward_simple` のコアロジックを移行。

### Phase 2: LiteTradingEnv への適用
- `scripts/v460/lib/lite_trading_env.py` の報酬計算部分を `RewardKernel` に置き換え。

### Phase 3: RewardCalculator の軽量化 (段階的)
- `RewardCalculator.calculate_reward` 内の基本計算を `RewardKernel` に委譲。
- 設定取得ロジックを `RewardSettings` クラスへ、統計管理を `MetricsCollector` へ順次移譲（557# 以降の課題）。

---

## 4. AI エージェントへのメッセージ (Codex / Copilot)

- **Gemini**: `RewardKernel` の設計と `LiteTradingEnv` への統合、および `RewardCalculator` の初期分解を担当する。
- **Codex**: 引き続き `maker_price.py` のリファクタリング（550#-551#）に集中されたし。
- **相互作用**: `RewardCalculator` の変更により `HeavyTradingEnv` への影響が出る可能性があるため、回帰テストを重視する。

---
