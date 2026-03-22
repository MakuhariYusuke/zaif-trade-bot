# 558# [phg] [impl] 報酬ロジックの共通化と RewardKernel の実装報告

> **ステータス**: 実装完了 (Gemini 担当)  
> **作成日**: 2026-03-23  
> **作成者**: Gemini CLI Agent

---

## 1. 実施内容：報酬計算ロジックの SSOT (Single Source of Truth) 化

### 1.1 RewardKernel の新設
`ztb/trading/environment/components/calculators/reward_kernel.py` を作成し、報酬計算の「ステートレスな核」を抽出した。
- **機能**: PnL スケーリング、基本ペナルティ（HOLD/Position Change）、ボーナス（Trade Frequency）、破産ペナルティ。
- **利点**: 環境の重さ（Heavy/Lite）に依存せず、常に同一の計算式を保証。

### 1.2 LiteTradingEnv への適用
`scripts/v460/lib/lite_trading_env.py` のハードコードを `RewardKernel` 呼び出しに置換。
- **パラメータ同期**: `hold_penalty_multiplier` などのパラメータを `LiteEnvConfig` に追加し、正規の報酬体系を Lite 版でも利用可能にした。
- **メリット**: 実験環境（Lite）での評価が、より正確に本番（Heavy）に近づく。

### 1.3 RewardCalculator (ztb) のリファクタリング
2,200行ある本体の `calculate_reward_simple` メソッドを `RewardKernel` 呼び出しに集約。
- **成果**: 複雑なバリデーションと条件分岐を整理し、コードの可読性を向上。
- **互換性**: 動的シェーピング、シグナル統合、非対称スケーリングなどの「高度な機能」は `RewardCalculator` 側のオーケストレーションとして維持。

---

## 2. 残された課題と次フェーズの提案

### 2.1 引数爆発の解消 (RewardContext 化)
`calculate_reward` の引数が 15 個以上あり、結合度が非常に高い。
- **提案**: `RewardContext` オブジェクトを全面的に採用し、インターフェースをシンプルにする。

### 2.2 巨大クラスの物理分割
`RewardCalculator` 内の `_calculate_xxx_reward` 群を、それぞれ独立した `RewardStrategy` クラスへ分離し、ファイル分割を行う。

### 2.3 設定取得の高速化
ステップ毎の `get_setting_xxx` 呼び出しを削減し、`RewardParams` (Dataclass) による一括パラメータ管理を導入する。

---

## 3. AI エージェントへの共有事項

- **Codex / Copilot**: `RewardKernel` の導入により、報酬の「基本」は一元化された。今後の報酬アルゴリズムの変更は、可能な限り `RewardKernel` または `RewardStrategy` コンポーネントに限定されたし。
- **Git**: 今回のコミットにより、リポジトリの構成とソースコードの整合性が大幅に改善された。

---
