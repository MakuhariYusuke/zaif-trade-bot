# 558# [phg] [impl] 報酬ロジックの共通化と RewardKernel の実装報告 (完結版)

> **ステータス**: 実装完了・テスト通過 (Gemini 担当)  
> **更新日**: 2026-03-23  
> **作成者**: Gemini CLI Agent

---

## 1. 実施内容：報酬ロジックの「ztb 委譲」と重複排除

### 1.1 RewardKernel の強化と SSOT 化
`ztb/trading/environment/components/calculators/reward_kernel.py` を強化し、単なる PnL 計算機から、**環境非依存の報酬エンジン**へと昇格させた。
- **RewardUtils との統合**: `ztb` 側の既存ヘルパー (`RewardUtils`) を内部で呼び出すように設計。
- **サポート機能**: 
    - 基本 PnL スケーリング
    - 取引頻度ボーナス / HOLD ペナルティ
    - ポジション変更ペナルティ (v431/v440 準拠)
    - アクティビティボーナス (直近取引頻度)
    - バランスペナルティ (アクション比率の偏り)

### 1.2 scripts/v460 (LiteTradingEnv) への完全委譲
`scripts/v460/lib/lite_trading_env.py` 内の独自報酬計算を廃止。
- **修正内容**: `RewardKernel.calculate_basic_reward` への呼び出しに完全置換。
- **効果**: Lite 環境での学習・評価が、Heavy 環境と同一の報酬スケール・定義で行われることを保証。

### 1.3 RewardCalculator (ztb) のリファクタリング
巨大クラス `RewardCalculator` の `calculate_reward_simple` メソッドを `RewardKernel` 委譲型に書き換え。
- **成果**: 複雑なバリデーションと条件分岐を削除し、ロジックを Kernel に集約。
- **保守性**: 報酬定義の基本部分を修正する場合、`RewardKernel` 1 箇所を直せば全環境に反映される。

---

## 2. 品質保証 (Testing)

### 2.1 整合性テストの実施
`tests/unit/v460/test_558_reward_unification.py` を作成し、`.venv/pytest` で検証。
- **テスト 1 (基本整合)**: `RewardCalculator` と `RewardKernel` の PnL スケーリング一致を確認。
- **テスト 2 (破産)**: 資産ゼロ時のペナルティ適用の一致を確認。
- **テスト 3 (高度ヘルパー)**: `RewardUtils` 経由のボーナス・ペナルティ計算の正確性を確認。
- **結果**: **All 3 Tests PASSED.**

---

## 3. 今後の展望

- **God Object 分割の加速**: 今回の Kernel 抽出により、`RewardCalculator` の物理分割（Strategy パターンへの完全移行）への道筋が明確になった。
- **設定ドリフトの完全防止**: `RewardParams` を SSOT とし、YAML 設定からの自動変換機能をさらに強化する。

---
