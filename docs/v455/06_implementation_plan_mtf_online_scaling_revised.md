# Implementation Plan: Online Scaling & Multi-Timeframe System (Revised)

## 1. 概要
本ドキュメントは、`docs/v455/05_backtest_analysis_and_creative_solutions.md` で提起された課題（データリーク、Alpha不足）に対する具体的解決策の実装計画である。
**「全期間統計によるスケーリング（リーク）」を廃止し、「Online Scaler」へ移行する**とともに、**既存の「Multi-Timeframe (MTF) System」**を活用してモデルの予測精度（Alpha）を強化する。

**変更点 (2025-12-26):**
ユーザーの要望により、新規の `MultiTimeframeManager` 実装ではなく、既存の `ztb.features.generators.multi_timeframe` モジュールを使用する方針に変更した。

---

## 2. Online Scaler Design (データリーク防止)

従来の「全期間の統計量を使ったスケーリング」は、未来の情報を現在に持ち込むデータリークの原因となる。これを防ぐため、**Welfordのアルゴリズム**を用いたオンラインスケーラーを導入する。

### 2.1 Python Class Structure: `OnlineScaler`

(変更なし: `ztb/processing/online_scaler.py` に実装済み)

### 2.2 Integration Strategy
*   `HeavyTradingEnv` の初期化時に `OnlineScaler` をインスタンス化。
*   `ObservationBuilder` でのグローバルスケーリングを無効化。
*   `_get_observation` メソッドで、観測ベクトル取得後に `OnlineScaler.update` および `transform` を適用。

---

## 3. Multi-Timeframe Architecture (MTF) - Revised

既存の `ztb.features.generators.multi_timeframe` システムを活用し、環境初期化時にMTF特徴量を生成・結合する。

### 3.1 Integration Logic
*   `HeavyTradingEnv._initialize_features_and_spaces` にて、`MultiTimeframeFeatureSystem` を呼び出す。
*   `include_multi_timeframe_features` フラグを強制的に有効化する。
*   生成されたMTF特徴量（5m, 15m, 1h等のテクニカル指標）を `self.df` に結合する。
*   これにより、`ObservationBuilder` は通常のプロセスとしてMTF特徴量を観測ベクトルに含めることができる。

### 3.2 Advantages of Using Existing System
*   **信頼性**: 既存のテスト済みコードベースを使用するため、バグのリスクが低い。
*   **一貫性**: 他のコンポーネント（バックテスト、学習パイプライン）との整合性が保たれる。
*   **機能**: 既に実装されている高度な機能（欠損値補完、同期など）を利用できる。

---

## 4. Implementation Steps

1.  **Revert Custom MTF**: 新規作成した `ztb/features/multi_timeframe/manager.py` を削除し、関連する変更を元に戻す。
2.  **Enable Existing MTF**: `ztb/trading/environment/heavy_env/mixins/initialization.py` を修正し、既存のMTFシステムを有効化するロジックを強化する。
3.  **Integrate Online Scaler**: `OnlineScaler` を `HeavyTradingEnv` に組み込み、データリークのないスケーリングを実現する。
4.  **Verify**: 統合テストを実行し、MTF特徴量が正しく生成され、スケーリングが機能していることを確認する。
