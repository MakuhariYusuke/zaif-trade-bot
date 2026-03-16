# 外部AIコーディングエージェント向けプロンプト: Gate 0 + Phase B 実装依頼

## プロジェクト概要

BTC/JPY 1分足取引の強化学習ボット（SAC）を開発中です。現在、ROI約-5%の問題を調査しており、以下の2つの機能追加が必要です。

**大義**: 短期間での高収益性システムの実現

---

## 依頼事項

### タスク1: Gate 0 - 設定反映検証ログの追加

**背景**: 
過去のv444で「設定伝播バグ」（設定値がデフォルト値で上書きされる）が発生しました。P1実験でreward_paramsを変更しましたが、実際に環境に適用されたか確認できていません。

**要件**:
1. SACトレーニング開始時に、実際に適用されているreward_paramsをWARNINGレベルでログ出力
2. 環境初期化時に、報酬計算に使用されるパラメータをログ出力
3. 期待値と実際値の比較ができるフォーマット

**既存実装の参照先**:
- `ztb/training/unified_trainer/algorithms/sac_trainer.py` - SACTrainerクラス
- `ztb/trading/environment/heavy_env/core.py` - HeavyTradingEnv（メイン環境）
- `ztb/trading/environment/components/reward_calculator.py` - RewardCalculator
- `docs/BALANCE_PENALTY_ROOT_CAUSE_FIX_FINAL.md` - v444バグ修正の参考

**出力例**:
```
WARNING: ========== REWARD PARAMS VERIFICATION ==========
WARNING: EXPECTED: alpha=0.0, beta=0.0, gamma=0.0, fee_penalty_weight=0.0
WARNING: ACTUAL:   alpha=0.0, beta=0.0, gamma=0.0, fee_penalty_weight=0.0
WARNING: STATUS: ✅ MATCH - Settings correctly applied
WARNING: ================================================
```

または不一致時:
```
WARNING: STATUS: ❌ MISMATCH - Settings may not be applied correctly!
WARNING: Check config propagation path.
```

---

### タスク2: Phase B - コスト分解ログの追加

**背景**:
v457.2の分析で「Gross PnL（手数料前）はプラスだが、Net PnL（手数料後）はマイナス」という発見がありました。現在の環境ではこの分解ができていません。

**要件**:
1. 以下のメトリクスを環境から取得可能にする:
   - `gross_pnl`: 手数料・スリッページ前の純利益
   - `total_fees`: 全取引の手数料合計
   - `total_slippage`: 全取引のスリッページ合計
   - `net_pnl`: gross_pnl - total_fees - total_slippage

2. トレーニング終了時にコスト分解サマリーをWARNINGレベルで出力

3. `ztb/utils/env_metrics.py` の `extract_trainer_env_metrics()` で取得できるようにする

**既存実装の参照先**:
- `ztb/trading/environment/heavy_env/core.py` - 環境の取引実行ロジック
- `ztb/trading/environment/heavy_env/mixins/trading.py` - 取引処理ミックスイン
- `ztb/trading/environment/components/position_manager.py` - ポジション管理
- `ztb/utils/env_metrics.py` - メトリクス抽出ユーティリティ（既存）
- `docs/BALANCE_EXPLORATION_AND_MEMORY_OPTIMIZATION.md` - v447のreward_components保存バグ修正の参考

**出力例**:
```
WARNING: ========== COST BREAKDOWN ANALYSIS ==========
WARNING: Gross PnL:     +2,500 JPY (+2.50%)
WARNING: Total Fees:    -4,000 JPY (-4.00%)
WARNING: Total Slippage: -500 JPY (-0.50%)
WARNING: Net PnL:       -2,000 JPY (-2.00%)
WARNING: Cost Ratio:    180.0% (costs / |gross_pnl|)
WARNING: Interpretation: 取引自体は利益だがコストに負けている
WARNING: ===============================================
```

---

## 実装方針

### 必須
- **既存コードの活用を最優先**: 新規ファイル作成より既存ファイルへの追加を優先
- **ログレベルはWARNING**: 通常のINFOログに埋もれないよう
- **テスト追加**: 新機能には対応するユニットテストを追加

### 禁止
- 大規模なリファクタリング
- 報酬計算ロジックの変更
- パフォーマンスに影響する重い処理の追加

### 参考パターン

**v447でのreward_components保存パターン**:
```python
# ztb/trading/environment/components/reward_calculator.py
def calculate_reward_simple(self, action: int, info: Dict[str, Any]) -> float:
    # ... 報酬計算 ...
    
    # Store reward components for analysis
    self._last_reward_components = {
        "stage": stage,
        "pnl": pnl,
        "adjusted_pnl": adjusted_pnl,
        "base_reward": base_reward,
        # ...
    }
    
    return final_reward
```

同様のパターンでコスト情報を保存してください。

---

## ファイル構成

主要なファイルパス:
```
ztb/
├── training/
│   └── unified_trainer/
│       └── algorithms/
│           └── sac_trainer.py      # タスク1: ここでログ出力
├── trading/
│   └── environment/
│       ├── heavy_env/
│       │   ├── core.py             # タスク2: コスト累積変数追加
│       │   └── mixins/
│       │       └── trading.py      # タスク2: 取引時にコスト記録
│       └── components/
│           ├── reward_calculator.py
│           └── position_manager.py
└── utils/
    └── env_metrics.py              # タスク2: コスト取得関数追加
```

---

## 検証方法

### タスク1の検証
```python
# P1実験と同じ設定で実行し、ログを確認
# reward_params = {"alpha": 0.0, "beta": 0.0, ...}
# ログに「ACTUAL: alpha=0.0」と出力されればOK
```

### タスク2の検証
```python
# 10,000ステップのトレーニング後、以下が取得できること
metrics = extract_trainer_env_metrics(trainer)
assert 'gross_pnl' in metrics
assert 'total_fees' in metrics
assert 'net_pnl' in metrics
```

---

## 期待する成果物

1. 修正されたファイル一覧とdiff
2. 追加されたユニットテスト
3. 動作確認結果（ログ出力のスクリーンショットまたはテキスト）

---

## 補足情報

- Python 3.11
- stable-baselines3 使用
- ログはPython標準logging（loggerはモジュールレベルで定義済み）
- 型ヒントを使用（mypy対応）

質問があれば先に確認してください。
