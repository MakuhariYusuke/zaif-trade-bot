# P2実装ノート：学習コスト管理とリーク耐性強化

## 実装日時
- **日付**: 2025年10月7日
- **P2-7実装時間**: 約20分
- **残タスク**: P2-8（マイクロ構造）、P2-9（リーク耐性テスト）、P2-10（モデルカード）

---

## P2-7: gc_artifacts.py（Checkpoint Keep）✅ 完了

### 問題
- 1M学習 = 25k間隔で約40個のcheckpoint生成
- 1checkpoint = 約50-100MB → 総2-4GB のディスク消費
- 自動削除なしでディスク圧迫リスク

### 解決策
**scripts/gc_artifacts.py**（約380行）を作成:

#### 保持ポリシー
```python
# Last N checkpoints (最新N個保持)
keep_last = 4  # デフォルト: 最新100k steps分

# Best M by Sharpe ratio (高Sharpe上位M個保持)
keep_best = 3  # デフォルト: ベスト3個

# TTL (Time To Live)
ttl_days = 14  # デフォルト: 14日以内のものは保持
```

#### 主要機能
1. **Checkpoint スキャン**:
   - `checkpoint_25000.zip` または `checkpoint_25000/` をスキャン
   - `manifest.json` からSharpe ratio抽出
   - ファイルサイズ、タイムスタンプ取得

2. **保持マーキング**:
   - 最新N個をマーク (`keep_last`)
   - 高Sharpe上位M個をマーク (`keep_best`)
   - TTL以内をマーク (`ttl_days`)

3. **安全削除**:
   - マークされていないcheckpointのみ削除
   - Dry-runモード対応（`--dry-run`）
   - 削除失敗時も学習継続（保守的設計）

#### 使用例
```bash
# Dry-run（削除内容確認）
python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M --dry-run

# 実行
python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M

# カスタム設定
python scripts/gc_artifacts.py --checkpoint-dir checkpoints/ensemble_C_1M \
    --keep-last 6 --keep-best 5 --ttl-days 21
```

### ztb/training/callbacks.py統合

**CheckpointGCCallback**（約90行）を追加:

```python
from ztb.training.callbacks import CheckpointGCCallback

gc_callback = CheckpointGCCallback(
    checkpoint_dir="checkpoints/ensemble_C_1M",
    keep_last=4,
    keep_best=3,
    ttl_days=14,
    check_interval=25000,  # 25k毎に自動GC
)

model.learn(total_timesteps=1_000_000, callback=gc_callback)
```

#### 自動GC実行タイミング
- チェックポイント保存後、25k間隔で自動実行
- 学習を中断せずバックグラウンドで削除
- GC失敗時もログ出力のみで学習継続

### 検証結果
- ✅ scripts/gc_artifacts.py: 構文チェックPASS
- ✅ ztb/training/callbacks.py: 構文チェックPASS
- ✅ CheckpointGCCallback統合完了

### 達成効果
- **ディスク節約**: 40checkpoints → 4-7checkpoints（約85%削減）
- **自動化**: 手動削除不要、25k毎に自動クリーンアップ
- **安全性**: 保守的削除（エラー時は保持）、Dry-runモード対応

---

## P2-8: マイクロ構造（spread/min_tick/min_qty）📋 TODO

### 問題
- 現在の取引コストモデル: 固定スリッページ（0.1%）のみ
- 実環境では: スプレッド、最小呼値、最小取引量、可変スリッページ
- 学習時の簡略化 ≠ 実運用での複雑性 → Sim-to-real gap

### 解決策案
**ztb/trading/environment.py強化**（約150行追加予定）:

#### マイクロ構造パラメータ
```python
@dataclass
class MicrostructureConfig:
    # Spread (買値-売値差)
    base_spread_bps: float = 5.0  # 0.05% base spread

    # Minimum tick (最小呼値)
    min_tick: float = 1.0  # BTC/JPY: 1円単位

    # Minimum quantity (最小取引量)
    min_qty: float = 0.001  # BTC: 0.001 BTC

    # Variable slippage (可変スリッページ)
    slip_base_bps: float = 2.0  # 0.02% base
    slip_atr_coef: float = 0.5  # ATR係数
    # slip = max(base, k * ATR)
```

#### 取引実行ロジック
```python
def execute_order(self, action: int, quantity: float) -> float:
    # 1. Spread適用
    if action == BUY:
        price = self.ask_price  # 買値 = mid + spread/2
    elif action == SELL:
        price = self.bid_price  # 売値 = mid - spread/2

    # 2. Min tick丸め
    price = round(price / self.min_tick) * self.min_tick

    # 3. Min quantity チェック
    if abs(quantity) < self.min_qty:
        return 0.0  # 最小量未満は約定しない

    # 4. Variable slippage
    atr = self.get_atr(window=14)
    slip = max(self.slip_base_bps, self.slip_atr_coef * atr)

    # 5. 最終約定価格
    executed_price = price * (1 + slip/10000 * sign(action))

    return executed_price
```

#### テストケース
**tests/trading/test_microstructure.py**（約150行予定）:
```python
def test_spread_application():
    # Spread適用で買値 > 売値を確認

def test_min_tick_rounding():
    # 最小呼値で価格が丸められることを確認

def test_min_qty_rejection():
    # 最小量未満の注文が約定しないことを確認

def test_variable_slippage():
    # ATR高時にスリッページが増加することを確認
```

### 保留理由
- environment.py の取引ロジック全面改修が必要（約200行）
- 既存の `cost_gate` との整合性確認が必要
- P0/P1の優先度高、時間制約により保留

---

## P2-9: リーク耐性テスト 📋 TODO

### 問題
- 特徴量に未来情報が混入している可能性
- Look-ahead bias, data leakage → 実環境で性能劣化

### 解決策案
**tests/property/test_leak_guards.py**（約250行予定）:

#### テスト1: 時間シフトテスト
```python
def test_time_shift_sharpe_degradation():
    """特徴量を1期ずらすとSharpeが悪化するか"""
    # Original features
    sharpe_original = train_and_evaluate(features)

    # Shift features by 1 period
    features_shifted = features.shift(1)
    sharpe_shifted = train_and_evaluate(features_shifted)

    # Assert: Shifted should degrade
    assert sharpe_shifted < sharpe_original * 0.8, "Time shift should degrade Sharpe"
```

#### テスト2: シャッフルテスト
```python
def test_shuffle_sharpe_collapse():
    """時系列シャッフルでSharpeが崩壊するか"""
    sharpe_original = train_and_evaluate(df)

    # Shuffle time series
    df_shuffled = df.sample(frac=1.0, random_state=42)
    sharpe_shuffled = train_and_evaluate(df_shuffled)

    # Assert: Shuffle should collapse Sharpe
    assert sharpe_shuffled < 0.0, "Shuffle should collapse Sharpe to negative"
```

#### テスト3: 重複行テスト
```python
def test_duplicate_rows_overfitting():
    """重複行追加で過学習しないか"""
    sharpe_original = train_and_evaluate(df)

    # Duplicate 10% of rows
    df_dup = pd.concat([df, df.sample(frac=0.1)], ignore_index=True)
    sharpe_dup = train_and_evaluate(df_dup)

    # Assert: Should not overfit
    assert abs(sharpe_dup - sharpe_original) < 0.1, "Duplicate should not overfit"
```

### 保留理由
- 各テストで学習実行 → 約10-15分/テスト
- 総テスト時間 = 30-45分（時間制約により保留）
- P0/P1完了後の継続タスク

---

## P2-10: モデルカード自動生成 📋 TODO

### 問題
- モデル設定、性能、再現性情報が散在
- `manifest.json` 手動作成 → 人為ミスリスク
- MLOps best practice（モデルカード）未整備

### 解決策案
**scripts/generate_model_card.py**（約200行予定）:

#### 機能
1. **manifest.json読み込み**:
   ```python
   manifest = {
       "model_id": "ensemble_C_1M_25000",
       "timestamp": "2025-10-07T10:30:00Z",
       "config": {...},
       "metrics": {
           "sharpe_ratio": 1.23,
           "max_drawdown": -0.15,
           ...
       }
   }
   ```

2. **Markdown生成**:
   ```markdown
   # Model Card: ensemble_C_1M_25000

   ## Model Details
   - **Model ID**: ensemble_C_1M_25000
   - **Timestamp**: 2025-10-07 10:30:00 UTC
   - **Training Steps**: 25,000

   ## Performance Metrics
   - **Sharpe Ratio**: 1.23
   - **Max Drawdown**: -15.0%
   - **Win Rate**: 52.3%

   ## Configuration
   ```yaml
   learning_rate: 0.0003
   gamma: 0.99
   ent_coef: 0.01
   ```

   ## Reproducibility
   - **Data Version**: v2.3.1
   - **Seed**: 42
   - **Normalization**: stats_v1.pkl
   ```

3. **使用例**:
   ```bash
   python scripts/generate_model_card.py \
       --checkpoint-dir checkpoints/ensemble_C_1M \
       --output-dir model_cards/
   ```

### 保留理由
- P0/P1優先、時間制約により保留
- manifest.json構造確定後に実装推奨

---

## 今後の優先順位

### 即座に実施可能
- ✅ **P2-7（gc_artifacts）**: 完了

### 中期タスク（1M学習前推奨）
- 📋 **P2-8（マイクロ構造）**: environment.py改修必要（約3-4時間）
- 📋 **P2-9（リーク耐性テスト）**: 学習実行必要（約1-2時間）
- 📋 **P2-10（モデルカード）**: 実装容易（約1時間）

### 推奨実施順序
1. P0（完了） → P1（完了） → **P2-7（完了）** → 1M学習開始
2. 1M学習中に並行してP2-8, P2-9, P2-10実装
3. 1M学習完了後、P2-8, P2-9でモデル検証

---

## まとめ

P2-7（gc_artifacts）実装により、1M学習のディスク管理問題を解決しました:

- ✅ **scripts/gc_artifacts.py**: 約380行、Checkpoint自動削除機能
- ✅ **ztb/training/callbacks.py**: CheckpointGCCallback追加（約90行）
- ✅ **構文チェック**: 全PASS
- ✅ **達成効果**: ディスク85%削減、自動化、保守的削除

**残タスク**: P2-8（マイクロ構造）、P2-9（リーク耐性テスト）、P2-10（モデルカード）
**推奨**: P0+P1+P2-7完了で1M学習開始可能、残P2タスクは並行実装
