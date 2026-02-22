# 40. Phase 4 計画: Walk-Forward最適化 & システム統合（修正版）

**日付**: 2026-01-27（2026-01-27 レビュー反映修正）  
**前提**: Phase 3.5完了（特徴生成最適化 99.8%削減達成）  
**v459大義**: 「**短期間での高収益性システム**」の実現（0番プロポーザル）  
**Phase 4目標**: WF評価の独立性を保ちつつ、実験効率化 + システム統合  
**設計原則**: シンプル化、検証、統合（0番 Section 3.1）  
**重要**: 41番レビューの指摘を反映し、評価の整合性を優先

---

## 0番プロポーザルとの関係

Phase 4は0番プロポーザルの**Phase 3 + Phase 4前半**に相当します：

| 0番フェーズ | 対応内容 | 完了状況 |
|------------|----------|----------|
| Phase 0-2 | 仕様固定 + P0/P1バグ修正 | ✅ 完了 |
| **Phase 3** | **報酬設計の段階検証** | 🟡 本Phase 4で実施 |
| **Phase 4前半** | **評価・検証** | 🟡 本Phase 4で実施 |
| Phase 4後半-5 | Paper Trading統合 | ⏳ Phase 5以降 |

**Phase 4の位置づけ**: 
- 速度最適化ではなく、**収益性検証のための基盤整備**
- 0番の成功基準（Gate 1-4）をクリアするための技術基盤構築

---

## Phase 4の前提条件（42番再レビュー対応）

**重要**: 以下の3点がPhase 4実装の**必須条件**です。これらが未実装の場合、「計画は正しいが実装が動かない」リスクがあります。

### 1. MTF強制有効化の解除（必須）

**現状**: `initialization.py:277-279`で強制有効化されている  
**前提**: Week 1 Day 1で**コード修正コミットが必須**  
**検証**: 設定ファイルで`include_multi_timeframe_features: false`が機能すること

```python
# 削除必須：
if not include_mtf:
    logger.info("Forcing enable of multi-timeframe features (v455 requirement)")
    include_mtf = True  # ← この3行を完全削除
```

**注意**: 設定だけでは無効化できません。コード修正が必須です。

### 2. Parquet統合経路の一本化（必須）

**現状**: データ読み込み経路が2系統に分散  
- SACTrainer: `_load_data_with_format_detection()` (実装済み)  
- UnifiedTrainer: 読み込み経路が不明確  

**前提**: Week 1 Day 2-3で**「訓練データ読込は必ずこの関数を通る」設計保証**  
**目標**: SAC/UnifiedTrainerの二重経路を一本化

```python
# UnifiedTrainerの必須実装
class UnifiedTrainer:
    def setup_data(self):
        """setup_dataが必ずload_dataを呼ぶことを保証"""
        data_path = self.config.data_path
        self.data = self.load_data(data_path)  # ← この経路が必須
        # ... 既存の処理継続 ...
```

**検証方法**:
1. SACでParquet読み込みが機能することを確認（✅ 実装済み）
2. UnifiedTrainerでも同じパスでParquetが読み込まれることを確認
3. ABRewardExperimentのパス切り替えがすべてのアルゴリズムで機能することを確認

### 3. 特徴検出ロジックの厳密化（必須）

**現状**: `feature_cols >= 5`だけで判定（誤検出リスク）  
**前提**: Week 1 Day 2-3で**必須列チェックを厳密化**

```python
# 改善後：厳密な検出
def _has_precomputed_features(self, df: pd.DataFrame) -> bool:
    """事前計算特徴の存在を厳密に検証"""
    # 1. 必須列の存在確認
    required_ohlcv = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
    if not required_ohlcv.issubset(df.columns):
        return False
    
    # 2. 特徴列のカウント
    feature_cols = [c for c in df.columns if c not in required_ohlcv]
    
    # 3. 明示的なフラグがあれば優先（将来拡張）
    if hasattr(self.config, 'precomputed_features'):
        return self.config.precomputed_features
    
    # 4. 特徴数で判定（保守的な閾値）
    return len(feature_cols) >= 5
```

**検証方法**:
- OHLCVのみのParquetで誤検出しないことを確認
- 8特徴Parquetで正しく検出することを確認
- 不完全な特徴データ（3列のみ）で拒否されることを確認

---

## Phase 3.5の成果（実測値で確定）

**検証完了**: 2026-01-27、12実験で実測  
**詳細**: [42番検証結果](42_phase3.5_verification_results.md)

### 達成した最適化効果（実測値）
- **特徴生成時間**: 466秒 → **0.79秒**（99.83%削減、589倍高速化）
- **総トレーニング時間**: 720秒 → **201.9秒**（71.9%削減、3.6倍高速化）
- **メモリ使用量**: 970MB → 590MB（38%削減、ピーク値）
- **12実験総時間**: 144分 → **80.8分**（43.9%削減）
- **成功率**: **100%**（12/12実験成功、エラー率0%）

**重要**: 当初の暫定値（46分）より80.8分は長いですが、これはPhase 3.5**前**（144分）と比較して**43.9%の大幅削減**です。

### 使用技術
- Parquet形式による特徴量事前計算
- 相関ベース特徴選択（**8特徴に削減は仮説**、Week 1でA/B検証必須）
- 自動Parquet検出と特徴生成スキップ（SACTrainer実装済み、UnifiedTrainerはWeek 1で実装）

**重要**: 8特徴への削減は「速度最適化のための仮説」であり、**収益性への影響は未検証**です。Phase 4 Week 1でフル特徴版との比較を必ず実施します。

---

## Phase 4の目標

### 1. Walk-Forward最適化（優先度: 高）

**現状分析**:
- 4 splits × 230秒/split = 920秒/実験
- シーケンシャル実行のため並列化余地あり
- Window間でデータ重複が多い

**最適化案**:

#### 1.1 Window並列実行（オプション、Phase 4後半で検討）

**重要**: Windows環境では ProcessPoolExecutor の spawn 方式により SIGINT 再発・メモリスパイクのリスクがあります。Phase 4前半では**実験単位の分離実行**を優先し、並列化は安定化後に検討します。

```python
# 現状: シーケンシャル（安全・評価独立性が保証される）
for window in range(4):
    train()  # 230秒
    validate()
    
# オプション: 並列実行（Phase 4後半、リスク評価後）
# Windows環境での動作検証とメモリ監視が前提
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(train_window, i) for i in range(4)]
    results = [f.result() for f in futures]
```

**期待効果**: 920秒 → 250秒（理論値、Windows環境での検証必須）  
**優先度**: 低（Phase 4前半では実装せず、安定化を優先）

#### 1.2 Window間データキャッシュ（WF独立性ガード付き）

**重要**: Walk-Forwardの評価独立性を保つため、キャッシュできるのは**OHLCVと未来情報を使わない生特徴のみ**です。スケーリング・相関削減・正規化統計は各Window内で再計算します。

```python
# Window 1: [0:60%] train, [60:80%] val, [80:100%] test
# Window 2: [25:85%] train, [85:105%] val, [105:125%] test
# → [25:60%]は重複データ

# キャッシュ戦略（WF独立性ガード + キー設計）
class WFSafeCache:
    """Walk-Forward評価の独立性を保証するキャッシュ"""
    
    def _make_cache_key(self, data_version: str, data_range: tuple) -> str:
        """キャッシュキーの生成（独立性に影響しない要素のみ）"""
        # キー要素:
        # - data_version: データファイルのバージョン/更新日時
        # - data_range: (start_date, end_date) データの期間
        # ❌ 含めてはいけない: feature_set, correlation_threshold, scaling_params
        #    （これらはWindow毎に独立計算すべき）
        return f"{data_version}_{data_range[0]}_{data_range[1]}"
    
    def get_cacheable_data(self, window, data_version: str):
        """キャッシュ可能なデータの取得"""
        cache_key = self._make_cache_key(data_version, window.date_range)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # キャッシュミス: OHLCV + 生の技術指標を読み込み
        data = load_ohlcv_and_raw_features(window)
        self.cache[cache_key] = data
        return data
    
    def compute_per_window(self, window, raw_data):
        """Window毎に独立計算（キャッシュ不可）"""
        # Window毎に再計算必須:
        # - スケーリング（trainデータのmin/maxを使用）
        # - 相関ベース特徴削減（trainデータの相関行列を使用）
        # - 正規化統計（trainデータの平均/標準偏差を使用）
        return scale_and_select_features(
            raw_data, 
            stats_from=window.train_data
        )

window_cache = WFSafeCache()
for window in windows:
    raw_data = window_cache.get_cacheable_data(window)
    processed_data = window_cache.compute_per_window(window, raw_data)
```

**期待効果**: OHLCV読み込み時間 30-40%削減（特徴計算は各Window独立実行）  
**優先度**: 中（WF独立性を侵害しない範囲で実装）

#### 1.3 増分学習（別実験枠、Phase 4後半で検討）

**重要**: 増分学習は**Walk-Forwardの独立評価ではなくローリング継続学習**です。0番で定義した評価ベースラインと混在させず、**別実験タグで独立実行**します。

```python
# ❌ Baseline WFと混在させない（評価の整合性が崩れる）
for window in baseline_wf:
    model = train_from_scratch()  # 各Window独立

# ✅ 別実験タグで独立実行
for window in incremental_experiment:
    if window == 0:
        model = train_from_scratch()
    else:
        model = copy.deepcopy(previous_model)
        model.continue_training(new_data)
```

**実装方針**:
- Baseline WFとは別の実験タグ（例: `incremental_wf`）で管理
- 結果の比較時に明示的に分離
- ドキュメントに「評価方法の違い」を明記

**期待効果**: 収束時間 30-40%削減（理論値、評価方法変更を伴う）  
**優先度**: 低（Baseline WF評価の安定化後に検討）

### 2. MTF制御の完全実装（優先度: 高、Phase 4前半で必須）

**Phase 3.5で保留した課題**:
```python
# ztb/trading/environment/heavy_env/mixins/initialization.py:277-279
include_mtf = feature_flags.get("include_multi_timeframe_features", False)
if not include_mtf:
    logger.info("Forcing enable of multi-timeframe features (v455 requirement)")
    include_mtf = True  # ← この強制有効化を削除
```

**修正計画**:
1. **強制有効化の削除**（コード修正必須）
   ```python
   # 修正後
   include_mtf = feature_flags.get("include_multi_timeframe_features", True)
   # 強制有効化は削除、デフォルトはTrue（後方互換性）
   ```

2. **設定ファイルでの無効化確認**
   ```yaml
   # config.yaml
   feature_flags:
     include_multi_timeframe_features: false  # 無効化が機能することを確認
   ```

3. **単体テストの追加**
   - MTF有効時と無効時の動作を検証
   - 特徴数の変化を確認

**影響範囲**: 
- ✅ Phase 3.5のParquet最適化は影響なし（特徴生成スキップが機能）
- ⚠️ 設定のみの無効化は現状不可能（コード修正が前提）

### 3. UnifiedTrainer統合（優先度: 高、Phase 4前半で必須）

**現状**: SACTrainerのみParquet対応、ABRewardExperimentのパス切り替えは効果なし

**問題点**: 
- ABRewardExperimentでParquetパスを設定しても、UnifiedTrainerが読み込めない
- 現在の実装ではCSV→Parquet自動切り替えが機能していない
- **データ読み込み経路が2系統に分散**（SAC側とUnifiedTrainer側）

**必須条件（42番再レビュー）**:
1. **「訓練データ読込は必ずこの関数を通る」設計保証**
2. SAC/UnifiedTrainerの二重経路を一本化

**拡張計画**:
```python
# ztb/training/unified_trainer/trainer.py
class UnifiedTrainer:
    def load_data(self, data_path: str) -> pd.DataFrame:
        """CSV/Parquet自動検出読み込み"""
        from pathlib import Path
        
        path = Path(data_path)
        if path.suffix == '.parquet':
            return self._load_parquet(data_path)
        else:
            return self._load_csv(data_path)
    
    def _load_parquet(self, path: str) -> pd.DataFrame:
        """Parquet読み込み + 特徴検出（厳密化）"""
        from ztb.cache.parquet_io import read_parquet
        df = read_parquet(path)
        
        # 42番再レビュー: feature_cols >= 5 だけでは誤検出リスク
        # 対策: 必須特徴の存在確認を追加
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        feature_cols = [c for c in df.columns if c not in ohlcv_cols]
        
        # 想定している8特徴（またはそのサブセット）の存在確認
        expected_features = {'rsi', 'macd', 'bb_width', 'volatility', 'momentum', 'volume_ma_ratio', 'atr', 'obv'}
        detected_features = set(feature_cols) & expected_features
        
        if len(feature_cols) >= 5 and len(detected_features) >= 3:
            logger.info(f"Detected {len(feature_cols)} precomputed features ({len(detected_features)} known), skipping generation")
            self.config.feature_set = "minimal"  # 特徴生成スキップ
        else:
            logger.warning(f"Only {len(detected_features)} expected features found, will generate features")
        
        return df
    
    def _load_csv(self, path: str) -> pd.DataFrame:
        """既存のCSV読み込み"""
        df = DataLoader.load_csv_optimized(path)
        if "timestamp" in df.columns:
            df["timestamp"] = safe_to_datetime_series(df["timestamp"])
        return df
```

**対象アルゴリズム**: SAC（既存）、PPO, DQN, A2C（Phase 4で追加）

**実装ファイル**: `ztb/training/unified_trainer/trainer.py`

**既存コードとの統合**:
```python
# 既存の DataLoader を活用
from ztb.io.data_loader import DataLoader
from ztb.features.generators.multi_timeframe.datetime_utils import safe_to_datetime_series
from ztb.cache.parquet_io import read_parquet

class UnifiedTrainer:
    def __init__(self, config):
        # ... 既存の初期化 ...
        self._data_cache = {}  # データキャッシュ
    
    def setup_data(self):
        """既存メソッドを拡張"""
        data_path = self.config.data_path
        self.data = self.load_data(data_path)  # ← 新規メソッド追加
        # ... 既存の処理継続 ...
```

**後方互換性の保証**:
- CSV読み込みは既存の `DataLoader.load_csv_optimized()` を使用
- Parquet検出時のみ新しい `_load_parquet()` を使用
- `feature_set` の変更はオプトイン方式

**検証方法**:
1. UnifiedTrainerでParquetを読み込み
2. 特徴生成時間が1秒以下になることを確認
3. 既存のCSV読み込みが影響を受けないことを確認
4. PPO/DQN/A2Cでの動作確認

### 4. メモリ最適化（優先度: 低）

**現状**: 590MB（許容範囲内）

**将来的な最適化案**:
- チャンク処理による大規模データセット対応
- 不要な中間結果の即時破棄
- Experience Replayバッファサイズの動的調整

---

## Phase 4実装順序（修正版）

### Week 1: 基盤統合（最優先、評価の整合性確保）
- [ ] Day 1: **MTF制御修正**（強制有効化削除、テスト追加）
- [ ] Day 2-3: **UnifiedTrainer Parquet対応**（load_data統合、特徴検出）
- [ ] Day 4: **WFセーフキャッシュ設計**（独立性ガード実装、キー設計）
- [ ] Day 5: **統合テスト・Phase 3.5数値の再検証 + A/B検証開始**
  - 8特徴版 vs フル特徴版の比較実験（2実験×2パターン = 最低4実験）
  - **0番 Phase 3連携**: 報酬設計段階検証の準備（Stage 1: 純PnLベースライン）

**目標**: 
- 全TrainerでParquet使用可能
- WF評価の独立性保証
- Phase 3.5成果の12実験での検証
- **8特徴削減の妥当性検証**（収益率・Sharpe比への影響を測定）
- **0番 Phase 3への接続**: 報酬設計ABテストの基盤整備

### Week 2: WF最適化 + ベースライン検証
- [ ] Day 1-2: **WFセーフキャッシュ実装**（OHLCV + 生特徴のみ）
- [ ] Day 3-4: **実験単位の分離実行**（メモリ・SIGINT対策）
- [ ] Day 5: **ベースライン比較実装 + 統計検定**
  - Buy-and-Hold, SMA Crossover, Random Actionの3種
  - Mann-Whitney U検定 + Holm-Bonferroni補正
  - **0番 Gate 2基準**: 必須超過の検証

**目標**: 
- 評価の整合性を保ちつつ、データ読み込み時間30%削減
- 12実験の安定実行（失敗率<1%）
- **0番ベースライン比較**: 3種で統計的有意に超過することを確認

### Week 3: 報酬設計検証 + Go/No-Go準備
- [ ] Day 1-2: **0番 Phase 3実装**: 報酬設計の段階検証
  - Stage 1: 純PnL only（ベースライン）
  - Stage 2: PnL + Trend Guidance（固定重み）
  - Stage 3: PnL + Trend Guidance（Decay付き）
  - 各Stageで1実験×4seed = 8実験
- [ ] Day 3-4: **並列化の可否検討**（Windows環境での検証、オプション）
- [ ] Day 5: **Go/No-Go準備**: 0番 Gate 1-4基準の総合検証
  - Gate 1: 技術検証（WF独立性、指標整合性）
  - Gate 2: 収益性検証（ROI > 5%, PF > 1.20, Sharpe > 1.0）
  - Gate 3: リスク管理（DD < 15%, 連敗縮退）
  - Gate 4: 実行コスト（手数料・スリッページ）

**目標**: 
- 0番 Phase 3の完了（報酬設計段階検証）
- Phase 5施策の準備（並列化は慎重に検討）
- **0番 Go/No-Go判定**: 全Gate通過でPhase 4完了

---

## 成功指標（Phase 4完了時）

### パフォーマンス（実験効率）
- [ ] 12実験時間: **Phase 3.5数値の再検証後に目標設定**（暫定: 40分以内）
- [ ] メモリ使用: 600MB以下維持
- [ ] データ読み込み時間: 30%削減（WFセーフキャッシュ）
- [ ] CPU効率: シングルプロセス最適化（並列化はPhase 5検討）

### 収益性（0番 Gate 2基準、最重要）

**注意**: Phase 4の本質は速度ではなく**収益性検証のための基盤整備**です。0番 Section 5.2の基準をクリアすることがPhase 4の最終目標です。

| 指標 | 最低基準（Go判定） | 目標基準 | 測定条件 |
|------|-----------------|----------|----------|
| **Net ROI** | **> 5%** | > 15% | 年率換算、コスト込み |
| **Profit Factor** | **> 1.20** | > 1.50 | 手数料・スリッページ後 |
| **Sharpe Ratio** | **> 1.0** | > 1.5 | 日次リターン、年率換算 |
| **Max Drawdown** | **< 15%** | < 10% | 高値からの最大下落 |
| **Win Rate** | **> 35%** | > 45% | 手数料込み勝敗 |
| **期待値/取引** | **> ¥500** | > ¥1,000 | コスト控除後 |

**出典**: 0番 Section 5.2 - 収益性検証（Gate 2: Go/No-Go判定軸）

### ベースライン比較（0番 Gate 2、必須）

| 比較対象 | 条件 | 判定 | 統計検定 |
|----------|------|------|----------|
| **Buy-and-Hold** | 同期間、同コスト、同ポジション | **必須超過** | 必須 |
| **SMA Crossover** | 20/50期間、同条件 | **必須超過** | 必須 |
| **Random Action** | 同頻度、同コスト | **必須超過** | 必須 |
| **Momentum (1h)** | 1時間リターン追従 | 参考 | 任意 |

**出典**: 0番 Section 5.5 - ベースライン比較（最終判定基準）

**統計検定仕様**（0番 Section 5.6）:
- Mann-Whitney U検定（ノンパラメトリック）
- 有意水準 α = 0.05
- Holm-Bonferroni法（3比較の多重比較補正）
- Cliff's Delta（|d| > 0.33で中程度の効果）
- サンプル数 n ≥ 16（4seed × 4split）
**検出力評価（42番再レビュー）**:
- n=16で中程度の効果（d=0.5）を検出できる確率: ~80%
- 8特徴削減で収益分散が増加する場合、検出力が低下する可能性
- **対策**: Week 1でフル特徴版との分散比較を先行実施
### 技術品質
- [ ] 実験再現性: **統計的再現性**（収益率±5%、Sharpe比±0.2の範囲内）
  - **根拠（42番再レビュー指摘）**: Phase 3.5の12実験で総時間のバラツキが小さい（±3.6%）ことを確認済み
  - 但し、収益率の再現性はPhase 3.5では未検証
  - **Week 1 Day 5**: 8特徴A/B検証で収益率の分散を実測し、±5%目標の妥当性を評価
  - 分散が大きい場合、サンプル数増強またはシード固定を検討
- [ ] エラー率: <1%（12実験中11以上成功）
  - **実績**: Phase 3.5で12/12成功（エラー率0%）
- [ ] コードカバレッジ: **Phase 4対象モジュール60%以上**（全体80%は非現実的）

### 評価の整合性（0番 Gate 1基準）
- [ ] WF独立性: 各Windowのスケーリング・正規化が独立計算されていることを確認
- [ ] Baseline WF: 増分学習と明確に分離されていることを確認
- [ ] 特徴量削減の影響: 8特徴版 vs フル特徴版のA/B比較実施

**注意**: 速度最適化よりも評価の正当性と収益性を優先します。

### ドキュメント
- [ ] Phase 4完了レポート
- [ ] 最適化ガイド（ベストプラクティス集）
- [ ] トラブルシューティングガイド

---

## リスク管理

### 評価整合性リスク（最優先）

**WFキャッシュによる未来情報リーク**:
- リスク: Window間でスケーリング統計を共有すると評価が歪む
- 対策: OHLCV + 生特徴のみキャッシュ、統計は各Window独立計算
- 検証: 各Windowのスケーリングパラメータが異なることを確認

**増分学習とBaseline WFの混在**:
- リスク: 評価方法の違いを混同すると0番との整合性が崩れる
- 対策: 別実験タグで明確に分離、ドキュメントに評価方法の違いを明記

**特徴量削減による収益性劣化**:
- リスク: 8特徴への削減が過度で、収益性が大幅低下
- 対策: Phase 4前半でフル特徴版とのA/B比較を実施

### 技術リスク

**Windows環境でのSIGINT再発**:
- リスク: Phase 3.5でSIGINT問題を経験済み、並列化で再発の可能性
- 対策: Phase 4前半では並列化を実装せず、実験単位の分離実行を優先

**Parquet読み込み経路の未統合**:
- リスク: UnifiedTrainer統合が遅れると、Phase 3.5の最適化が活用できない
- 対策: Week 1で最優先実装、統合テストを実施

**メモリリーク**:
- リスク: 長時間実験でメモリ使用量が増加
- 対策: 各Window終了時に明示的なガベージコレクション

### スケジュールリスク

**Phase 3.5数値の過大評価**:
- リスク: 暫定数値をもとに目標設定すると、達成不可能になる
- 対策: Week 1で12実験を完了させ、実測値をもとに目標を再設定

**想定外のバグ**:
- 対策: 各機能を独立してテスト、段階的統合、フィーチャーフラグ使用

---

## Phase 5への展望（Phase 4結果を踏まえて）

### Phase 4で得られる知見を活用した次施策

**A. Phase 4で並列化が成功した場合**:
1. **実験バッチの並列実行**: 12実験を4プロセスで分割実行（3実験×4並列）
2. **増分学習の本格導入**: Baseline WFとの性能比較が明確になった後に本格採用
3. **複数データセットの同時実験**: BTC/ETH/XRPなど複数通貨ペアの並列実験

**B. Phase 4で並列化が困難だった場合**:
1. **モデル学習の内部最適化**: Buffer size, gradient steps, learning rateの最適化
2. **特徴生成のさらなる削減**: 5特徴まで削減（A/B検証必須）
3. **データローダーの最適化**: メモリマップドファイル、lazy loading

**C. 8特徴削減で収益性が維持された場合**:
1. **動的特徴選択**: 市場状況に応じた特徴セットの切り替え
2. **特徴重要度分析の自動化**: SHAP値による定期的な特徴見直し
3. **軽量モデルへの移行**: より高速な学習アルゴリズムの検討

**D. 8特徴削減で収益性が劣化した場合**:
1. **中間サイズの特徴セット**: 15-20特徴での最適化
2. **特徴エンジニアリングの改善**: より情報量の多い合成特徴の作成
3. **アンサンブル手法**: 複数の特徴セットを使う複数モデルの統合

### 本番運用への段階的移行

1. **Phase 4後半**: ペーパートレーディング開始（実データ、仮想資金）
2. **Phase 5前半**: 小額実運用（リスク限定、継続的監視）
3. **Phase 5後半**: 本格運用展開（Phase 4の最適化を実戦投入）

### インフラ整備（Phase 4結果次第で優先度決定）

1. **実験管理**: MLflow導入（Phase 4で実験数が増えた場合に必須）
2. **モニタリング**: Grafana + Prometheus（本番運用開始時に必須）
3. **CI/CD**: GitHub Actions（Phase 4で安定性が確認できた場合）

**判断基準**: Phase 4完了時点で、最も効果が高かった施策を Phase 5で深掘りする

---

## 参考資料

- **[0番: Project Proposal v459](00_project_proposal_v459.md)** - v459全体計画、成功基準、Gate 1-4
- [Phase 4計画レビュー（41番）](41_phase4_planning_review.md) - レビュー指摘事項
- [Phase 3.5実装レポート（39番）](39_review_response_修正計画.md) - 前フェーズの成果
- [特徴生成最適化実装](../../scripts/v459/precompute_optimized_features.py) - 8特徴削減の実装
- [AB実験ランナー](../../scripts/v459/run_ab_reward_experiments.py) - 実験実行スクリプト
- [SACTrainer Parquet対応](../../ztb/training/unified_trainer/algorithms/sac_trainer.py) - 既存実装の参考
