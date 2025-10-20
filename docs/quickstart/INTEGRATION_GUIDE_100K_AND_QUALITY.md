# 100kテスト & コード品質改善 - 統合ガイド

**日付**: 2025年10月7日
**目的**: 1M学習前の100kテスト実施 + 型安全性・保守性向上

---

## 📋 提供ドキュメント一覧

### 1. **QUICKSTART_100K_TEST.md** - 100kテスト実行ガイド

**用途**: 1M学習前の動作確認・パラメータ調整
**所要時間**: 15-30分（並列実行時）
**成果物**:
- 3モデルの学習結果（10チェックポイント/モデル）
- TensorBoardログ
- パラメータ調整の知見

**クイックスタート**:
```bash
# モデルA（Conservative）のみテスト
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json
```

---

### 2. **CODE_QUALITY_IMPROVEMENT_PLAN.md** - コード品質改善計画

**用途**: 型安全性向上、保守性強化、不整合解消
**対象**: zaif-trade-bot全体（特にztb/training/）
**期間**: 1-2週間

**5つのPhase**:
1. **型安全性検査**（優先度: 🔴 高）
   - 型ヒント追加
   - type: ignore 削減
   - Protocol/ABC 導入

2. **設定ファイル不整合検査**（優先度: 🟠 中）
   - パラメータ命名統一
   - デフォルト値統一
   - 必須パラメータ明示化

3. **インターフェース統一**（優先度: 🟠 中）
   - Trainerインターフェース統一
   - Callbackインターフェース統一

4. **ドキュメント整備**（優先度: 🟡 低）
   - Docstring追加
   - 型スタブファイル作成

5. **テストカバレッジ向上**（優先度: 🟢 推奨）
   - 型安全性テスト
   - 設定ファイルバリデーションテスト

**クイックスタート**:
```bash
# Step 1: 現状分析
mypy --strict ztb/ > mypy_strict_report.txt
python scripts/check_config_consistency.py > config_inconsistencies.txt

# Step 2: 優先度付け
# CODE_QUALITY_IMPROVEMENT_PLAN.md を参照

# Step 3: 段階的改善
# Week 1: 型安全性（Phase 1）
# Week 2: 設定・インターフェース（Phase 2-3）
```

---

### 3. 100kテスト用設定ファイル（3種）

**作成済み**:
- `configs/train/ensemble_A_100k_test.json` - Conservative（ent_coef=0.6, SELL=0.8）
- `configs/train/ensemble_B_100k_test.json` - Moderate（ent_coef=0.7, SELL=0.9）
- `configs/train/ensemble_C_100k_test.json` - Aggressive（ent_coef=0.8, SELL=1.0, reverse=True）

**特徴**:
- `total_timesteps: 100000`（1Mの1/10で高速テスト）
- `checkpoint_interval: 10000`（10チェックポイント）
- CustomPPO統合（PAN + Target Entropy）

---

## 🚀 推奨ワークフロー

### Week 1: 100kテスト（優先）

**Day 1-2: 単一モデルテスト**
```bash
# モデルBでテスト（最もバランスが良い）
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json
```

**成功基準**:
- ✅ 学習完了（エラーなし）
- ✅ CustomPPO動作（pan_total_samples > 0）
- ✅ SELL発生（legal_sell_rate ≥ 0.05）

**Day 3-4: 3モデル並列テスト**
```powershell
# 並列実行（所要時間: 15-30分）
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_A_100k_test.json"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m ztb.training.unified_trainer --config configs\train\ensemble_C_100k_test.json"
```

**Day 5: 結果分析**
- TensorBoardで3モデル比較
- 最良パラメータ特定
- 1M学習の設定調整

---

### Week 2-3: コード品質改善（並行作業）

**Phase 1: 型安全性向上**（Week 2前半）
```bash
# 型ヒント追加開始
# 他のCopilotに指示: CODE_QUALITY_IMPROVEMENT_PLAN.md Phase 1を実施
```

**Phase 2: 設定ファイル統一**（Week 2後半）
```bash
# 設定ファイル不整合検出
python scripts/check_config_consistency.py

# 統一作業
# 他のCopilotに指示: CODE_QUALITY_IMPROVEMENT_PLAN.md Phase 2を実施
```

**Phase 3: インターフェース統一**（Week 3）
```bash
# TrainerParams導入
# 他のCopilotに指示: CODE_QUALITY_IMPROVEMENT_PLAN.md Phase 3を実施
```

---

### Week 4: 1M学習実施

**前提条件**:
- ✅ 100kテスト成功（Week 1完了）
- ✅ 型安全性改善完了（Phase 1完了）
- ✅ 設定ファイル統一完了（Phase 2完了）

**実行**:
```bash
# 1M学習開始（並列実行推奨）
python -m ztb.training.unified_trainer --config configs\train\ensemble_A_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_B_1M.json
python -m ztb.training.unified_trainer --config configs\train\ensemble_C_1M.json
```

**所要時間**: 3-5時間/モデル

---

## 📊 Copilotへの指示文例

### 指示1: 100kテスト実行

```
# Copilot指示: 100kテスト実行

QUICKSTART_100K_TEST.md を参照し、以下を実施してください:

1. 前提条件確認:
   - ml-dataset-enhanced.csv 存在確認
   - 設定ファイル3種確認

2. モデルB（Moderate）で100kテスト実行:
   python -m ztb.training.unified_trainer --config configs\train\ensemble_B_100k_test.json

3. TensorBoard起動・監視:
   tensorboard --logdir logs --port 6006

4. 重要指標確認:
   - train/pan_total_samples > 0 か?
   - train/legal_sell_rate ≥ 0.05 か?
   - train/grad_norm(SELL) ≠ 0 か?

5. 成功基準チェック:
   - 学習完了（エラーなし）
   - CustomPPO動作確認
   - SELL発生確認

6. 結果レポート作成
```

---

### 指示2: 型安全性向上（Phase 1）

```
# Copilot指示: 型安全性向上

CODE_QUALITY_IMPROVEMENT_PLAN.md の Phase 1 を実施してください:

1. 現状分析:
   mypy --strict ztb/training/ > mypy_strict_report.txt

2. 型ヒント追加:
   - ztb/training/ppo_trainer.py
   - ztb/training/unified_trainer.py
   - ztb/training/custom_ppo.py

3. type: ignore 削減:
   - 具体的なエラーコード指定
   - コメントで理由説明

4. Protocol導入:
   - ztb/training/protocols.py 作成
   - Trainer, Environment, Callback のProtocol定義

5. 検証:
   mypy --strict ztb/training/

目標: mypy エラー50%削減
```

---

### 指示3: 設定ファイル不整合解消（Phase 2）

```
# Copilot指示: 設定ファイル不整合解消

CODE_QUALITY_IMPROVEMENT_PLAN.md の Phase 2 を実施してください:

1. 不整合検出:
   python scripts/check_config_consistency.py

2. パラメータ命名統一:
   - スネークケースに統一
   - 省略形を避ける
   - 階層をトップレベルに統一

3. デフォルト値統一:
   - ztb/config/constants.py 作成
   - 全ファイルで同じ定数参照

4. スキーマ定義:
   - ztb/config/schemas.py 作成
   - TypedDict で設定スキーマ定義
   - Required で必須パラメータ明示

5. バリデーション:
   python scripts/validate_all_configs.py

目標: 全設定ファイルでバリデーション全パス
```

---

### 指示4: インターフェース統一（Phase 3）

```
# Copilot指示: インターフェース統一

CODE_QUALITY_IMPROVEMENT_PLAN.md の Phase 3 を実施してください:

1. TrainerParams定義:
   @dataclass
   class TrainerParams:
       data_path: str
       checkpoint_dir: str
       checkpoint_interval: int = 10000

2. 全トレーナーでインターフェース統一:
   - PPOTrainer
   - SELLBiasMitigationPPOTrainer
   - PPOTrainerAutoHalt

3. Callbackインターフェース統一:
   - on_step() シグネチャ統一
   - 戻り値の型統一

4. テスト追加:
   - tests/training/test_trainer_interface.py

目標: 全トレーナーで統一インターフェース採用
```

---

## ✅ チェックリスト

### 100kテスト（Week 1）
- [ ] モデルBで100kテスト成功
- [ ] 3モデル並列テスト成功
- [ ] TensorBoard分析完了
- [ ] 結果レポート作成
- [ ] 1M学習用パラメータ決定

### コード品質改善（Week 2-3）
- [ ] Phase 1完了（型安全性向上）
- [ ] Phase 2完了（設定ファイル統一）
- [ ] Phase 3完了（インターフェース統一）
- [ ] mypy --strict 全パス
- [ ] 設定ファイルバリデーション全パス

### 1M学習準備（Week 4）
- [ ] 100kテスト成功
- [ ] コード品質改善完了
- [ ] 1M学習用設定ファイル調整
- [ ] ディスク容量確認（5GB以上）
- [ ] 実行環境確認

---

## 📚 関連ドキュメント

1. **QUICKSTART_100K_TEST.md** - 100kテスト実行ガイド
2. **CODE_QUALITY_IMPROVEMENT_PLAN.md** - コード品質改善計画
3. **QUICKSTART_1M_ENSEMBLE.md** - 1M学習ガイド
4. **CHECKPOINT_INTERVAL_EXTENSION.md** - チェックポイント詳細
5. **UNIFIED_TRAINER_INTEGRATION_SUMMARY.md** - 統合サマリー

---

## 🎯 期待成果

### Week 1終了時
- ✅ 100kテスト完了（3モデル）
- ✅ CustomPPO動作確認
- ✅ 最適パラメータ特定

### Week 2-3終了時
- ✅ 型安全性50%向上
- ✅ 設定ファイル統一完了
- ✅ インターフェース統一完了

### Week 4終了時
- ✅ 1M学習完了（3モデル）
- ✅ 儲かるモデル候補発見
- ✅ アンサンブル構成決定

---

**次のアクション**:

1. **今すぐ**: 100kテスト開始（モデルB推奨）
2. **並行作業**: 他のCopilotにコード品質改善を依頼
3. **1週間後**: 1M学習開始の判断

**成功の鍵**: 100kテストで問題を早期発見し、1M学習前に解決すること！
