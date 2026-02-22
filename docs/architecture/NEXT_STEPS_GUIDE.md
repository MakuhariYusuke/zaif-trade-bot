# ディレクトリ構造移行 - 次のステップガイド

## 📊 分析結果サマリー

### 現在の問題点
- **ルート直下に73個のファイル**が散在
- そのうち実在する移行対象ファイル: **20個**
  - configs関連: 2個 (`.eslintrc.json`, `.markdownlint.json`)
  - docs関連: 6個 (MDファイル)
  - scripts関連: 12個 (分析・デバッグスクリプト)

### 分析ツール完成
- ✅ `scripts/maintenance/analyze_structure.py` - 構造分析
- ✅ `scripts/maintenance/migrate_structure.py` - 段階的移行

---

## 🎯 選択肢1: 構造移行を優先

### メリット
- 保守性の即時改善
- 今後の開発効率向上
- ファイル発見性の向上

### 実行手順

#### Step 1: 最小限の移行（推奨）
```bash
# 実在する20ファイルのみ移行
python scripts\maintenance\migrate_structure.py --execute
```

**影響範囲**: 極小（ルート直下の散在ファイルのみ）
**所要時間**: 5分
**リスク**: 低（DRY-RUN確認済み）

#### Step 2: ztb/の深化（オプション）
現在の`ztb/optimization/`をより深い構造に拡張：
```
ztb/optimization/
├── methods/
│   ├── evolutionary/
│   ├── bayesian/
│   ├── grid/
│   └── random/
├── objectives/
│   ├── sac/
│   └── ppo/
├── strategies/
│   ├── staged/
│   └── adaptive/
└── results/
```

**所要時間**: 1-2時間
**効果**: 最適化機能の拡張性向上

#### Step 3: 全体移行（長期計画）
`docs/architecture/DIRECTORY_STRUCTURE_PROPOSAL.md`の完全実施：
- ztb/core/への再編成
- ztb/training/の詳細分割
- ztb/analysis/の新設

**所要時間**: 10-15時間
**効果**: プロジェクト全体の保守性向上

---

## 🔬 選択肢2: 最適化実行を優先

### メリット
- SAC v395iの性能向上の即時確認
- 最適化フレームワークの実戦検証
- 実データによる改善

### 実行手順

#### Step 1: scikit-optimizeインストール
```bash
pip install scikit-optimize
```

#### Step 2: essential parametersの最適化
```bash
python -c "from ztb.optimization.examples import example_1_random_search; example_1_random_search()"
```

**パラメータ**: `learning_rate`, `batch_size`
**試行回数**: 20回
**所要時間**: 各試行30分 = 約10時間

#### Step 3: Binary Searchで微調整
```bash
python -c "from ztb.optimization.examples import example_3_binary_search; example_3_binary_search()"
```

**パラメータ**: `learning_rate` (前段の最適値付近)
**所要時間**: 約3時間

#### Step 4: 最適パラメータでv396訓練
見つかった最適値で5k→10k訓練を実施

---

## 🎨 選択肢3: 段階的アプローチ（推奨）

### Phase 1: クイックウィン（今日）
1. **最小限の構造移行** (5分)
   ```bash
   python scripts\maintenance\migrate_structure.py --execute
   ```
2. **ztb/optimization/の動作確認** (10分)
   ```bash
   python -c "from ztb.optimization.examples import example_1_random_search; example_1_random_search()"
   ```

### Phase 2: 最適化実験（今週）
1. **Random Searchでパラメータ探索** (10時間)
2. **Binary Searchで微調整** (3時間)
3. **最適値でv396訓練** (5時間)

### Phase 3: 構造深化（来週以降）
1. **ztb/optimization/の深化** (1-2時間)
2. **結果に基づき全体移行計画の調整**

---

## 📋 推奨される次のアクション

### オプションA: すぐに効果を見たい
```bash
# 1. 構造整理（5分）
python scripts\maintenance\migrate_structure.py --execute

# 2. 最適化テスト（5分）
python -c "from ztb.optimization.examples import example_1_random_search; example_1_random_search()"
```

### オプションB: 慎重に進めたい
```bash
# 1. scikit-optimizeインストール確認
pip list | findstr scikit-optimize

# 2. Bayesian Optimizationのテスト
python -c "from ztb.optimization.examples import example_4_bayesian_optimization; example_4_bayesian_optimization()"
```

### オプションC: 本格的な改善に着手
1. 構造移行の承認
2. ztb/optimization/の深化設計レビュー
3. 段階的実施計画の策定

---

## 💭 判断のポイント

| 優先事項 | 推奨選択 |
|---------|---------|
| すぐに成果を出したい | 選択肢2（最適化実行） |
| 長期的な保守性重視 | 選択肢1（構造移行） |
| バランス重視 | 選択肢3（段階的） |
| リスク最小化 | オプションA（クイックウィン） |

---

## ❓ ご質問

**どの方向で進めましょうか？**

1. **「構造移行を実行」** → `python scripts\maintenance\migrate_structure.py --execute`を実行します
2. **「最適化を試す」** → scikit-optimizeインストールと例実行を行います
3. **「両方少しずつ」** → 段階的アプローチ（Phase 1のクイックウィン）を実行します
4. **「もっと詳細を見る」** → 特定の部分の詳細分析を行います

お考えをお聞かせください！
