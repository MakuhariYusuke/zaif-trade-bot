# ハイパーパラメータ最適化 - 実行準備完了レポート

## ✅ 準備完了

### 1. 環境セットアップ ✅
- scikit-optimize インストール完了
- 最適化フレームワーク修正完了（numpy型対応）
- モックテスト成功

### 2. 作成されたスクリプト

#### scripts/optimization/test_optimization_mock.py
- **目的**: フレームワークの動作確認
- **実行時間**: 約2秒
- **結果**: ✅ 成功（Random Search + Binary Search）

#### scripts/optimization/run_sac_optimization_lightweight.py ⭐推奨
- **目的**: 実際のSAC訓練でパラメータ最適化
- **設定**:
  - Random Search: 8試行
  - 各試行: 3000ステップ（約10-15分）
  - 総所要時間: **約1.5-2時間**
- **最適化パラメータ**:
  - learning_rate: 1e-4 ~ 1e-3
  - batch_size: [128, 256, 512]
  - gamma: 0.985 ~ 0.999
  - target_update_interval: 1 ~ 5

#### scripts/optimization/run_sac_optimization.py（フルバージョン）
- **目的**: より詳細な最適化（Random + Binary Search）
- **設定**:
  - Random Search: 15試行
  - Binary Search: 10反復
  - 各試行: 5000ステップ
  - 総所要時間: **約6-8時間**

---

## 🎯 実行方法

### オプション1: 軽量版（推奨、約2時間）
```bash
python scripts\optimization\run_sac_optimization_lightweight.py
```

**利点**:
- 短時間で結果が得られる
- GPU/CPU負荷が低い
- v396用の推奨パラメータを生成

**出力**:
- `ztb/optimization/results/sac_opt_*/optimization_result.json`
- `configs/sac_v396_optimized.json` ✨

### オプション2: フルバージョン（約6-8時間）
```bash
python scripts\optimization\run_sac_optimization.py
```

**利点**:
- より精緻な最適化
- Binary Searchで微調整
- 長期訓練データ

---

## 📊 期待される結果

### v395iベースライン
- Critic Loss: **0.0918**
- Actor Loss: -5.61
- ent_coef: 0.279-0.917

### 最適化目標
- Critic Loss: **< 0.09**（目標値）
- 改善率: 5-15%

### モックテスト結果（参考）
```
Random Search:
  ベストCritic Loss: 0.172473
  learning_rate: 2.65e-04
  batch_size: 256
  gamma: 0.987
  target_update_interval: 3

Binary Search:
  最適Critic Loss: 0.146409
  最適learning_rate: 3.14e-04
```

---

## 🔧 技術詳細

### 修正したフレームワーク

#### ztb/optimization/base.py
**修正内容**:
- numpy.float64 → Python float 自動変換
- 目的関数の戻り値: float → TrialResult 自動変換
- エラーハンドリング改善

**修正箇所**:
```python
# numpy型を標準Python型に変換
clean_params = {}
for key, value in parameters.items():
    if hasattr(value, 'item'):  # numpy型
        clean_params[key] = value.item()
    else:
        clean_params[key] = value

# floatが返された場合、TrialResultを作成
if isinstance(objective_value, (int, float)):
    result = TrialResult(
        trial_id=trial_id,
        parameters=clean_params,
        objective_value=float(objective_value),
        ...
    )
```

### 最適化アルゴリズム

#### Random Search
- **戦略**: 全パラメータ空間をランダムサンプリング
- **利点**: 高次元でも効率的、局所最適に陥らない
- **適用**: 初期探索、広範囲サーチ

#### Binary Search (Golden Section)
- **戦略**: 黄金分割探索（φ = 1.618...）
- **利点**: O(log n)の効率、単一パラメータで最高精度
- **適用**: learning_rateの微調整

---

## 📋 実行チェックリスト

### 実行前
- [x] scikit-optimize インストール
- [x] 最適化フレームワーク動作確認
- [x] ベース設定ファイル確認（sac_v395i_complete_fix.json）
- [ ] GPU/CPU使用率確認
- [ ] ディスク空き容量確認（結果保存用）

### 実行中
- [ ] プログレスモニタリング
- [ ] Critic Loss推移確認
- [ ] エラー発生時の対応

### 実行後
- [ ] 最適化結果確認
- [ ] v396設定ファイル検証
- [ ] バックアップ作成

---

## 🚀 次のステップ

### Step 1: 最適化実行（このステップ）
```bash
python scripts\optimization\run_sac_optimization_lightweight.py
```

### Step 2: v396訓練
最適化で見つかったパラメータで長期訓練：
```bash
python -m ztb.training.train --config configs\sac_v396_optimized.json
```

### Step 3: 評価・比較
- TensorBoard分析
- 統計的検定（v395i vs v396）
- バックテスト評価

---

## ⚠️ 注意事項

1. **実行時間**: 軽量版でも約2時間かかります
2. **リソース**: GPU使用率が高くなります
3. **中断**: Ctrl+Cで安全に中断可能
4. **再開**: 現在は再開機能なし（将来実装予定）
5. **結果保存**: 各試行の結果は自動保存されます

---

## 💡 トラブルシューティング

### エラー: "ベース設定ファイルが見つかりません"
```bash
# 利用可能な設定ファイルを確認
dir configs\sac*.json
```

### エラー: "タイムアウト（40分超過）"
- 各試行の timesteps を減らす（3000 → 2000）
- または timeout を増やす

### エラー: "Critic Lossを抽出できませんでした"
- 訓練スクリプトの出力形式を確認
- extract_critic_loss() 関数を調整

---

## 📁 ファイル構成

```
scripts/optimization/
├── test_optimization_mock.py          # モックテスト
├── run_sac_optimization_lightweight.py # 軽量版（推奨）
└── run_sac_optimization.py            # フルバージョン

ztb/optimization/
├── base.py                            # 修正済み（numpy対応）
├── methods/
│   ├── random_search.py
│   ├── binary_search.py
│   ├── bayesian_optimization.py
│   └── grid_search.py
└── results/                           # 結果出力先
    └── sac_opt_*/                     # タイムスタンプ付き
        ├── trial_*_config.json
        ├── trial_*_result.json
        └── optimization_result.json
```

---

## ✅ 準備完了！

すべての準備が整いました。以下のコマンドで最適化を開始できます：

```bash
python scripts\optimization\run_sac_optimization_lightweight.py
```

**推定所要時間**: 約1.5-2時間
**期待される改善**: Critic Loss 5-15%向上

🚀 **最適化を開始しますか？**
