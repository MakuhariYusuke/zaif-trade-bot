# バイナリサーチ実行ガイド (初めての方向け)

## 🎯 このガイドの目的

パラメータ最適化のバイナリサーチを**初めて実行する方**向けの、ステップバイステップガイドです。

---

## ✅ 事前チェックリスト

実行前に以下を確認してください:

- [ ] Pythonがインストールされている (Python 3.11推奨)
- [ ] プロジェクトルートにいる (`C:\Users\Admin\dev\zaif-trade-bot`)
- [ ] `.venv` フォルダが存在する
- [ ] 依存パッケージがインストールされている

---

## 🚀 実行手順 (Windows PowerShell)

### ステップ 1: ターミナルを開く

1. VS Code を開く
2. `Ctrl + @` でターミナルを開く (またはメニュー → ターミナル → 新しいターミナル)
3. PowerShellが選択されていることを確認

### ステップ 2: プロジェクトルートに移動

```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
```

### ステップ 3: 仮想環境をアクティベート ⚠️ 重要!

```powershell
.\.venv\Scripts\Activate.ps1
```

**成功すると**:
```powershell
(.venv) PS C:\Users\Admin\dev\zaif-trade-bot>
```
プロンプトの先頭に `(.venv)` が表示されます。

**エラーが出る場合** (実行ポリシー):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 再度アクティベート
.\.venv\Scripts\Activate.ps1
```

### ステップ 4: 環境確認

```powershell
# Python実行環境の確認 (仮想環境のPythonが使われているか)
python -c "import sys; print(sys.executable)"
# 期待される出力: C:\Users\Admin\dev\zaif-trade-bot\.venv\Scripts\python.exe

# 必須パッケージの確認
python -c "import numpy; print('numpy OK:', numpy.__version__)"
python -c "import stable_baselines3; print('stable-baselines3 OK')"
```

**全て成功すれば準備完了!** ✅

### ステップ 5: バイナリサーチ実行

#### 例1: batch_size の粗選別 (20k, 推奨時間: 2-3時間)

```powershell
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000
```

#### 例2: learning_rate の粗選別 (10k, 推奨時間: 1-2時間)

```powershell
python -m ztb.training.binary_search.learning_rate_optimized --mode binary --max_iterations 4 --timesteps 10000
```

#### 例3: max_grad_norm の検証 (30k, 推奨時間: 3-4時間)

```powershell
python -m ztb.training.binary_search.max_grad_norm_optimized --mode binary --max_iterations 3 --timesteps 30000 --seed 42
```

### ステップ 6: 実行中のモニタリング

別のターミナルで TensorBoard を起動:

```powershell
# 新しいターミナルを開く (Ctrl + Shift + @)
# 仮想環境をアクティベート
.\.venv\Scripts\Activate.ps1

# TensorBoard起動
tensorboard --logdir tensorboard/binary_search/
```

ブラウザで http://localhost:6006 を開く

### ステップ 7: 完了確認と次ステップの判断

#### 7-1: 結果ファイルの生成確認

実行が終わると以下が生成されます:

```
binary_search_results/
├── batch_size_history.json       # スコア履歴 ← これを確認!
├── gamma_history.json
└── ...

models/binary_search/
├── iter01_batch_size_16.zip      # 各候補のモデル
├── iter02_batch_size_32.zip
└── ...

tensorboard/binary_search/
└── batch_size/                   # 学習ログ
```

#### 7-2: Phase 1 (粗選別) 結果の分析

**batch_size を例に説明**:

1. **履歴ファイルを開く**:
```powershell
notepad binary_search_results\batch_size_history.json
```

2. **最終iterationの improvement 値を確認**:
```json
{
  "iteration_3": {
    "candidate_value": 32,
    "score": -185.5,
    "baseline_score": -194.2,
    "improvement": 8.7  // ← この値をチェック!
  }
}
```

3. **Phase 2 (50k精査) の範囲を決定**:

| improvement | 判断 | Phase 2 の設定 |
|-------------|------|----------------|
| **≥10** | 高信頼度 ✅ | 最適値のみ再検証 (`--search_range 32`) |
| **5-10** | 中信頼度 ⚠️ | 最適値±1段階 (`--search_range 16,32,64`) |
| **<5** | 低信頼度 ❌ | より広範囲 (`--search_range 8,16,32,64,128`) |

#### 7-3: Phase 2 (50k精査) の実行

**例: improvement=8.7 (中信頼度) の場合**:

```powershell
python -m ztb.training.binary_search.batch_size_optimized `
  --mode binary `
  --max_iterations 2 `
  --timesteps 50000 `
  --search_range 16,32,64
```

**期待時間**: 5-8時間

**💡 ヒント**:
- TensorBoard で複数候補のスコア推移を比較
- 収束の安定性も重要な判断材料
- 時間に余裕があれば、広めの範囲で検証を推奨

---

## 🔧 トラブルシューティング

### ❌ エラー: ModuleNotFoundError: No module named 'numpy'

**原因1**: 仮想環境がアクティベートされていない
**原因2**: 仮想環境が不完全 (依存関係未インストール)

**解決策**:

**まず診断**:
```powershell
# アクティベーションファイルの存在確認
Test-Path .\.venv\Scripts\Activate.ps1
Test-Path .\.venv\Scripts\activate.bat

# .venv\Scripts\の中身を確認
dir .\.venv\Scripts\
```

**診断結果別の対処**:

1. **Activate.ps1やactivate.batが存在しない**:
```powershell
# 仮想環境を再作成
Remove-Item -Recurse -Force .venv
python -m venv .venv

# Command Promptでアクティベート
cmd
.venv\Scripts\activate.bat

# 依存関係インストール
pip install -r requirements.txt
```

2. **Activate.ps1が存在するが、アクティベートできない**:
```powershell
# Command Prompt推奨
cmd
.venv\Scripts\activate.bat

# またはPowerShellでCall Operator使用
& .\.venv\Scripts\Activate.ps1
```

3. **アクティベートはできているが、numpyがない**:
```powershell
# プロンプトに (.venv) が表示されていることを確認
# 依存関係を再インストール
pip install -r requirements.txt
```

### ❌ エラー: パッケージが見つからない

**症状**: `ModuleNotFoundError: No module named 'XXX'` (numpy, sb3_contrib等)

**原因**: 依存関係がインストールされていない、または不完全

**解決策**:
```cmd
REM 仮想環境をアクティベート後 ((.venv) 表示を確認)
pip install -r requirements.txt

REM 個別パッケージのインストール (必要に応じて)
pip install sb3-contrib
pip install numpy
pip install stable-baselines3

REM インストール確認
python -c "import numpy, stable_baselines3, sb3_contrib; print('All packages OK')"
```

**💡 よくあるケース**:
- `sb3_contrib` (sb3-contrib) が特によく不足します
- `requirements.txt`の一括インストールが最も確実です

### ❌ エラー: ImportError: DLL load failed

**原因**: PyTorch/CUDA関連の問題

**解決策**:
```powershell
# CPU版PyTorchを再インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❌ エラー: The file cannot be loaded because running scripts is disabled

**原因**: PowerShellの実行ポリシー制限

**解決策**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ⚠️ 警告: KL早期停止が多発

**症状**: ログに "KL divergence exceeded" が頻繁に表示

**原因**: KL制約が厳しすぎる

**解決策**:
```powershell
# target_klを緩和して再実行
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000 --target_kl 0.015
```

### 💡 実行が遅い

**対処法**:

1. **timestepsを減らす** (粗選別時):
   ```powershell
   --timesteps 10000  # 20000 → 10000
   ```

2. **並列実行** (複数パラメータを同時に):
   ```powershell
   # 別々のターミナルで実行
   # Terminal 1
   python -m ztb.training.binary_search.batch_size_optimized ... &

   # Terminal 2
   python -m ztb.training.binary_search.learning_rate_optimized ... &
   ```

3. **夜間実行**:
   - 長時間検証 (50k) は夜間にスケジュール

---

## 📊 結果の確認方法

### 1. コンソール出力

実行中、以下のような出力が表示されます:

```
[INFO] Binary search iteration 1/3
[INFO] Testing batch_size = 64
[INFO] Training for 20000 timesteps...
[INFO] Episode 10: reward = -285.32
...
[INFO] Best batch_size: 32 (score: -268.45)
```

### 2. 履歴ファイル

```powershell
# JSONファイルを確認
Get-Content binary_search_results\batch_size_history.json | ConvertFrom-Json | Format-Table
```

### 3. TensorBoard

- ブラウザで http://localhost:6006
- 報酬曲線、KL divergence、エントロピーをグラフで確認

---

## 📝 結果の記録

検証完了後、以下を更新:

### 1. 追跡ドキュメント更新

`PARAMETER_VALIDATION_TRACKING.md` を開いて:

```markdown
| # | パラメータ | 現在値 (短期) | ステータス | ... |
|---|-----------|-------------|-----------|-----|
| 1 | batch_size | 16 | ✅ 完了 | ... |  ← ❌ → ✅ に変更
```

### 2. 詳細結果の追記

```markdown
## batch_size 検証結果 (2025-10-XX)

最適値: 32
差分: +5.2 (vs デフォルト64)
95%CI: ±7.1
統計的有意: ❌ (差分 < CI幅)

次アクション:
- [ ] 50k再検証 (差分が小さいため)
```

---

## 🎯 よくある質問

### Q: 何時間かかりますか?

**A**: パラメータと timesteps による:

| 検証内容 | Timesteps | 推奨時間 |
|---------|----------|---------|
| 粗選別 | 10k-20k | 1-3時間 |
| 精査 | 50k | 5-8時間 |
| マルチシード | 50k×2 | 10-16時間 |

### Q: 途中で止めたい場合は?

**A**: `Ctrl + C` で停止可能。中断時点までの結果は保存されています。

### Q: 複数のパラメータを同時に検証できますか?

**A**: 可能です。別々のターミナルで実行してください。

### Q: GPUは必要ですか?

**A**: 不要です。CPU版PyTorchで動作します (ただしGPU利用で高速化可能)。

---

## 📚 関連ドキュメント

- **詳細追跡シート**: [PARAMETER_VALIDATION_TRACKING.md](./PARAMETER_VALIDATION_TRACKING.md)
- **クイックリファレンス**: [PARAMETER_VALIDATION_QUICK_REFERENCE.md](./PARAMETER_VALIDATION_QUICK_REFERENCE.md)
- **包括的結果**: [BINARY_SEARCH_COMPREHENSIVE_RESULTS.md](./BINARY_SEARCH_COMPREHENSIVE_RESULTS.md)

---

## ✅ チェックリスト (印刷して使用可)

検証実行時のチェックリスト:

```
準備
□ プロジェクトルートに移動
□ 仮想環境アクティベート (.venv 表示確認)
□ 環境確認コマンド実行 (numpy等)

実行
□ パラメータ名を確認
□ コマンドをコピー&実行
□ エラーがないか確認

モニタリング
□ TensorBoard起動 (別ターミナル)
□ 学習曲線を定期確認
□ KL早期停止の頻度確認

完了後
□ 結果ファイル生成確認
□ 追跡ドキュメント更新
□ 次のパラメータを決定
```

---

**このガイドに従えば、初めての方でも安全にバイナリサーチを実行できます!** 🎉

**問題が発生した場合**: トラブルシューティングセクションを確認してください。
