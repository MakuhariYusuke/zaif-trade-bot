# バイナリサーチ実行ガイド - Windows版

## 🚀 最速スタート (コピペ用)

### ⚠️ 重要: PowerShellよりCommand Promptが簡単です

**Command Prompt (推奨) ✅**
```cmd
REM 1. プロジェクトルートに移動
cd C:\Users\Admin\dev\zaif-trade-bot

REM 2. 仮想環境アクティベート (PowerShellより簡単!)
.venv\Scripts\activate.bat

REM 3. 環境確認
python -c "import numpy, stable_baselines3; print('Environment OK')"

REM 4. batch_size 粗選別 (20k, 2-3時間)
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000
```

### PowerShellで一括実行

方法1: Call Operator使用 (推奨)
```powershell
# 1. プロジェクトルートに移動
cd C:\Users\Admin\dev\zaif-trade-bot

# 2. 仮想環境アクティベート
& .\.venv\Scripts\Activate.ps1

# 3. 環境確認
python -c "import numpy, stable_baselines3; print('Environment OK')"

# 4. batch_size 粗選別 (20k, 2-3時間)
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000
```

方法2: 実行ポリシー変更後
```powershell
# 初回のみ: 実行ポリシー変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 1. 仮想環境アクティベート
.\.venv\Scripts\Activate.ps1

# 2. 環境確認
python -c "import numpy, stable_baselines3; print('Environment OK')"

# 3. batch_size 粗選別 (20k, 2-3時間)
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000
```

---

## 📋 実行前チェック

```powershell
# プロジェクトルート確認
cd C:\Users\Admin\dev\zaif-trade-bot

# .venv フォルダ存在確認
dir .venv

# 仮想環境アクティベート
.\.venv\Scripts\Activate.ps1

# 環境テスト
python -c "import sys; print('Python:', sys.executable)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import stable_baselines3; print('SB3: OK')"
python -c "import gymnasium; print('gymnasium: OK')"
```

全て成功すれば準備完了 ✅

---

## 🎯 優先度順の実行コマンド

### Week 1: 高優先度パラメータ

#### Day 1-2: batch_size (20k粗選別)

```powershell
python -m ztb.training.binary_search.batch_size_optimized `
  --mode binary `
  --max_iterations 3 `
  --timesteps 20000
```

**期待時間**: 2-3時間  
**出力先**: `binary_search_results/batch_size_history.json`

**結果の見方 (50k精査の値を決める)**:

**📁 結果ファイルの種類**:
- `batch_size_binary_search.jsonl` - **イベントログ (JSONL形式)** ← 最新版
- `batch_size_history.json` - 履歴サマリー (古い形式)

**方法1: JSONLファイルから最終結果を確認 (推奨)**
```powershell
# 最終行 (event: complete) を表示
Get-Content binary_search_results\batch_size_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json | Format-List

# 出力例:
# event           : complete
# parameter_value : 256        ← 最適候補
# score           : -298.24    ← 最終スコア
```

**方法2: notepadで直接確認**
```powershell
notepad binary_search_results\batch_size_binary_search.jsonl
# 最終行の "event": "complete" を見る
# {"event": "complete", ..., "parameter_value": 256, "score": -298.24}
```

**チェックポイント**:
1. **最終行の`parameter_value`を確認** → これが20k時点の最適候補
2. **improvement (改善度) を計算**:
   ```
   improvement = 初期スコア - 最適スコア
   
   例: (-303.02) - (-298.24) = 4.78
   ```
3. **スコア差分で判断**:
   - 差分 ≥10: 高信頼度 ✅ → そのまま50k精査へ
   - 差分 5-10: 中信頼度 ⚠️ → 前後の値も含めて50k精査
   - 差分 <5: 低信頼度 ❌ → より広い範囲で50k精査が必要

**実例: improvement=4.78の場合**
```json
// 最初の評価 (lower)
{"event": "evaluation", "parameter_value": 16, "score": -303.02}

// 最終結果 (complete)
{"event": "complete", "parameter_value": 256, "score": -298.24}

// improvement = -303.02 - (-298.24) = 4.78 ← 低信頼度!
```
→ **improvement=4.78** (差分<5) なので、`--search_range 64,128,256`で50k精査を推奨

#### Day 3: learning_rate (10k粗選別)

```powershell
python -m ztb.training.binary_search.learning_rate_optimized `
  --mode binary `
  --max_iterations 4 `
  --timesteps 10000
```

**期待時間**: 1-2時間  
**出力先**: `binary_search_results/learning_rate_history.json`

#### Day 4-6: max_grad_norm (30k×seed1)

```powershell
python -m ztb.training.binary_search.max_grad_norm_optimized `
  --mode binary `
  --max_iterations 3 `
  --timesteps 30000 `
  --seed 42
```

**期待時間**: 3-4時間  
**出力先**: `binary_search_results/max_grad_norm_history.json`

**再現性確認 (seed2)**:
```powershell
python -m ztb.training.binary_search.max_grad_norm_optimized `
  --mode binary `
  --max_iterations 3 `
  --timesteps 30000 `
  --seed 123
```

### Week 2: 精査フェーズ

#### batch_size (50k精査) - 粗選別結果に基づく

**📊 20k結果から50k精査範囲を決定**

**ステップ1: 20k結果の分析**
```powershell
# 履歴ファイルを開く
Get-Content binary_search_results\batch_size_history.json | ConvertFrom-Json
```

**ステップ2: 改善度に応じた精査範囲の決定**

| 20k改善度 | 判断 | 50k精査範囲の設定例 |
|-----------|------|-------------------|
| **≥10** | 高信頼度 | 最適値のみ再検証 (例: `--search_range 32`) |
| **5-10** | 中信頼度 | 最適値±1段階 (例: `--search_range 16,32,64`) |
| **<5** | 低信頼度 | より広範囲 (例: `--search_range 8,16,32,64,128`) |

**ステップ3: 実際の50k精査コマンド**

```powershell
# 例1: 改善度=12 (高信頼度) の場合
python -m ztb.training.binary_search.batch_size_optimized `
  --mode binary `
  --max_iterations 1 `
  --timesteps 50000 `
  --search_range 32

# 例2: 改善度=7 (中信頼度) の場合
python -m ztb.training.binary_search.batch_size_optimized `
  --mode binary `
  --max_iterations 2 `
  --timesteps 50000 `
  --search_range 16,32,64

# 例3: 改善度=3 (低信頼度) の場合
python -m ztb.training.binary_search.batch_size_optimized `
  --mode binary `
  --max_iterations 3 `
  --timesteps 50000 `
  --search_range 8,16,32,64,128
```

**期待時間**: 
- 高信頼度 (1値のみ): 3-4時間
- 中信頼度 (3値): 5-8時間
- 低信頼度 (5値): 10-15時間

**💡 ヒント**: TensorBoardで複数候補のスコア推移を比較すると、収束の安定性も確認できます

#### learning_rate (50k精査)

```powershell
python -m ztb.training.binary_search.learning_rate_optimized `
  --mode binary `
  --max_iterations 2 `
  --timesteps 50000
```

**期待時間**: 5-8時間

#### max_grad_norm (30k×seed2 - 再現性確認)

```powershell
python -m ztb.training.binary_search.max_grad_norm_optimized `
  --mode binary `
  --max_iterations 3 `
  --timesteps 30000 `
  --seed 123
```

**期待時間**: 3-4時間

---

## 🖥️ TensorBoard モニタリング

### 別ターミナルで実行

```powershell
# 新しいターミナルを開く (Ctrl + Shift + @)

# 仮想環境アクティベート
.\.venv\Scripts\Activate.ps1

# TensorBoard起動
tensorboard --logdir tensorboard/binary_search/

# ブラウザで開く
start http://localhost:6006
```

---

## ⚡ 並列実行 (複数パラメータ同時)

### PowerShellで並列実行

```powershell
# Terminal 1: batch_size
.\.venv\Scripts\Activate.ps1
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 20000

# Terminal 2 (別ウィンドウ): learning_rate
.\.venv\Scripts\Activate.ps1
python -m ztb.training.binary_search.learning_rate_optimized --mode binary --max_iterations 4 --timesteps 10000

# Terminal 3 (別ウィンドウ): TensorBoard
.\.venv\Scripts\Activate.ps1
tensorboard --logdir tensorboard/binary_search/
```

---

## 🔧 トラブルシューティング (コピペ用)

### ❌ ModuleNotFoundError: No module named 'numpy'

**解決策1: Command Promptを使う (最も簡単) ✅**
```cmd
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat
python -c "import numpy; print('OK')"
```

**解決策2: PowerShell - Call Operator**
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
& .\.venv\Scripts\Activate.ps1
python -c "import numpy; print('OK')"
```

**解決策3: PowerShell - 実行ポリシー変更**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### ❌ ModuleNotFoundError: No module named 'sb3_contrib' 🆕

**原因**: `sb3-contrib` (SB3拡張ライブラリ) がインストールされていない

**解決策**:
```cmd
REM 仮想環境アクティベート済みの状態で
pip install sb3-contrib

REM または、全依存関係を再インストール (推奨)
pip install -r requirements.txt

REM インストール確認
python -c "import sb3_contrib; print('sb3-contrib OK')"
```

**💡 ヒント**: `requirements.txt`に記載されているはずなので、`pip install -r requirements.txt`が確実です。

### ❌ error: unrecognized arguments: --seed 42 🆕

**原因**: `--seed`引数がスクリプトに実装されていない (修正済み)

**解決策**: 最新版にアップデート後、以下で実行可能:
```cmd
python -m ztb.training.binary_search.max_grad_norm_optimized --mode binary --max_iterations 3 --timesteps 30000 --seed 42
```

**💡 ヒント**: `--seed`は再現性確認のためのオプション引数です。省略するとデフォルト(42)が使用されます。

### ❌ error: unrecognized arguments: --search_range 64,128,256 🆕

**原因**: `--search_range`引数がスクリプトに実装されていない (修正済み)

**解決策**: 最新版にアップデート後、以下で実行可能:
```cmd
REM カンマ区切りで具体的な値を指定
python -m ztb.training.binary_search.batch_size_optimized --mode binary --max_iterations 3 --timesteps 50000 --search_range 64,128,256

REM learning_rateでも使用可能
python -m ztb.training.binary_search.learning_rate_optimized --mode binary --max_iterations 2 --timesteps 50000 --search_range 1e-5,1e-4,0.001
```

**💡 ヒント**: 
- `--search_range`は特定の値のみを評価したい場合に使用
- Phase 1の結果に基づいてPhase 2の範囲を絞り込む際に便利
- 省略すると、デフォルトの範囲でバイナリサーチが実行されます

### ❌ PowerShellで '.\.venv\Scripts\Activate.ps1' が認識されない 🆕

**原因1**: PowerShellのパス認識問題  
**原因2**: 仮想環境が不完全 (Activate.ps1が存在しない)

**🔍 まず診断: アクティベーションファイルの存在確認**
```powershell
# .venv\Scripts\ の中身を確認
dir .\.venv\Scripts\

# Activate.ps1 が存在するか確認
Test-Path .\.venv\Scripts\Activate.ps1
# 期待される結果: True

# activate.bat が存在するか確認
Test-Path .\.venv\Scripts\activate.bat
# 期待される結果: True
```

**診断結果の判断**:
- `python.exe`**のみ**存在 → **仮想環境が不完全** → 解決策A (再作成)
- `Activate.ps1`や`activate.bat`も存在 → PowerShellのパス問題 → 解決策B (Call Operator等)

---

#### 🔧 解決策A: 仮想環境が不完全な場合 (Activate.ps1が存在しない)

**症状**: `dir .\.venv\Scripts\` で`python.exe`しか表示されない

```powershell
# ステップ1: 古い仮想環境を削除
Remove-Item -Recurse -Force .venv

# ステップ2: 新しい仮想環境を作成
python -m venv .venv

# ステップ3: 作成確認 (Activate.ps1とactivate.batが生成されているはず)
dir .\.venv\Scripts\

# ステップ4: Command Promptでアクティベート (推奨)
cmd
.venv\Scripts\activate.bat

# ステップ5: 依存関係のインストール
pip install -r requirements.txt

# ステップ6: 環境確認
python -c "import numpy, stable_baselines3; print('Environment OK')"
```

**⚠️ 注意**: `Remove-Item`実行前に重要なデータがないか確認してください

---

#### 🔧 解決策B: Activate.ps1は存在するがPowerShellで認識されない場合

**解決策B-1: Command Promptを使用 (最も簡単) ✅**
```cmd
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat
```

**解決策B-2: Call Operator (&) を使用**
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
& .\.venv\Scripts\Activate.ps1
```

**解決策B-3: フルパス指定**
```powershell
C:\Users\Admin\dev\zaif-trade-bot\.venv\Scripts\Activate.ps1
```

### ❌ 実行ポリシーエラー

```powershell
# 解決策: 実行ポリシー変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 再度アクティベート
.\.venv\Scripts\Activate.ps1
```

### ❌ ImportError: DLL load failed

```powershell
# 解決策: PyTorch再インストール
.\.venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❌ パッケージ不足

```powershell
# 解決策: 依存関係再インストール
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📊 結果確認コマンド

### 履歴ファイル確認

```powershell
# JSON を読みやすく表示
Get-Content binary_search_results\batch_size_history.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### 最新モデル確認

```powershell
# モデルファイル一覧
dir models\binary_search\ | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### ログファイル確認

```powershell
# 最新ログ表示 (Linux tail -f 相当)
Get-Content tensorboard\binary_search\batch_size\events.out.tfevents.* -Wait
```

---

## 📝 完了後の記録テンプレート

```markdown
## batch_size 検証完了 (2025-10-XX)

### 実行情報
- 日時: 2025-10-XX HH:MM
- Timesteps: 20,000
- Iterations: 3
- 所要時間: X時間

### 結果
- 最適値: XX
- スコア: -XXX.XX
- 差分: +/-XX.XX (vs デフォルト)
- 95%CI: ±7.1
- 統計的有意: ✅/❌

### 次アクション
- [ ] 50k精査 (必要時)
- [ ] 設定ファイル更新
- [ ] PARAMETER_VALIDATION_TRACKING.md 更新
```

---

## 🎯 実行チェックリスト

検証開始前に確認:

```
□ プロジェクトルート C:\Users\Admin\dev\zaif-trade-bot
□ 仮想環境アクティベート済み (プロンプトに .venv 表示)
□ numpy, stable_baselines3 インポート確認
□ TensorBoard起動 (別ターミナル)
□ 実行コマンド確認
□ 推定所要時間確認
```

---

**このシートをブックマークして、検証時にすぐアクセスできるようにしてください!** 🔖
