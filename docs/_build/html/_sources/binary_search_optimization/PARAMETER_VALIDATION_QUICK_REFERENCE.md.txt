# パラメータ検証クイックリファレンス

## 🚀 クイックスタート

### ⚠️ 事前準備: 仮想環境のアクティベート

**Windows (PowerShell) - 推奨: Command Promptの方が簡単**:

方法1: Call Operator使用 (最も確実)
```powershell
# プロジェクトルート確認
cd C:\Users\Admin\dev\zaif-trade-bot

# & (コールオペレータ) でアクティベート
& .\.venv\Scripts\Activate.ps1
```

方法2: フルパス指定
```powershell
C:\Users\Admin\dev\zaif-trade-bot\.venv\Scripts\Activate.ps1
```

方法3: 実行ポリシー変更後
```powershell
# 初回のみ実行ポリシー変更
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# アクティベート
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt) - 最も簡単 ✅ 推奨**:
```cmd
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat
```

**Linux/Mac**:
```bash
source .venv/bin/activate
```

**確認**:
```bash
# プロンプトに (.venv) が表示されているか確認
# (.venv) PS C:\Users\Admin\dev\zaif-trade-bot>

# Python実行環境の確認
python -c "import numpy; print('numpy version:', numpy.__version__)"
# 出力例: numpy version: 1.26.4
```

### 次に検証すべきパラメータ TOP 3

1. **batch_size** (優先度: 🔴 High)
   ```bash
   # Phase 1: 20k粗選別
   python -m ztb.training.binary_search.batch_size_optimized \
     --mode binary --max_iterations 3 --timesteps 20000
   
   # 📊 結果を確認して Phase 2 の範囲を決定
   # 結果ファイル: binary_search_results/batch_size_history.json
   # チェック項目: 最終iterationのimprovement値を確認
   #   - improvement ≥10 → 高信頼度: 最適値のみ再検証
   #   - improvement 5-10 → 中信頼度: 最適値±1段階を再検証
   #   - improvement <5 → 低信頼度: より広い範囲で再検証
   
   # Phase 2: 50k精査 (例: 中信頼度の場合)
   python -m ztb.training.binary_search.batch_size_optimized \
     --mode binary --max_iterations 2 --timesteps 50000 \
     --search_range 16,32,64
   ```
   
   **📖 詳細な判断基準は下記「Phase 1→Phase 2 移行ガイド」参照**

2. **learning_rate** (優先度: 🔴 High)
   ```bash
   # Phase 1: 10k粗選別
   python -m ztb.training.binary_search.learning_rate_optimized \
     --mode binary --max_iterations 4 --timesteps 10000
   
   # Phase 2: 50k精査
   python -m ztb.training.binary_search.learning_rate_optimized \
     --mode binary --max_iterations 2 --timesteps 50000
   ```

3. **max_grad_norm** (優先度: 🔴 High)
   ```bash
   # 30k × 2シード
   # Seed 1
   python -m ztb.training.binary_search.max_grad_norm_optimized \
     --mode binary --max_iterations 3 --timesteps 30000 --seed 42
   
   # Seed 2
   python -m ztb.training.binary_search.max_grad_norm_optimized \
     --mode binary --max_iterations 3 --timesteps 30000 --seed 123
   ```

---

## 📊 ステータス一覧 (2025-10-10時点)

### ✅ 完了 (本番適用可能)

| パラメータ | 最適値 | 検証ステップ | 信頼度 |
|-----------|--------|------------|--------|
| n_steps | 1024 | 50k | 高 (差分+11) |
| vf_coef | 0.1 | 50k | 高 (差分+1) |
| reward_multipliers | 5.0 | 2k短期 | 高 (差分+152) ⚠️長期要確認 |
| risk_free_rate | 0.0 | 50k | 完全収束 |

### ⚠️ 要再検証 (差分小、追加検証推奨)

| パラメータ | 暫定値 | 差分 | 次アクション |
|-----------|--------|------|-------------|
| gamma | 0.8475 | 1~2 | 50k×2シード |
| n_epochs | 16 | <1 | 50k×2シード |

### ❌ 未検証 (長期検証未実施)

**高優先度**:
- batch_size (短期: 16)
- learning_rate (短期: 1e-5)
- max_grad_norm (デフォルト: 0.5)

**中優先度**:
- gae_lambda (短期: 0.8)
- clip_range (短期: 0.1)
- target_kl (短期: 0.001)
- ent_coef (短期: 0.001)

**低優先度**:
- normalize_advantage (短期: True)
- reward_scaling (短期: 0.1)
- transaction_cost (短期: 0.0001)

---

## 🎯 検証判定基準

### 統計的有意性

**50k検証の場合** (n=51エピソード):
- 標準誤差 SE ≈ 3.62
- 95%信頼区間 ≈ ±7.1

| 差分 | 判定 | アクション |
|------|------|-----------|
| ≥10 | ✅ 統計的有意 | 本番適用OK |
| 5-9 | ⚠️ 有意傾向 | 追加検証推奨 |
| 3-4 | ❓ 微差 | 100k or マルチシード |
| <3 | ❌ ノイズレベル | デフォルト継続検討 |

### 成功基準チェックリスト

検証完了時に確認:
- [ ] 差分が95%CI幅を超えている
- [ ] KL早期停止が過度でない (<50%)
- [ ] 報酬分散が許容範囲 (σ<30)
- [ ] エントロピーが適切に減衰
- [ ] 学習曲線が単調改善

---

## 📅 推奨検証順序

### Week 1-2: 高優先度
```
Day 1-2:  batch_size (20k粗選別)
Day 3:    learning_rate (10k粗選別)
Day 4-6:  max_grad_norm (30k×seed1)
Day 7-11: batch_size (50k精査)
Day 7-11: learning_rate (50k精査) [並行]
Day 12-14: max_grad_norm (30k×seed2)
```

### Week 3-4: 中優先度
```
並行実行推奨:
- gamma (50k×2シード)
- n_epochs (50k×2シード)
- gae_lambda (50k×1)
- clip_range (50k×1)
- target_kl (50k×1)
- ent_coef (50k×1)
```

---

## � Phase 1→Phase 2 移行ガイド

### batch_size の例

**ステップ1: Phase 1 (20k粗選別) 結果の確認**

**📁 結果ファイルの種類**:
- `batch_size_binary_search.jsonl` - **イベントログ (JSONL形式)** ← 最新版
- `batch_size_history.json` - 履歴サマリー (古い形式、存在しない場合あり)

**JSONLファイルから最終結果を確認 (推奨)**:

```bash
# Windows PowerShell - 最終行を表示
Get-Content binary_search_results\batch_size_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json | Format-List

# または notepad で直接確認
notepad binary_search_results\batch_size_binary_search.jsonl
# 最終行の "event": "complete" を見る

# Linux/Mac
tail -n 1 binary_search_results/batch_size_binary_search.jsonl | jq .
```

**ステップ2: improvement 値を計算**

JSONLの最終行から:
```json
{"event": "complete", "parameter_value": 256, "score": -298.24}
```

最初の評価から:
```json
{"event": "evaluation", "stage": "lower", "parameter_value": 16, "score": -303.02}
```

**improvement計算**:
```
improvement = 初期スコア - 最適スコア
            = (-303.02) - (-298.24)
            = 4.78  // ← この値で判断!
```

**ステップ3: Phase 2 範囲の決定**

| improvement | 信頼度 | Phase 2 範囲の設定 | コマンド例 |
|-------------|--------|-------------------|-----------|
| **≥10** | 🟢 高 | 最適値のみ再検証 | `--search_range 32` |
| **5-10** | 🟡 中 | 最適値±1段階 | `--search_range 16,32,64` |
| **<5** | 🔴 低 | より広範囲 | `--search_range 8,16,32,64,128` |

**ステップ4: Phase 2 (50k精査) 実行**

```bash
# 例: improvement=8.7 (中信頼度) の場合
python -m ztb.training.binary_search.batch_size_optimized \
  --mode binary \
  --max_iterations 2 \
  --timesteps 50000 \
  --search_range 16,32,64
```

**💡 ヒント**: 
- TensorBoard (`tensorboard --logdir tensorboard/binary_search/`) で複数候補のスコア推移を比較
- 収束の安定性も判断材料に含める
- 時間がある場合、低信頼度でも広範囲検証を推奨

---

## �💡 よくある質問

### Q1: 差分が小さい場合どうする?
**A**: 以下の優先順位で対処:
1. 100k×1走で信頼区間を半減 (推奨)
2. 50k×2シードで標準誤差改善
3. デフォルト値継続を検討

### Q2: KL早期停止が多発する
**A**: 以下を確認:
1. `target_kl`を0.01→0.015に緩和
2. `clip_range`を調整
3. `learning_rate`を下げる

### Q3: スコアが横並びになる
**A**: 原因別対処:
1. timesteps不足 → 50k→100k拡張
2. パラメータレンジ狭い → 範囲拡大
3. 効果が本当に小さい → デフォルト継続

### Q4: 並列実行できる?
**A**: 可能です:
```bash
# 異なるパラメータを別GPU/マシンで
CUDA_VISIBLE_DEVICES=0 python -m ztb.training.binary_search.batch_size_optimized ... &
CUDA_VISIBLE_DEVICES=1 python -m ztb.training.binary_search.learning_rate_optimized ... &
```

### Q5: ModuleNotFoundError: No module named 'numpy'
**A**: 仮想環境がアクティベートされていません:

**解決策1: Command Promptを使用 (最も簡単) ✅**
```cmd
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat
python -c "import numpy; print('OK')"
```

**解決策2: PowerShell - Call Operator使用**
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
& .\.venv\Scripts\Activate.ps1
# または
& C:\Users\Admin\dev\zaif-trade-bot\.venv\Scripts\Activate.ps1
```

**解決策3: PowerShell - 実行ポリシー変更**
```powershell
# 初回のみ実行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 再度アクティベート
.\.venv\Scripts\Activate.ps1
```

**確認コマンド**:
```bash
# プロンプトに (.venv) が表示されるか確認
# (.venv) PS C:\Users\Admin\dev\zaif-trade-bot>

python -c "import numpy; print('OK')"
```

**それでもエラーが出る場合**:
```bash
# 依存関係の再インストール
pip install -r requirements.txt
```

### Q5.1: ModuleNotFoundError: No module named 'sb3_contrib' 🆕
**A**: `sb3-contrib` (Stable-Baselines3の拡張ライブラリ) がインストールされていません:

**解決策**:
```cmd
REM 仮想環境がアクティベートされている状態で (.venv) 表示確認
pip install sb3-contrib

REM または、全依存関係を再インストール (推奨)
pip install -r requirements.txt

REM インストール確認
python -c "import sb3_contrib; print('sb3-contrib OK')"
```

**💡 ヒント**: `requirements.txt`で一括管理されているので、`pip install -r requirements.txt`が最も確実です。

### Q5.2: PowerShellで '.\.venv\Scripts\Activate.ps1' が認識されない 🆕
**A**: 以下の診断手順で原因を特定してください:

**🔍 診断ステップ1: カレントディレクトリ確認**
```powershell
pwd
# 出力: C:\Users\Admin\dev\zaif-trade-bot であることを確認
```

**🔍 診断ステップ2: アクティベーションファイルの存在確認**
```powershell
# Activate.ps1の存在確認
Test-Path .\.venv\Scripts\Activate.ps1

# .venv\Scripts\の中身を確認
dir .\.venv\Scripts\
```

**判定**:
- `python.exe`**のみ**存在 → **仮想環境が不完全** (解決策A)
- `Activate.ps1`も存在 → PowerShellのパス問題 (解決策B)

---

**解決策A: 仮想環境が不完全な場合 (Activate.ps1が存在しない)**

```powershell
# ステップ1: 古い .venv を削除
Remove-Item -Recurse -Force .venv

# ステップ2: 新しい仮想環境を作成
python -m venv .venv

# ステップ3: 作成確認 (Activate.ps1等が生成されているはず)
dir .\.venv\Scripts\

# ステップ4: Command Promptでアクティベート (推奨)
cmd
.venv\Scripts\activate.bat

# ステップ5: 依存関係のインストール
pip install -r requirements.txt

# ステップ6: 環境確認
python -c "import numpy, stable_baselines3; print('Environment OK')"
```

---

**解決策B: Activate.ps1は存在するがPowerShellで認識されない場合**

**B-1: Call Operator (&) で実行** (最も確実):
```powershell
& .\.venv\Scripts\Activate.ps1
```

**B-2: フルパス指定**:
```powershell
C:\Users\Admin\dev\zaif-trade-bot\.venv\Scripts\Activate.ps1
```

**B-3: Command Promptを使う** (最も簡単):
```cmd
cmd
.venv\Scripts\activate.bat
```

# 新規作成
python -m venv .venv

# アクティベート
& .\.venv\Scripts\Activate.ps1

# 依存関係インストール
pip install -r requirements.txt
```

### Q6: ImportError: DLL load failed
**A**: PyTorch/CUDA関連の問題:
```bash
# CPU版PyTorchの確認
pip list | findstr torch

# 必要に応じて再インストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 📝 検証完了時の記録

### 最小限の記録項目

```markdown
## [パラメータ名] 検証完了 (YYYY-MM-DD)

最適値: [値]
差分: +[値] (vs デフォルト)
95%CI: ±[値]
統計的有意: ✅/❌

次アクション:
- [x] 設定ファイル更新
- [x] PARAMETER_VALIDATION_TRACKING.md 更新
- [ ] 追加検証 (必要時)
```

### 更新すべきファイル

1. `PARAMETER_VALIDATION_TRACKING.md`
   - ステータスを ❌→✅ に更新
   - 結果セクションに詳細記録

2. `configs/training/ppo_binary_search_validated.json`
   - 最適値を反映
   - コメントに検証日追記

3. `docs/BINARY_SEARCH_COMPREHENSIVE_RESULTS.md`
   - 該当パラメータセクション更新

---

## 🔗 関連リンク

- **🆕 Windows実行コマンド集**: [BINARY_SEARCH_WINDOWS_COMMANDS.md](./BINARY_SEARCH_WINDOWS_COMMANDS.md) - コピペ用コマンド集
- **🆕 実行ガイド**: [BINARY_SEARCH_EXECUTION_GUIDE.md](./BINARY_SEARCH_EXECUTION_GUIDE.md) - 初心者向けステップバイステップ
- **詳細追跡**: [PARAMETER_VALIDATION_TRACKING.md](./PARAMETER_VALIDATION_TRACKING.md)
- **包括的結果**: [BINARY_SEARCH_COMPREHENSIVE_RESULTS.md](./BINARY_SEARCH_COMPREHENSIVE_RESULTS.md)
- **設定ファイル**: [ppo_binary_search_validated.json](../configs/training/ppo_binary_search_validated.json)

---

**このカードは定期的に更新してください (推奨: 週次)**

**最終更新**: 2025-10-10
