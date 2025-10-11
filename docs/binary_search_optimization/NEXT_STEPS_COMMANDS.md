# 次のステップ実行コマンド集

**作成日**: 2025年10月10日  
**対象**: max_grad_norm の再検証 (100k×2シード)

---

## 🎯 目的

max_grad_norm の30k検証で最適値が異なったため (7.525 vs 5.05)、より長期的な100k評価で再現性を確認します。

---

## 📋 実行前チェックリスト

- [ ] 仮想環境がアクティベート済み (`(.venv)` 表示確認)
- [ ] プロジェクトルート: `C:\Users\Admin\dev\zaif-trade-bot`
- [ ] 前回の検証結果を確認済み
- [ ] 実行時間を確保 (各5-8時間、計10-16時間)

---

## 🚀 実行コマンド (コピペ用)

### Step 1: max_grad_norm 100k検証 (seed 42)

**推定所要時間**: 5-8時間

```cmd
REM Command Promptで実行推奨
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat

REM max_grad_norm 100k × seed 42
python -m ztb.training.binary_search.max_grad_norm_optimized --mode binary --max_iterations 2 --timesteps 100000 --seed 42 --search_range 5.0,5.05,6.3,7.5,7.525

REM 実行完了後、結果を確認
notepad binary_search_results\max_grad_norm_binary_search.jsonl
```

**PowerShell版**:
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
& .\.venv\Scripts\Activate.ps1

# max_grad_norm 100k × seed 42
python -m ztb.training.binary_search.max_grad_norm_optimized `
  --mode binary `
  --max_iterations 2 `
  --timesteps 100000 `
  --seed 42 `
  --search_range 5.0,5.05,6.3,7.5,7.525

# 結果確認
Get-Content binary_search_results\max_grad_norm_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json | Format-List
```

---

### Step 2: max_grad_norm 100k検証 (seed 123 - 再現性確認)

**推定所要時間**: 5-8時間  
**注意**: Step 1 完了後に実行してください

```cmd
REM Command Promptで実行推奨
cd C:\Users\Admin\dev\zaif-trade-bot
.venv\Scripts\activate.bat

REM max_grad_norm 100k × seed 123
python -m ztb.training.binary_search.max_grad_norm_optimized --mode binary --max_iterations 2 --timesteps 100000 --seed 123 --search_range 5.0,5.05,6.3,7.5,7.525

REM 実行完了後、結果を確認
notepad binary_search_results\max_grad_norm_binary_search.jsonl
```

**PowerShell版**:
```powershell
cd C:\Users\Admin\dev\zaif-trade-bot
& .\.venv\Scripts\Activate.ps1

# max_grad_norm 100k × seed 123
python -m ztb.training.binary_search.max_grad_norm_optimized `
  --mode binary `
  --max_iterations 2 `
  --timesteps 100000 `
  --seed 123 `
  --search_range 5.0,5.05,6.3,7.5,7.525

# 結果確認
Get-Content binary_search_results\max_grad_norm_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json | Format-List
```

---

## 📊 結果の確認方法

### 最終結果の抽出

**Command Prompt**:
```cmd
REM 最終行を確認 (手動)
notepad binary_search_results\max_grad_norm_binary_search.jsonl
REM 最終行の "event": "complete" を見る
```

**PowerShell**:
```powershell
REM 最終行を自動抽出
Get-Content binary_search_results\max_grad_norm_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json

REM 整形表示
Get-Content binary_search_results\max_grad_norm_binary_search.jsonl | Select-Object -Last 1 | ConvertFrom-Json | Format-List
```

### 期待される出力例

```json
{
  "event": "complete",
  "timestamp": "2025-10-10T...",
  "iteration": 5,
  "parameter_value": 5.05,    // ← 最適値
  "score": -270.123456        // ← スコア
}
```

---

## 🔍 結果の判定基準

### 再現性の確認

**Step 1 (seed 42)** と **Step 2 (seed 123)** の結果を比較:

| 項目 | 判定基準 | アクション |
|------|---------|-----------|
| **最適値が一致** | 差が±0.5以内 | ✅ 再現性あり → 本番適用OK |
| **最適値が近似** | 差が±1.0以内 | ⚠️ 中程度の再現性 → 平均値を採用 |
| **最適値が大きく異なる** | 差が±1.0超 | ❌ 再現性なし → さらに追加検証 |

### スコア改善の確認

100kスコアと30kスコアを比較:

```
30k最良スコア: -286.41 (5.05) または -287.44 (7.525)
100kスコア: ?

improvement = 30kスコア - 100kスコア
```

| improvement | 判定 | アクション |
|-------------|------|-----------|
| **≥10** | ✅ 統計的有意 | 本番適用推奨 |
| **5-10** | ⚠️ 有意傾向 | 追加検証検討 |
| **<5** | ❌ 微差 | デフォルト継続検討 |

---

## 📝 結果記録テンプレート

検証完了後、以下の情報を記録してください:

```markdown
## max_grad_norm 100k検証結果

### Run 1: seed 42
- **実行日時**: 2025-10-XX HH:MM
- **最適値**: X.XX
- **スコア**: -XXX.XX
- **所要時間**: X時間XX分
- **エピソード数**: XX

### Run 2: seed 123
- **実行日時**: 2025-10-XX HH:MM
- **最適値**: X.XX
- **スコア**: -XXX.XX
- **所要時間**: X時間XX分
- **エピソード数**: XX

### 再現性分析
- **最適値の差**: X.XX (Run1) - X.XX (Run2) = X.XX
- **スコアの差**: -XXX.XX - (-XXX.XX) = X.XX
- **判定**: ✅/⚠️/❌

### 本番適用推奨値
- **推奨値**: X.XX
- **根拠**: 両シードで一貫 / 平均値 / その他
```

---

## 🔄 並行実行の注意

**同時実行は非推奨** (リソース競合のため):
- Step 1完了後にStep 2を実行
- または、異なるマシン/GPUで実行

**TensorBoardモニタリング** (別ターミナル):
```cmd
.venv\Scripts\activate.bat
tensorboard --logdir tensorboard/binary_search/
start http://localhost:6006
```

---

## 💾 バックアップ推奨

長時間実行のため、定期的なバックアップを推奨:

```cmd
REM 結果ファイルのバックアップ
copy binary_search_results\max_grad_norm_binary_search.jsonl binary_search_results\max_grad_norm_binary_search_backup_20251010.jsonl

REM モデルファイルのバックアップ (大容量注意)
xcopy /E /I models\binary_search models\binary_search_backup_20251010
```

---

## ❓ トラブルシューティング

### エラー: CUDA out of memory
**解決策**: `--no-progress-bar` を追加してメモリ削減
```cmd
python -m ztb.training.binary_search.max_grad_norm_optimized --mode binary --max_iterations 2 --timesteps 100000 --seed 42 --search_range 5.0,5.05,6.3,7.5,7.525 --no-progress-bar
```

### 実行が途中で停止
**解決策**: 
1. 結果ファイルを確認 (`binary_search_results\max_grad_norm_binary_search.jsonl`)
2. 最後に評価された値を確認
3. `--search_range` から既に評価済みの値を除外して再実行

### スコアが想定より悪い
**原因**: ランダムシードの影響、データの順序など
**対処**: 正常動作、複数シードで平均化

---

## 📚 関連ドキュメント

- **検証結果サマリー**: [`VALIDATION_RESULTS_2025-10-10.md`](./VALIDATION_RESULTS_2025-10-10.md)
- **進捗管理**: [`PARAMETER_VALIDATION_TRACKING.md`](./PARAMETER_VALIDATION_TRACKING.md)
- **クイックリファレンス**: [`PARAMETER_VALIDATION_QUICK_REFERENCE.md`](./PARAMETER_VALIDATION_QUICK_REFERENCE.md)

---

**準備ができたら、Step 1から実行してください!** 🚀
