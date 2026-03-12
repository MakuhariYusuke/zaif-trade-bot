# Git リポジトリ整理 — AI コーディングエージェント用プロンプト

## あなたの役割

あなたは Git リポジトリの整理専門エージェントです。  
対象リポジトリ `zaif-trade-bot` の肥大化した Git 作業ツリーとヒストリを  
**安全に・段階的に・不可逆操作は最後に** クリーンアップして下さい。

---

## 現状の診断

| 指標 | 値 | 問題度 |
|------|-----|--------|
| `.git/` サイズ | **~11 GB** | 🔴 異常 (pack 3 GB + loose objects) |
| 総コミット数 | 493 | 普通 |
| Git tracked ファイル数 | 6,262 | 🟡 多い |
| `git status` 差分数 | **2,575** | 🔴 致命的 (diff が終わらない) |
| 内訳: deleted (tracked→実体なし) | 2,246 | 🔴 大量の幽霊エントリ |
| 内訳: untracked (新規) | 233 | 🟡 |
| 内訳: modified | 96 | 普通 |
| `.gitignore` 行数 | 91 | 🟡 不足気味 |

### 主要な問題

1. **幽霊 D (Deleted) 2,246 件**: 過去にトラックされたファイルが物理削除されたが `git rm` されていない。`git status` / `git diff` が常時 2,000+ ファイルを走査し、IDE や AI エージェントが機能不全。
2. **.git 11 GB**: 大きなバイナリ (モデル `.pt`/`.pth`、`.parquet`、datasets) が履歴に残存している可能性が高い。
3. **untracked 233 件**: `configs/`, `archived/`, `backtest/` 等に未追跡ファイルが散在。`.gitignore` の不足。

---

## クリーンアップ手順 (この順序で実行)

### Phase 1: 安全バックアップ (必須・最初に)

```powershell
# 現在の HEAD をタグで保存
git tag backup-before-cleanup

# .git ごとフルバックアップ (別ドライブ推奨)
robocopy . D:\backup\zaif-trade-bot /MIR /XD .venv node_modules
```

### Phase 2: Deleted ファイルの一括 `git rm` (最優先)

```powershell
# 確認 (dry-run)
git status --porcelain | Where-Object { $_ -match '^ D ' } | Measure-Object -Line

# 一括 git rm (インデックスから除去)
git ls-files --deleted -z | git rm --cached -z --stdin

# コミット
git commit -m "cleanup: git rm 2,246 deleted files (ghost entries)"
```

**効果**: `git status` が即座に軽くなる。HEAD コミットから消えるが、履歴には残るためいつでも復元可能。

### Phase 3: `.gitignore` 強化

以下のパターンが不足している可能性が高い。現在の `.gitignore` と照合し、不足分を追記:

```gitignore
# モデル・チェックポイント (大容量バイナリ)
*.pt
*.pth
*.onnx
*.pkl
*.joblib
checkpoints/
models/

# データ・キャッシュ
data/
cache/
*.parquet
*.h5
*.hdf5
*.feather
*.jsonl.gz

# 実験・一時ファイル
results/
runs/
temp/
tmp/
*.tmp
debug/
plots/
notebooks/.ipynb_checkpoints/

# archived ディレクトリ (既に別管理)
archived/

# バックテスト結果
backtest/

# hypothesis テストデータベース
.hypothesis/

# IDE
.idea/
.vscode/settings.json
*.code-workspace

# OS
Thumbs.db
.DS_Store
```

追記後:
```powershell
git add .gitignore
git commit -m "cleanup: strengthen .gitignore for data/models/temp"
```

### Phase 4: untracked ファイルの仕分け

233 件の untracked を以下に分類:

| 分類 | 処置 |
|------|------|
| 追跡すべきもの (ソースコード) | `git add` → コミット |
| `.gitignore` で除外すべきもの | Phase 3 で対応済みのはず |
| 不要な一時ファイル | 物理削除 |

```powershell
# untracked の一覧 (ディレクトリ単位)
git status --porcelain | Where-Object { $_ -match '^\?\?' } |
  ForEach-Object { ($_ -replace '^\?\?\s+','') -split '/' | Select-Object -First 1 } |
  Sort-Object | Group-Object | Sort-Object Count -Descending
```

### Phase 5: Git 履歴の大容量ファイル掃除 (要注意・不可逆)

```powershell
# まず巨大ファイルを特定
git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  Where-Object { $_ -match '^blob' } |
  ForEach-Object {
    $parts = $_ -split '\s+', 4
    [PSCustomObject]@{Size=[long]$parts[2]; Path=$parts[3]}
  } |
  Sort-Object Size -Descending |
  Select-Object -First 30 |
  Format-Table @{N='SizeMB';E={[math]::Round($_.Size/1MB,1)}}, Path
```

10 MB 以上のファイルが見つかった場合:

```powershell
# git-filter-repo による履歴書換え (pip install git-filter-repo)
# ⚠️ 不可逆操作 — backup-before-cleanup タグ確認後に実行
git filter-repo --strip-blobs-bigger-than 10M --force

# 再 pack + GC
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Phase 6: 最終圧縮

```powershell
git gc --aggressive --prune=now
git repack -a -d --depth=250 --window=250

# サイズ確認
(Get-ChildItem .git -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
```

---

## 制約・注意事項

- **Phase 5 (filter-repo) は最終手段**。Phase 1-4 だけでも `git status` 問題は解決する
- `main` ブランチのみの単一ブランチリポジトリ (リモートなし) なので rebase / force push のリスクは低い
- `.env` ファイルが履歴に入っていないか確認: `git log --all --diff-filter=A -- '*.env' '.env'`
- `data/v460/raw/` は現在収集中のライブデータ。誤削除禁止
- 作業中の `scripts/v460/run_fill_test.py` が 72h バックグラウンド実行中の場合あり。ファイルロックに注意

## 期待する最終状態

| 指標 | 目標値 |
|------|--------|
| `.git/` サイズ | < 500 MB |
| `git status` 差分 | < 20 件 |
| `git status` 実行時間 | < 2 秒 |
| tracked ファイル | ソースコード + ドキュメントのみ (< 1,000) |
| `.gitignore` | データ・モデル・一時ファイルを網羅 |

## 完了報告フォーマット

```
cleanup完了:
  .git サイズ: XX GB → YY MB
  tracked: XXXX → YYY files
  git status 差分: XXXX → YY files
  削除したパターン: [...]
  filter-repo 対象: [...] (実施した場合)
```
