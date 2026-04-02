# 701# IDE パフォーマンス監査 & クリーンアップ計画

## 調査背景

IDE (VS Code) の重さが体感で悪化。700# Codex 完了後の全面点検として、ワークスペース全体のファイル構成・設定・テスト基盤を多角的に監査した。

---

## 1. 診断結果サマリ

### ワークスペース全体像

| 指標 | 値 | 評価 |
|------|------|------|
| git tracked ファイル数 | **4,749** | 大規模 |
| untracked ファイル数 | 12 | 正常 |
| ディスク上の全ファイル数 (.venv/.git 除外) | **41,174** | 異常 |
| ディスク使用量 | **52 GB** | 異常 |
| git index サイズ | 0.4 MB | 正常 |
| git pack サイズ | 51.2 MB | 正常 |
| Python (.py) tracked 数 | 2,524 | 大規模 (Pylance負荷) |

### ディレクトリ別ディスク使用量 (top 10)

| ディレクトリ | サイズ | ファイル数 | 状態 |
|-------------|--------|-----------|------|
| checkpoints/ | **48.2 GB** | 6,729 | gitignored, watcher除外済 |
| data/ | 1.5 GB | 15,106 | gitignored, watcher除外済 |
| .venv/ | 1.5 GB | 37,381 | gitignored, watcher除外済 |
| models/ | 1.3 GB | 231 | gitignored, watcher除外済 |
| results/ | 727 MB | 1,566 | gitignored, watcher除外済 |
| logs/ | 254 MB | 2,112 | gitignored, watcher除外済 |
| reports/ | 94 MB | 1,519 | gitignored, watcher除外済 |
| temp/ | 57 MB | 923 | gitignored, watcher除外済 |
| ztb/ | 32 MB | 1,815 | tracked, ソースコード |
| scripts/ | 28 MB | 1,052 | tracked, ソースコード |

---

## 2. 問題の根本原因分析

### P1 (Critical): config/ 内の ab_search_temp ゴミファイル — 5,402 個

```
config/
  ab_search_temp_0002ed40.json  ← 5,402 個 (全て .json)
  ab_search_temp_0003cba0.json
  ...
  ab_search/                     ← 正規のサブディレクトリ (2 ファイル)
```

- `.gitignore` に `config/ab_search_temp_*.json` があるので **git tracked ではない**
- しかし **ディスク上に 5,402 ファイル** が存在
- `files.watcherExclude` に `**/config/**` は設定済みだが、OS ファイルシステム自体が重い
- `files.exclude` には **config は含まれていない** → Explorer に表示され得る
- **影響**: ファイルシステムスキャン時 (VS Code 起動, git operations) に 5K ファイルの存在確認が走る

**対策**: 全削除 → 安全 (gitignore 対象, 古い A/B 検索の一時ファイル)

### P2 (High): data/temp/ 内のキャッシュファイル — 12,740+ 個

```
data/temp/
  .mypy_cache/   ← 12,740 ファイル (!)
  .ruff_cache/   ← 183 ファイル
  .hypothesis/   ← 70 ファイル
  tmp-*          ← 各種テスト一時ディレクトリ
```

- `.gitignore` で `data/` が除外済み
- `files.watcherExclude` で `**/data/**` 除外済み
- しかし `.mypy_cache` が data/temp の中に 12,740 個。通常 mypy は `.mypy_cache/` をルート配置するが、何らかの temp 実行環境で data/temp に生成された
- **影響**: ディスクI/O、メモリマッピングのオーバーヘッド

**対策**: `data/temp/` 全削除 → 安全 (自動再生成されるキャッシュのみ)

### P3 (High): tracked の大量ドキュメント・結果 JSON

| カテゴリ | ファイル数 | サイズ合計 |
|---------|-----------|-----------|
| docs/v460/ | 679 | 8.2 MB |
| docs/ (v460以外の旧版) | 646 | ~10 MB |
| ztb/analysis/sac_v432_*.json | 7 | 7.8 MB |
| CHANGELOG.md | 1 | 521 KB |

- **docs/v460/ 679 ファイル**: 現行バージョンのドキュメント。IDE の Language Server がマークダウンを解析
- **docs/ 旧版 646 ファイル**: v455-v459, evaluation, api, 等。参照頻度ゼロに近い
- **sac_v432_*.json**: v432 時代の最適化結果 JSON (最大 1.7 MB/file)。コード参照なし

**対策**: 
- sac_v432 JSON → archived/ または .gitignore へ移動
- 旧版 docs → 段階的に archived に移動、または `files.exclude` に追加

### P4 (Medium): test_codex_408_409_fixes.py のハング

```
TestT1IdempotencyLock (3 tests) → class 全体実行でハング
  - test_process_lock_is_exclusive: 排他ロック + timeout
  - test_process_lock_releases_on_exit: ロック取得→解放
  - test_stale_lock_recovery: stale lock の回復

個別実行 → OK
class全体実行 → KBI (KeyboardInterrupt)
T1 以外の 33 テスト → 全パス (2.08s)
```

**根因**: `IdempotencyStore._process_lock()` のファイルロックがテスト間で解放されずデッドロック。tmp_path の分離が不十分か、lock_file のクリーンアップ漏れ。

**対策**: T1 テストの fixture 分離修正 (各テストで独立した tmp_path/db_path を確認)

### P5 (Medium): Pylance の解析スコープ

- tracked .py: 2,524 ファイル
- `python.analysis.exclude` で archived/sb3_contrib/config 等は除外済み
- `python.analysis.diagnosticMode`: `openFilesOnly` ← これは正しい
- **しかし** `python.analysis.indexing: true` でインデックス構築は全ファイル対象
- `python.analysis.persistAllIndices: false` でメモリ常駐はしない ← OK

**現状**: 設定は 644# で最適化済み。大幅な追加対策余地は少ない。

### P6 (Low): ルート直下の散らかり

| ファイル | サイズ | 状態 |
|---------|--------|------|
| test_collect.txt | 595 KB | gitignored |
| CHANGELOG.md | 521 KB | tracked, 肥大化 |
| test_results.json | 83 KB | gitignored |
| testout.txt | 51 KB | gitignored |
| README.md | 56 KB | tracked |
| CASCADE_DISCOVERIES.md | 1.3 KB | tracked |
| .coverage | 176 KB | gitignored |

- CHANGELOG.md 521KB は大きいが分割の手間に見合わない
- ルート直下の gitignore 対象は IDE パフォーマンスへの影響は軽微

---

## 3. 多角的検証

### 視点 1: ファイルウォッチャー負荷

644# で `files.watcherExclude` を設定済み。主要な大量ディレクトリ (checkpoints, data, models, results 等) は除外。
**未除外**: `plots/`, `notebooks/`, `prometheus_client/`, `utils/`, `.devcontainer/` → 小さいため影響軽微。

### 視点 2: Git 操作速度

- git index: 0.4 MB (正常)
- git pack: 51.2 MB (正常)
- tracked 4,749 ファイル → `git status` は高速だが、untracked scan で config/ の 5,402 ファイルがヒット (.gitignore のパターンマッチ処理が毎回走る)
- **改善**: config/ab_search_temp_*.json を物理削除すれば git status も高速化

### 視点 3: Pylance / Language Server

- openFilesOnly モードで問題なし
- ただし **ztb/ 下に 1,258 ファイル (277 trading + 254 training + ...)** があり、import graph の構築は全体に及ぶ
- `python.analysis.autoSearchPaths: false` で extra path 探索は無効 ← Good
- **ボトルネック**: ztb/analysis/ 内の大型 JSON はPylance対象外 (非.py) だが、同ディレクトリの .py が 164 ファイル

### 視点 4: テスト基盤

- v460 テスト: 5,916 テスト (263 ファイル)
- フル実行: ~53s
- test_codex_408 (36 テスト) がハング → テスト全件を `-x` で通すと 76% で必ず止まる
- **根因**: ファイルロックのテスト分離問題 (P4)

### 視点 5: 将来的肥大化リスク

700# 以降の Codex 投入で毎回:
- テストファイル +1-3
- docs +1-2
- prompts +1-4
- ztb/scripts 変更

→ 月間 +30-50 ファイルのペース。年間 +400 ファイル。**管理可能だが蓄積はモニタすべき**。

---

## 4. 推奨アクション (優先度順)

### Codex T1: config/ ゴミファイル削除 + data/temp/ クリーンアップ (P1+P2)

**安全に削除可能**:
- `config/ab_search_temp_*.json` (5,402 個) → gitignore 対象、古い A/B 検索一時ファイル
- `data/temp/` (13,500+ 個) → gitignore 対象、キャッシュファイル

**scripts/cleanup_workspace.py 新規作成**:
- 定期実行可能な cleanup ユーティリティ
- dry-run モードで削除対象をプレビュー
- config/ab_search_temp_*, data/temp/.mypy_cache, data/temp/.ruff_cache 等を対象

### Codex T2: test_codex_408 ファイルロックデッドロック修正 (P4)

**修正箇所**: `tests/unit/v460/test_codex_408_409_fixes.py::TestT1IdempotencyLock`
- 各テストで固有の tmp_path/db_path を使用していることを確認
- fixture の teardown で lock ファイルの強制解放を追加
- クラス共有状態の排除

### Codex T3: sac_v432 大型 JSON の archived/ 移動 (P3)

**移動対象**: `ztb/analysis/sac_v432_*.json` (7 ファイル, 7.8 MB)
- `archived/analysis/` に移動
- import/reference のコード修正 (参照箇所を grep で確認)
- git tracked から除外

### 追加設定 (Codex で反映可能)

```jsonc
// .vscode/settings.json に追加
"files.exclude": {
    // 既存エントリに追加
    "**/config": true,           // ab_search_temp 5402 ファイル非表示
    "**/plots": true,            // 生成画像
    "**/notebooks": true,        // Jupyter
    "**/prometheus_client": true  // 監視クライアント
}
```

---

## 6. 実施結果 (2026-04-03)

**ステータス: 完了**

### 完了

- `scripts/v460/tools/cleanup_workspace.py` を新規作成
  - dry-run default
  - tracked file 保護
  - `config/ab_search_temp_*.json`
  - `data/temp/.mypy_cache`
  - `data/temp/.ruff_cache`
  - `data/temp/.hypothesis`
  - `data/temp/.pytest_cache`
  - `data/temp/tmp-*`
  を対象に実装
- `tests/unit/v460/test_701_cleanup_workspace.py` を追加
- `tests/unit/v460/test_codex_408_409_fixes.py::TestT1IdempotencyLock` を修正
  - lock cleanup fixture を追加
  - db 名を明示分離
  - stale lock 不在/解放確認を追加
- `ztb/analysis/sac_v432_*.json` を `archived/analysis/` に移動
- `.vscode/settings.json` の `files.exclude` を追加
  - `config`
  - `plots`
  - `notebooks`
  - `prometheus_client`
  - `archived`

### 追加で回収した hidden task

- `archived/` は `.gitignore` 対象だが、既存 tracked payload があるため `git mv -f` で履歴保持のまま移動
- archive 移動は prompt どおり `ztb` だけでなく `scripts/tests` も含めてコード参照なしを確認
- IDE 体感には watcher だけでなく Explorer の `files.exclude` も効くため、設定反映まで実施

### 検証

- cleanup script:
  - `5 passed in 1.29s`
- idempotency lock:
  - `TestT1IdempotencyLock` focused pass
- archive:
  - v432 JSON の archive 配置 / 非参照を test で固定

### 次の運用

- 定期 cleanup:
  - `python -m scripts.v460.tools.cleanup_workspace`
  - `python -m scripts.v460.tools.cleanup_workspace --execute`
- IDE が再び重くなったら、まず cleanup dry-run と `git status -uno` を確認する

---

## 5. 効果見積もり

| アクション | 削減ファイル数 | 削減サイズ | IDE 体感改善 |
|-----------|-------------|-----------|------------|
| config/ab_search_temp 削除 | ~5,400 | ~1.2 MB | **大** (git scan 高速化) |
| data/temp/ 削除 | ~13,000 | ~350 MB | 中 (ディスクI/O軽減) |
| sac_v432 JSON 移動 | 7 | 7.8 MB | 小 (git pack 軽量化) |
| files.exclude 追加 | - | - | 中 (Explorer 軽量化) |
| test_codex_408 修正 | - | - | テスト CI 安定化 |
| **合計** | **~18,400** | **~360 MB** | **体感 20-30% 改善期待** |

---

*生成: 2026-04-03 by cplt (701#)*
*入力: IDE パフォーマンス調査、ワークスペース構造分析*
