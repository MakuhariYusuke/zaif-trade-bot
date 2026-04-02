# Codex Task: 701# T3 — sac_v432 大型 JSON の archived/ 移動

## ゴール

`ztb/analysis/` にある v432 時代の大型結果 JSON (7 ファイル, 合計 7.8 MB) を `archived/analysis/` に移動し、git tracked から除外する。

## 背景

これらのファイルは v432 (2025年) の最適化実験結果で、現行 v460 では参照されていない。
IDE の git 操作・ファイルインデックスの負荷軽減が目的。

## 移動対象ファイル

```
ztb/analysis/sac_v432_1_advanced_position_management_results.json   (960 KB)
ztb/analysis/sac_v432_2_win_rate_optimization_results.json          (701 KB)
ztb/analysis/sac_v432_3_entry_exit_enhancement_results.json        (1,172 KB)
ztb/analysis/sac_v432_4_profit_focused_optimization_results.json   (1,372 KB)
ztb/analysis/sac_v432_5_strict_entry_optimization_results.json     (1,565 KB)
ztb/analysis/sac_v432_6_ensemble_approach_results.json             (1,743 KB)
ztb/analysis/sac_v432_7_real_market_data_results.json               (349 KB)
```

## 実装手順

1. `archived/analysis/` ディレクトリが存在しなければ作成
2. 上記 7 ファイルを `archived/analysis/` に `git mv` で移動
3. `ztb/analysis/` 内の Python コード (.py) で上記 JSON を参照している箇所がないか grep 確認
   - 参照があれば import パスを更新
   - **参照がなければ何も変更不要** (期待: 参照なし)
4. `.gitignore` の `archived/` エントリが既に存在するので、移動後はgit tracked から自動除外

## テスト

`tests/unit/v460/test_701_archived_v432.py`:

1. `test_v432_json_not_in_ztb_analysis` — `ztb/analysis/sac_v432_*.json` が存在しないことを確認
2. `test_v432_json_in_archived` — `archived/analysis/sac_v432_*.json` が存在することを確認
3. `test_no_code_references_v432_json` — `ztb/` 配下の .py で `sac_v432_` を参照しているファイルがないことを確認

## 制約

- `git mv` を使用 (git history を保持)
- archived/ は .gitignore で除外されるため、移動後は `git rm --cached` も不要
- ztb/analysis/__init__.py や他の __init__.py の変更は不要 (JSON は Python import 対象外)
- Type annotations 必須

## 注意

- `archived/` ディレクトリは `.gitignore` で除外されている。`git mv` 後に `.gitignore` の `archived/` エントリとの衝突があれば、一時的に `.gitignore` から `archived/analysis/sac_v432_*.json` を除外するか、`git mv` の代わりに手動コピー + `git rm` で対応
- 実際には `archived/` が .gitignore されているため、`git mv` は失敗する可能性がある。その場合は:
  1. ファイルを `archived/analysis/` にコピー (`shutil.copy2`)
  2. `git rm ztb/analysis/sac_v432_*.json` で tracked から削除
  3. `archived/analysis/` は gitignore されるのでそのまま

## 成果物

- `archived/analysis/sac_v432_*.json` (移動先)
- `ztb/analysis/sac_v432_*.json` が削除された状態
- `tests/unit/v460/test_701_archived_v432.py`
