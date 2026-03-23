# 589# Targeted mypy 実用化

## 背景

repo 全体の `mypy --config-file mypy.ini ...` は、現状では次の 2 問題で日常的な変更確認に使いにくい。

1. 過去負債が大量に残っていて、変更箇所と無関係なエラーが多い
2. `scripts/v460` 周辺は file path / module path の与え方で見え方がぶれやすい

このため、「今回触ったファイルに新しい型問題が増えていないか」を即座に見る用途に向かない。

## 実装

`scripts/quality/run_targeted_mypy.py` を追加した。

### 役割

- `mypy.ini` はそのまま使う
- changed files / target modules だけを入力にする
- `follow-imports`
  - `fast`: `skip`
  - `deep`: `silent`
  を選べる
- unrelated baseline diagnostics は suppress し、対象ファイルに関係するものだけ表示する

### 使い方

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/lib/fill_config_parser.py \
  scripts/v460/lib/offset_pipeline.py
```

より深く追いたい場合:

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  --mode deep \
  scripts/v460/lib/fill_cycle_executor.py
```

## 意図

- repo-wide mypy の完全正常化は別タスク
- それとは別に、日々の refactor で「今回の差分が悪化していないか」を見られる入口を先に作る
- 556# で指摘された「mypy が二重化/広域ノイズで使いにくい」問題への運用面の即効薬として扱う

## 判断

- `mypy.ini` 自体を軽くしすぎるのは避けた
- 代わりに、strict config は維持しつつ targeted runner で practical な入口を作った
- repo-wide 負債の解消は継続課題だが、今回の入口だけでもローカル開発の確認コストはかなり下がる

## 初回適用で拾えたもの

- `scripts/v460/lib/config_hot_reload.py`
  - `_HotReloadableRunner` protocol に `_config_hash` が漏れていた
  - targeted mypy で検出し、その場で補修した

## フォローアップ

targeted runner をそのまま使って、`scripts/v460` の低リスクな型残差も追加で整理した。

- `scripts/v460/lib/fill_config.py`
  - lazy parser resolver の返り値型を明示
  - `from_yaml()` の `Any` 流出を解消
- `scripts/v460/lib/fill_record_builder.py`
  - mixin 依存属性を型宣言
  - `SkipGate` 関連 payload を optional 契約に揃えた
- `scripts/v460/lib/fill_cycle_executor.py`
  - cross-venue EMA state / narrow-spread counter / place_order 戻り値の型を明示

確認コマンド:

```bash
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py \
  scripts/v460/lib/fill_config.py \
  scripts/v460/lib/fill_record_builder.py \
  scripts/v460/lib/fill_cycle_executor.py \
  scripts/v460/lib/offset_pipeline.py \
  scripts/v460/lib/multiplicative_pipeline.py
```

結果:

```text
Success: no issues found in 5 source files
```
