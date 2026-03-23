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

## 深掘り: どの順番で適用すると労力が小さいか

targeted mypy は「とりあえず全部にかける」より、次の順で当てる方が明らかに効率が良い。

1. **shared contract が既にある層**
   - 例:
     - `ztb.metrics.fill_quality`
     - `dict[str, object]` payload
     - `Protocol`
     - `dataclass`
     - `TypeAlias`
   - ここは `Any` を新しく発明しなくて済む

2. **mixin / helper の境界**
   - `fill_record_builder`
   - `fill_cycle_executor`
   - `config_hot_reload`
   - ここは「実際に必要な属性を宣言していない」だけのケースが多い

3. **analysis / reporting の read-only 層**
   - `scripts/v460/analysis/*`
   - 数値比較前に `float(...)` 正規化
   - `numpy.typing` と `TypeAlias` で enough

4. **stateful orchestration の本体**
   - ここは最後
   - targeted mypy を使っても、先に shared contract を固めないと直す量が増える

## 深掘り: low-risk fix の判断基準

今回の運用では、次を **low-risk** とみなす。

- 既存 API の返り値型を明示する
- lazy import resolver に返り値型を付ける
- `dict[str, object]` / `TypeAlias` / `Protocol` を使って `Any` 流出を止める
- optional 実引数に合わせて helper シグネチャを広げる
- 読み取り専用 script で比較前に `float(...)` 正規化する

逆に、次は **別タスク扱い** にする。

- repo-wide mypy を一気に clean にする
- stateful object の大分割
- 新しい base class / 共通抽象の導入
- runtime 振る舞いを変える型修正

## 深掘り: analysis 系での型ルール

analysis 系は次を正本ルールにする。

1. レコードは `dict[str, object]`
   - shared loader/filter と一致させる
2. 比較・集計前に `float(...)` / `int(...)` 正規化する
   - `object | None` のまま演算しない
3. numpy 配列は `numpy.typing` を使う
   - `NDArray[np.float64]`
4. `type: ignore` は最後の手段
   - まず `cast(...)` と `TypeAlias` を使う

## 深掘り: 実運用上の止めどころ

targeted mypy を回していて、次の状態になったらその module 群はいったん止めてよい。

- requested targets で diagnostics が 0
- focused pytest が通る
- その先が「他人の並行差分」か「repo-wide baseline debt」に依存する

この止めどころを守ると、型改善が無限化しにくい。
