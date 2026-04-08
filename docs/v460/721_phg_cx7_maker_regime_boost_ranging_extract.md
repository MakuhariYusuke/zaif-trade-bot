# 721# CX7 `_regime_boost_ranging()` OBI 抽出

## 背景

- prompt 上は `720# CX7` として依頼されたが、`docs/v460/720_*` は既存の int-key silent drop 修正文書で使用済み。
- そのため docs 側は **721#** で記録し、実装意図だけ `720# CX7` に追随する。
- 対象は [maker_regime_boost.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/scripts/v460/lib/maker_regime_boost.py) の `_regime_boost_ranging()`。

## prompt 検証

### 要求

1. `compute_ranging_obi_multiplier(...)` 周辺の OBI 非対称ロジックを private helper 化
2. `_regime_boost_ranging()` を 60 行以下にする
3. 既存 focused test を全パスさせる
4. 新 helper の unit test を追加

### 妥当性

- この prompt は妥当。
- 既存実装の責務境界とも整合する。
- `RegimeBoostMixin` はすでに sub-stage 分割済みであり、OBI 非対称ロジックだけが残っていた。

## 実装

- 追加:
  - `_apply_ranging_obi_asymmetry(...)`
- 変更:
  - `_regime_boost_ranging()` は side 別 discount 解決 + scale/logging に集中

### ロジック変更について

- **ロジック変更はしていない**
- helper には次の既存処理だけを移した
  - factor 無効時の no-op
  - threshold 以下の no-op
  - `compute_ranging_obi_multiplier(...)`
  - `min_offset_ratio` / `effective_max_ratio(side)` による multiplier clamp

## hidden task

- prompt では触れられていないが、`720` の docs 番号衝突を避ける必要があった
- source-contract test だけでは不十分なので、helper の behavior unit test を追加

## テスト

- [test_260_compute_extract_regime_split.py](/mnt/c/Users/Admin/dev/zaif-trade-bot/tests/unit/v460/test_260_compute_extract_regime_split.py)
  - source contract:
    - `_regime_boost_ranging()` が helper を呼ぶ
    - helper が `compute_ranging_obi_multiplier(...)` を使う
  - unit:
    - helper が既存 formula と一致
    - factor 無効時に base multiplier を返す

## 確認コマンド

```bash
.venv/Scripts/python.exe -m pytest tests/unit/v460/test_260_compute_extract_regime_split.py -x --tb=short --no-cov
python3 -m py_compile scripts/v460/lib/maker_regime_boost.py tests/unit/v460/test_260_compute_extract_regime_split.py
```
