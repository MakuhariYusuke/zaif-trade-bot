# 720# _clone_config_value int-key サイレントドロップ修正

## 概要

`config_loader.py::_clone_config_value()` が YAML から生成される **int キーの dict を無言でドロップ** していた致命的バグを発見・修正。

sell_hour_offset_boost (17 entries)、hour_ceiling_mult (10 entries)、skip_gate.hour_offsets (9 entries) の **計 36 エントリが全て空 dict `{}` に変換** され、3 つの時間帯別防御メカニズムが **導入以来一度も機能していなかった**。

## 根本原因

### バグ箇所: `scripts/v460/lib/config_loader.py` L44-47

```python
def _clone_config_value(value):
    ...
    if isinstance(value, dict):
        return {
            str(k): _clone_config_value(v)
            for k, v in value.items()
            if isinstance(k, str)  # ← int key をすべてドロップ
        }
```

### メカニズム

1. PyYAML は `0: 1.5` を `{0: 1.5}` (int key) としてパースする
2. `_clone_config_value` は `isinstance(k, str)` フィルタで **int key を除外**
3. sell_hour_offset_boost `{0: 1.5, 2: 5.0, ...}` → `{}` (空)
4. `parse_fill_config_yaml` で `shob = yaml_cfg.get("sell_hour_offset_boost", {})` → 空 dict
5. `if shob:` → False → FillTestConfig のデフォルト空 dict が使用される
6. `resolve_optional_hour_float({}, hour)` → 常に `None` → boost 適用ゼロ

### 実証

```
Raw YAML sell_hour_offset_boost: key_types={'int'}, count=17
After _clone_config_value:       count=0  ← 全滅
```

## 影響範囲

| 設定 | エントリ数 | 用途 | 影響 |
|------|-----------|------|------|
| `sell_hour_offset_boost` | 17 | sell 時間帯別 offset ×1.3〜5.0 | **100% no-op** — 高 AS 時間帯の防御が全く機能していない |
| `hour_ceiling_mult` | 10 | deep-night offset ceiling 緩和 ×1.3〜3.5 | **100% no-op** — ceiling が常時 default で防御を抑制 |
| `skip_gate.hour_offsets` | 9 | 時間帯別 skip_gate 閾値調整 | **100% no-op** — 時間帯別の閾値厳格化が無効 |

### 影響の深刻度

- **sell_hour_offset_boost 無効**: UTC 2h (JST 11h) や UTC 4h (JST 13h) の AS 100% 帯で ×5.0 boost が適用されるべきところ、base offset のまま。逆選択損失の主因。
- **hour_ceiling_mult 無効**: 688# で ceiling を ×3.0-3.5 に拡大した設定が死んでおり、sell_hour_boost が仮に効いても ceiling で頭打ちになる二重無効化の危険があった。
- **skip_gate.hour_offsets 無効**: 時間帯別スキップ閾値調整 (634# P0) が全て無効。

## 修正

```python
# Before (720# fix)
if isinstance(value, dict):
    return {
        str(k): _clone_config_value(v)
        for k, v in value.items()
        if isinstance(k, str)
    }

# After (720# fix)
if isinstance(value, dict):
    return {
        k: _clone_config_value(v)
        for k, v in value.items()
        if isinstance(k, (str, int))
    }
```

- `str` に加えて `int` キーも保持
- `str(k)` 変換を廃止し、元のキー型を維持（parser 側で `int(k)` 変換済み）

## 回帰テスト追加

### `test_config_validation.py::TestCloneConfigValue`
- `test_str_keys_preserved` — str key のクローンが既存動作を壊さない
- `test_int_keys_preserved` — int key が保持される（720# 回帰防止）
- `test_nested_int_keys_preserved` — sell_hour/hour_ceiling パターン
- `test_clone_is_independent` — 独立性（元 dict が変更されない）
- `test_list_cloned` — list クローンの既存動作確認

### `test_fill_test_config.py::TestIntKeyYamlMapsEndToEnd`
- `test_sell_hour_offset_boost_not_empty` — YAML→FillTestConfig で消えない
- `test_hour_ceiling_mult_not_empty` — YAML→FillTestConfig で消えない

## 発見の経緯

716-719# の sell_hour_offset_boost 100% no-op 調査で:
1. YAML に 17/24 時間定義あり → 正常
2. `parse_fill_config_yaml()` のパース → int キー変換正常
3. `_apply_sell_hour_boost()` 直接テスト → 正常動作
4. ランタイムで 100% no-op → **関数自体ではなく config ロード経路にバグ**
5. `_clone_config_value()` が int key を `isinstance(k, str)` で除外 → **root cause 確定**

## 718# への反論

718# は sell_hour を「phantom bug」（5 hours しか定義なし）と主張したが:
- YAML には **17 hours** 定義（718# の事実誤認）
- 問題は sell_hour の関数ロジックではなく config ロード層
- 718# の「モデル廃棄 + OBI 導入」提案は根本原因の把握なく提案された点に留意

## 次のアクション

| 優先度 | 項目 | 詳細 |
|--------|------|------|
| **P0** | 本番デプロイ | 修正を fill_test が参照する SHA に反映 |
| P1 | hour_ceiling_mult 有効化検証 | ceiling 緩和が正しく動作するか確認 |
| P2 | 経過観察 | 修正後の sell AS 率・PnL 変化をモニタ |
| P3 | spread_as_guard staleness | 60 秒 guard の last_spread 問題（別チケット） |
