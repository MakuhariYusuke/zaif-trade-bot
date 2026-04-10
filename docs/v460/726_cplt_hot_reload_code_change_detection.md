# 726# hot-reload 改善: コード変更検知 + CODE_COUPLED_FIELDS + YAML 凡例

## 背景

724# でコード変更 (`_is_bypass_mode_active` regime チェック) + YAML 変更
(`bypass_regime_exclude`) を同時に実施したが、hot-reload は YAML 値のみ反映し
コードは旧 SHA のまま。この「値は載るがコードが読まない」問題を検知可能にする。

## 変更内容

### 1. SHA 変更時のコード差分検知 (ランタイム WARNING)

`config_hot_reload.py` の `_do_reload()` で git SHA 変更を検知した際、
`git diff --name-only {old}..{new} -- scripts/v460/lib/` を実行し、
`.py` ファイルの変更があれば WARNING を出力。

```
[config_hot_reload] ⚠ CODE CHANGES DETECTED between e776027..ddd30ae — 3 Python
file(s) modified. YAML values updated but code logic changes require process
restart (hot-swap).
```

### 2. `_CODE_COUPLED_FIELDS` 注釈

```python
_CODE_COUPLED_FIELDS: dict[str, str] = {
    "skip_gate_bypass_regime_exclude": "ddd30ae",  # 724#
}
```

hot-reload 時にこれらのフィールドが変更された AND SHA も変更された場合、追加の
WARNING を出力:

```
[config_hot_reload] ⚠ CODE-COUPLED fields changed: ['skip_gate_bypass_regime_exclude'].
These fields require code from the corresponding commit to take effect.
```

### 3. `get_changed_py_files()` ヘルパー

`ztb/utils/git_utils.py` に追加。2つの SHA 間で指定パス配下の `.py` ファイル変更を返す。

### 4. YAML 凡例コメント

`configs/v460/fill_test.yaml` 先頭に hot-reload 凡例を追記:
- `⚠ CODE-COUPLED`: 値は載るが対応コード不在時は無効
- `⚠ RESTART-REQUIRED`: hot-reload 対象外

該当フィールドにも個別コメント追記:
- `bypass_regime_exclude` — CODE-COUPLED (724# ddd30ae)
- `symbol` — RESTART-REQUIRED
- `results_dir` — RESTART-REQUIRED
- `cross_venue_lead_lag.enabled` / `reference_exchange` — RESTART-REQUIRED

## 変更ファイル

| ファイル | 変更 |
|---------|------|
| `ztb/utils/git_utils.py` | `get_changed_py_files()` 追加 |
| `scripts/v460/lib/config_hot_reload.py` | `_CODE_COUPLED_FIELDS` + SHA diff 検知ロジック |
| `configs/v460/fill_test.yaml` | 凡例 + 個別 RESTART/CODE-COUPLED コメント |
| `tests/unit/v460/test_169_config_hot_reload.py` | CODE_COUPLED_FIELDS 整合性テスト 2件 |
| `tests/unit/utils/test_run_manifest.py` | `get_changed_py_files` テスト 2件 |

## テスト
42 passed (test_169: 25, test_run_manifest: 17)
