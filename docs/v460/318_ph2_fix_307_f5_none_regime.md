# 318# 307# F5 None レジーム問題 修正

> **種別**: fix  
> **起票**: 2026-03-09  
> **起源**: 309# §1.7「307# F5: `none` レジーム問題 ⚠️ 妥当だが今回は見送り」  
> **コミット**: (本コミット)

---

## §1 問題の全体像

309# で「妥当だが見送り」とされた 307# F5 の none レジーム問題を包括的に調査・修正。

### §1.1 発見された重大バグ (F5-1)

**Passive MM バイパスが本番で死んでいた。**

[maker_price.py L1487-L1516](../scripts/v460/lib/maker_price.py) の 303# C バイパスは
`_current_regime == "none"` をチェックしていたが:

1. `FillTestRegime` enum に `"none"` 値は **存在しない** (`UNKNOWN = "unknown"`)
2. 本番では regime detector が常に存在し、warmup 中は `"unknown"` を返す
3. 従って `"none"` には **一度も一致せず**、303# C は事実上 **死んだコード**

### §1.2 none / unknown / null の意味混同

| 値 | 型 | 発生条件 | 意味 |
|---|---|---|---|
| `"unknown"` | str (Enum) | detector 観測数 < window or confidence < 閾値 | warmup 中 / 低信頼度 |
| `None` | NoneType | `_current_regime_value()` で detector 未設定 | detector 未構成 |
| `null` | JSON | FillRecord の regime フィールドが None | 旧コード / 価格取得失敗 |
| `"none"` | str | `str(r.get("regime") or "none")` の変換結果 | 分析時の表記 |

317# 観測の "none 10.4%" は主に旧レコード（regime 未実装時代）+ 価格取得失敗レコード。

---

## §2 修正内容一覧

### F5-1: Passive MM バイパス修正 (P0 — Critical Bug)

**ファイル**: `scripts/v460/lib/maker_price.py`

```python
# 旧 (死んだコード)
if _current_regime == "none":

# 新 (unknown も対象)
if _current_regime in ("none", "unknown"):
```

**影響**:
- 219# bypass（10 連続 unknown 後の強制通過）の後で Passive MM が正しく発火
- warmup 中のサイクル全般で固定 2.0 bps offset が適用される

### F5-2: Sell Skip — 既存実装の確認

**結果**: 追加実装不要

`cycle_gate_aggregator.py` Gate 1 + Gate 7 が既に存在し、YAML で有効化済み:
- `skip_buy_unknown_regime: true` (Gate 1)
- `skip_sell_unknown_regime: true` (Gate 7)
- `unknown_regime_max_consecutive: 10` (219# bypass)

これらは `regime == "unknown"` を正しくチェックしており、本番で機能している。

### F5-3: `regime_at_order` + `regime_observation_count` 追加 (P1)

**ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/lib/fill_cycle_executor.py`

| フィールド | 型 | 意味 |
|---|---|---|
| `regime_at_order` | `str \| None` | 発注時（pricing 時）のレジーム値 |
| `regime_observation_count` | `int \| None` | detector の蓄積観測数 |

**用途**:
- `regime_at_order` vs `regime` の乖離で pricing/post-cycle のミスマッチを検出
- `observation_count < 20` (= window) → warmup、`>= 20` → mature unknown

**取得タイミング**: `run_single_cycle()` の先頭でキャプチャ（order 前）

### F5-4: 分析スクリプト改善 (P2)

**ファイル**: `analysis/311_observational_rerun.py`

| 改善 | 内容 |
|---|---|
| `none_regime_analysis()` | null (=None) と "unknown" を分離分析 |
| サブ分類 | warmup_unknown (obs<20) vs mature_unknown (obs>=20) |
| regime 乖離 | `regime_at_order ≠ regime` の件数検出 |
| `run_per_regime_comparison()` | `None` → `"null"` 表記に変更（`"none"` 混同を解消） |
| `extract_filled()` | `"null"` フィルタロジック追加 |

---

## §3 テスト

| テスト | 対象 | 結果 |
|---|---|---|
| `test_bypass_fires_for_unknown_regime` | F5-1 ソースコード検証 | ✅ |
| `test_fill_record_has_regime_at_order` | F5-3 フィールド存在 | ✅ |
| `test_fill_record_regime_at_order_default_none` | F5-3 デフォルト値 | ✅ |
| fill 全体テスト | 677 passed, 10 skipped | ✅ |

---

## §4 Config / YAML 更新

`configs/v460/fill_test.yaml` の `none_regime:` セクションにコメント追記:
```yaml
# 318# F5-1 修正: "unknown" (warmup/低信頼度) も対象に含める
# ※ 旧実装は "none" のみチェック → FillTestRegime に "none" 値は存在せず事実上無効だった
```

`scripts/v460/lib/fill_config.py` のフィールドコメントも同期更新。

---

## §5 関連ドキュメント

| # | 関連 |
|---|---|
| 307 | 元 Finding: F5 none レジーム問題 |
| 309 | §1.7 見送り判断 |
| 316 | §4 S-4 none sell skip 提案 (→ 既存 Gate で対応済み) |
| 303 | C: Passive MM バイパス初回実装 |
| 219 | unknown regime 連続バイパス (max_consecutive=10) |

---

## §6 今後の観測ポイント

1. **regime_at_order データ蓄積**: 新フィールドが記録され始めたら乖離率を観測
2. **warmup vs mature-unknown 比率**: observation_count でサブ分類し、warmup 問題の深刻度を定量化
3. **Passive MM 発火率**: F5-1 修正後、`[303# C] Passive MM bypass:` ログの出現頻度を監視
4. **219# bypass 後の PnL**: 10 連続 unknown 後の強制通過サイクルの PnL を観測
