# 599# _robust_sigma 初期化 + primary skip 安全弁 hot-reload + 閾値 10→5

## 背景

596# death spiral fix (bc2691711) をデプロイ後、fill_records を分析すると:
- **buy 側の fill_record が 0 件** (新 SHA `bc269171` から)
- sell 側は全件 `preflight_insufficient` (BTC=0 のため当然)

stderr ログを調査し、**全 buy サイクルが `AttributeError: 'MakerPriceCalculator' object has no attribute '_robust_sigma'`** で即死していることを発見。

## 原因

575# で追加された `get_robust_inputs()` メソッド (`maker_price.py:435`) が
`self._robust_sigma` を asymmetric EMA の前回値として読み取るが、
`__init__` でも `__slots__` でも初期化されていなかった。

```
エラートレースバック:
orchestrator_mid_cycle.py:475 → fill_cycle_executor.py:1474 →
fill_cycle_executor.py:867 → offset_pipeline.py:97 →
multiplicative_pipeline.py:234 → maker_price.py:443
```

`_execute_and_track_cycle` の except ブロックが Exception をキャッチしてログ出力するが
fill_record を生成しないため、**サイレントに失敗していた**。

## 修正内容

### 1. `_robust_sigma` 初期化 (maker_price.py)
- `__slots__` に `"_robust_sigma"` を追加
- `__init__` で `self._robust_sigma: float = 0.0` を初期化

### 2. `skip_gate_primary_max_consecutive_skip` hot-reload 対応 (config_hot_reload.py)
- `_HOT_RELOADABLE_FIELDS` に追加
- 再起動なしで安全弁閾値を変更可能に

### 3. 安全弁閾値 10→5 (fill_test.yaml)
- stale buy model (1ヶ月未更新) による全件 skip バイアスに対して
  5 回連続 skip で安全弁発動するよう引き下げ

## 影響

- buy サイクルの offset pipeline crash が解消
- 596# safety valve が初めて機能する (buy サイクルが skip_gate まで到達)
- primary_max_consecutive_skip の runtime 調整が可能に

## 検証

- 68 テスト passed (test_585_multiplicative_pipeline + test_571_robust_stats)
- smoke test: `get_robust_inputs('buy')` → sigma=0.0100, adverse_ofi=0.0 (正常)

## 付随発見

- buy 側 skip_gate model (`skip_gate_lgbm_pnl30_buy.pkl`) が **2月24日から1ヶ月未更新**
- unified model (`skip_gate_lgbm_pnl120.pkl`) は本日再訓練済み
- buy model 再訓練 or unified fallback を別途検討
