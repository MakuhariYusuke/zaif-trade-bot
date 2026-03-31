# 669# 1mBTC Cap 実装: restore パス max_lot クランプ + YAML 設定

## 概要
667# / 668# のレビューを検証し、両者が見落としていた盲点を修正して 1mBTC Cap を実装した。

## 667# / 668# レビュー検証結果

### 正確な主張 (全て ✅)
| 主張 | 検証 |
|------|------|
| 667#: BALS `capital_fraction` は downward clamp にならない | ✅ `_check_buy()` L284 の `current_lot < max_lot` が FALSE なら 476# 未到達 |
| 667#: `max_lot: 0.001` で 476# 拡大パスを遮断 | ✅ 厳密 `<` 比較で `0.001 < 0.001 = False` |
| 667#: `dust_sweep` が全 BTC を売る | ✅ L315 で `btc_free` 全額を `_current_lot` に代入 |
| 668#: sell 側に拡大ロジック不在 — 非対称は仕様 | ✅ `_check_sell` は shrink のみ |
| 668#: JPY Reserve 代替案の技術的実現性 | ✅ `jpy_free` は 476# ブロック内でアクセス可能 |

### 両者が見落としていた盲点 (本セッションで発見)

#### 盲点 1 (CRITICAL): `_try_lot_restore()` / `restore_lot_on_success()` が `max_lot` を無視

**攻撃シナリオ:**
1. `max_lot=0.005` → 476# が `_current_lot=0.002` に拡大
2. 残高不足 → `_apply_lot_shrink()` → `_pre_shrink_lot=0.002` を記憶
3. `max_lot: 0.001` を hot-reload
4. 残高回復 → `_try_lot_restore()` が `_current_lot = 0.002` に復元、**max_lot=0.001 を完全バイパス**

同様に `balance_shrink_active=True` の場合、fill 成功で `restore_lot_on_success()` が旧値に復元。

#### 盲点 2: 連続同方向注文で在庫 2mBTC に到達する可能性

- `smart_side_max_consecutive: 2` (デフォルト) → 連続 buy 2 回可能
- `ranging_buy_priority_max_consecutive: 0` (661# で無効化済み) → ranging buy bias は無し
- 最悪ケース: buy 0.001 × 2 = BTC 0.002、JPY ≈ 200 → PI 再発

ただし以下の理由で実運用リスクは限定的:
- `smart_side_enabled: false` が現行設定 → 連続 buy は SideSelector のデフォルト交互で抑制
- Microprice override も `microprice_side_enabled: false`
- 最も起きやすいのは 1 buy + 1 sell の交互パターン

#### 盲点 3: 668# の代替案 (JPY Reserve / Edge-Gated) も BALS と同じ構造的限界

668# は 667# の「BALS は downward clamp にならない」を受容しつつ、Section 4 で JPY Reserve を提案しているが、
これも 476# expansion ブロック内でしか効かず、`current_lot` が既に高値で残高十分なら 476# ブロック未到達で無効。

## 実装内容

### 1. `balance_checker.py`: restore パスに max_lot クランプ (盲点 1 修正)

**`_try_lot_restore()`** — 復元先が `config.max_lot` を超える場合はクランプ:
```python
_max_lot = self._config.max_lot
restored = self._pre_shrink_lot
if _max_lot > 0 and restored > _max_lot:
    restored = _max_lot
self._current_lot = restored
```

**`restore_lot_on_success()`** — 同様のクランプを追加:
```python
_max_lot = self._config.max_lot
restored = self._pre_shrink_lot
if _max_lot > 0 and restored > _max_lot:
    restored = _max_lot
self._current_lot = restored
```

`max_lot=0` (無効) の場合はクランプなし。既存動作を維持。

### 2. `fill_test.yaml`: `max_lot: 0.005 → 0.001`

`lot_sizing.max_lot` を 0.001 に変更。hot-reload 対象のため YAML 変更のみで反映。

### 3. テスト追加

`test_669_restore_max_lot_clamp.py` — 10 テストケース:
- `_try_lot_restore`: 5 テスト (クランプ, 範囲内, max_lot=0, ログ, hot-reload シナリオ)
- `restore_lot_on_success`: 5 テスト (クランプ, 範囲内, max_lot=0, hot-reload, 非 active)

## 期待される効果

| 指標 | 変更前 | 期待 |
|------|--------|------|
| `preflight_insufficient` | 45% | **< 10%** |
| `inventory_deadlock` 発火 | 5回/8h | **< 1回/8h** |
| fill rate | 7.4% | 10-12% |
| buy/sell 両側 viable 比率 | ~50% | **> 85%** |
| 注文量 | 1-2 mBTC (不安定) | **常時 1 mBTC** |

## 運用注意事項

1. **初回はボット再起動推奨**: hot-reload で `max_lot` は反映されるが、`_pre_shrink_lot` に旧値 (0.002) が残っている可能性。今回の修正で restore パスはクランプされるが、クリーンな初期化には再起動が最も安全。
2. **モニタリング指標**: PI 比率、Deadlock Escape 発火頻度、fill rate、pnl30 の順で確認。
3. **ロールバック**: `max_lot: 0.005` に戻すだけで即時復帰可能。
