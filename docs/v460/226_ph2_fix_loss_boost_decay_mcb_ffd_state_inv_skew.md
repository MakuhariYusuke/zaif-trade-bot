# 226# loss_boost指数減衰 + MCB/FFD state永続化 + inv_skew O(1) + toxic_veto修正 + halt中MCB/SAD更新

## 変更の目的
225# までの残課題 21 件を精査し、収益性・理論整合性・状態永続化・性能の 4 軸で 6 件を実装。

## 変更内容

### T1: loss_boost 指数減衰 (Avellaneda-Stoikov 理論)
- **ファイル**: `scripts/v460/lib/maker_price.py`, `scripts/v460/lib/fill_config.py`, `configs/v460/fill_test.yaml`
- **問題**: `set_loss_boost()` による offset 拡大が 1-shot (次回 `compute()` で即消費) → cliff-edge で流動性喪失
- **対策**: 指数減衰 `mult(t) = 1 + (M-1)·exp(-t/τ)` に変換。τ = 300s (5 分で 63% 減衰)
- **根拠**: AS 理論の在庫ペナルティは連続指数形式。Guéant-Lehalle-Fernandez-Tapia (2013) のリスク調整も指数的減衰
- `_loss_boost_set_time` スロット追加、`compute()` でリアルタイム減衰計算、1.01 以下で自動リセット

### S5: halt 中 MCB/SAD 更新継続
- **ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`
- **問題**: DD halt 中に MCB/SAD へのデータ供給が停止 → halt 解除時にσが stale → 誤判定
- **対策**: halt ブロック内 `continue` 前に MCB feed_price() / SAD feed_spread() を追加

### P5: inv_skew O(1) 化
- **ファイル**: `scripts/v460/lib/maker_price.py`
- **問題**: `update_inventory()` で毎回 `sum()` O(n) スキャン (n=最大 200)
- **対策**: `_inv_buy_count` インクリメンタルカウンタでデック eviction を追跡、O(1) 演算

### S2: toxic_veto 三重発火修正
- **ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`
- **問題**: `balance_forced_halt_block` → `continue` パスで toxic_veto カウンタが減算されない → 無限ループ
- **対策**: continue 前に veto カウンタ decrement を追加

### #4-2: MCB change_history 永続化
- **ファイル**: `scripts/v460/lib/micro_circuit_breaker.py`
- **問題**: `export_state()` / `import_state()` が `_change_history_5m/15m/1h` を含まない → 再起動時に MCB レベル誤判定
- **対策**: 3 つの deque を export/import に追加

### #2-1: FFD hot-reload state 保全
- **ファイル**: `scripts/v460/lib/fast_fill_defense.py`, `scripts/v460/run_fill_test.py`
- **問題**: `_rebuild_fast_fill_defense()` で FFD インスタンス再構築時に boost 状態が消失
- **対策**: `export_state()` / `import_state()` 新設、rebuild 時に状態を保全

## テスト
- `tests/unit/v460/test_226_loss_boost_decay_inv_skew_state.py`: 30 テスト新規追加
- 全 3046 v460 テスト PASS

## 影響範囲
| ファイル | 変更行数 |
|---|---|
| `scripts/v460/lib/maker_price.py` | +52 / -8 |
| `scripts/v460/lib/fill_config.py` | +7 / -0 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +18 / -2 |
| `scripts/v460/lib/micro_circuit_breaker.py` | +12 / -0 |
| `scripts/v460/lib/fast_fill_defense.py` | +22 / -0 |
| `scripts/v460/run_fill_test.py` | +6 / -2 |
| `configs/v460/fill_test.yaml` | +2 / -0 |
| `tests/unit/v460/test_226_*` | +320 / -0 |

## コミット
- SHA: `43b09080e`
- メッセージ: `fix: 226# loss_boost指数減衰 + MCB/FFD state永続化 + inv_skew O(1) + toxic_veto修正 + halt中MCB/SAD更新`
