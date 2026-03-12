# 096# Phase 2 Implementation — 095# Codex Review Response 実装

## 概要

095# Codex Review v3 に対する事後レビュー (096_ph2_rev_095.md) の指摘事項を実装。
8 件の指摘 + 追加見落とし 11 件を対処。

---

## §1 実装済み変更一覧

### CRITICAL: 特徴量契約不整合の修正

| 変更 | ファイル | 内容 |
|------|---------|------|
| `build_preorder_as_features()` 新設 | feature_enricher.py | 推論時利用可能な特徴量のみで訓練用データを構築。`log_queue_wait` / `edge_bps` (post-fill) を除外 |
| 訓練関数更新 | skip_gate.py | `train_and_save_as_skip_gate()` が `build_preorder_as_features` を使用 |

### CRITICAL: 状態分離 (base_offset + boost_multiplier)

| 変更 | ファイル | 内容 |
|------|---------|------|
| `_base_offset_ratio*` 導入 | run_fill_test.py | `config.spread_offset_ratio*` への直接変更を廃止。adapter / fast_fill_defense が互いの変更を上書きする競合を排除 |
| `_boost_multiplier` 導入 | run_fill_test.py | fast_fill_defense が乗数のみを操作。base offset は adapter 管轄 |
| `_compute_maker_price` 更新 | run_fill_test.py | `_base_offset_ratio*` × `_boost_multiplier` で最終 offset を計算 |
| fast_fill_defense 完全書換え | run_fill_test.py | config 変更ゼロ。乗数の set/reset のみ |
| `_try_auto_adapt` 更新 | run_fill_test.py | combined fallback パスも `_base_offset_ratio*` を更新 |

### HIGH: Threshold warm start

| 変更 | ファイル | 内容 |
|------|---------|------|
| `warm_start_skip_gate_thresholds()` 新設 | skip_gate.py | 直近 fill records から P(AS) 履歴を復元。逆順読み + 早期終了で I/O 最適化 |
| 起動時呼び出し | run_fill_test.py | adaptive_threshold 有効時に自動復元 |

### HIGH: Adapter rolling window

| 変更 | ファイル | 内容 |
|------|---------|------|
| `adapt_recency_window` config 追加 | run_fill_test.py | 直近 N 件のみで適応判断。全履歴混合を防止 |
| YAML パース | run_fill_test.py | `adaptation.recency_window` から読み込み |
| YAML 設定 | fill_test.yaml | `recency_window: 120` |

### HIGH: Stale order side-specific

| 変更 | ファイル | 内容 |
|------|---------|------|
| Side-specific config 6 フィールド追加 | run_fill_test.py | `stale_check_after_sec_buy/sell`, `stale_drift_bps_buy/sell`, `stale_max_reprice_buy/sell` |
| ポーリングループ更新 | run_fill_test.py | 実行時に side に応じて適切なパラメータを選択 |
| YAML パース | run_fill_test.py | `stale_order` セクションから side-specific 値を読み込み |
| YAML 設定 | fill_test.yaml | buy: 15s/4bps/1回, sell: 30s/5bps/2回 |

### MEDIUM: Timeout 削減

| 変更 | ファイル | 内容 |
|------|---------|------|
| `order_timeout_sec: 300 → 90` | fill_test.yaml | stale order + reprice で早期対処するため長時間待機不要 |

### MEDIUM: Time filter 強化

| 変更 | ファイル | 内容 |
|------|---------|------|
| UTC08 (JST17) buy ブロック追加 | fill_test.yaml | -3.81bps n=15 — 統計的に確実な悪時間帯 |

---

## §2 追加見落とし修正

| # | Issue | Severity | 修正内容 |
|---|-------|----------|---------|
| 1 | 早期リターン FillRecord が config 初期値を記録 | HIGH | L1162: `_base_offset_ratio`, L1376: `effective_offset_ratio` に修正 |
| 2 | `_try_auto_adapt` combined パスが config 直変更 | HIGH | `_base_offset_ratio*` を更新する設計に統一 |
| 3 | `_boost_multiplier` unfilled 時リセット漏れ | MEDIUM | unfilled サイクルでもブーストをリセットする分岐追加 |
| 4 | スプレッドガード退避時の offset 記録不整合 | MEDIUM | 退避時 `effective_offset_ratio = 0.0` を設定 |
| 5 | `warm_start` 全 JSONL 丸読み | LOW→修正 | 逆順読み + 必要件数で早期終了 |
| 6 | `feature_enricher.py` 重複 logger.info | LOW | 後方の重複を削除 |
| 7 | `mid_at_order` 取得失敗時のサイレント pass | MEDIUM | `logger.debug` 追加 |

---

## §3 テスト更新

| テスト | 変更理由 |
|--------|---------|
| `Test049SideOffset::test_side_offset_used_in_price_calc` | `_base_offset_ratio` 参照に変更 |
| `Test050FastFillDefenseRestore::test_pre_boost_offset_field_exists` | `_boost_multiplier` + `_base_offset_ratio` に変更 |
| `Test050FastFillDefenseRestore::test_offset_restore_logic_in_run_continuous` | `_boost_multiplier` ベースに変更 |
| `Test052AdaptSellOffsetSync::test_adapt_syncs_sell_offset_in_code` | `_base_offset_ratio_sell` に変更 |
| `TestTimeFilterNoRecord::test_yaml_side_specific_time_filter` | UTC08 追加 (7h ブロック) |

**結果: 781 passed, 0 failed**

---

## §4 変更ファイル一覧

```
scripts/v460/ml/feature_enricher.py    # build_preorder_as_features, 重複ログ削除
scripts/v460/ml/skip_gate.py           # warm_start, 訓練関数更新, 逆順読み最適化
scripts/v460/run_fill_test.py          # 状態分離, side-specific stale, rolling window, 各種バグ修正
configs/v460/fill_test.yaml            # timeout, stale, recency, UTC08 buy block
tests/unit/v460/test_fill_quality.py   # 5 テスト更新
tests/unit/v460/test_regime_detector.py # 1 テスト更新
docs/096_ph2_impl.md                   # 本ドキュメント
```

---

## §5 残課題

1. **SkipGate 再訓練**: `build_preorder_as_features` で再訓練が必要。現行モデルは post-fill 特徴量に依存しており、推論時は欠落している
2. **パフォーマンス**: `_try_auto_adapt` / `_try_auto_lot_size` が毎回全 JSONL をリロード。インクリメンタル管理が理想だがスコープ外
3. **bare `except Exception: pass`**: 主要箇所にログ追加したが、残存する箇所あり（段階的対応）
