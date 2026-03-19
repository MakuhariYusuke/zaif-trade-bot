# 498# Hot-Reload 横展開: フィールドカバレッジ拡大

## 背景

497# の調査で、491# の `offset_ceiling_ratio_buy: 0.25` 変更が hot-reload リスト上にあったにもかかわらず、プロセス再起動タイミングの問題で実際には適用されなかった事象が判明。
これを受けて、FillTestConfig 全 457 フィールドのうち hot-reload 対象外の 109 フィールドを体系的に分類し、安全に追加可能な 61 フィールドを横展開した。

## 変更サマリ

| 項目 | Before | After |
|------|--------|-------|
| `_HOT_RELOADABLE_FIELDS` 数 | 348 | **409** |
| 追加フィールド | - | 61 |
| 意図的除外 | - | 48 |

## 追加した 61 フィールド（カテゴリ別）

### micro_timeout_* (6 fields)
ランタイムで単純参照。コンポーネント再構築不要。
- `micro_timeout_wait_sec`, `micro_timeout_max_requote`, `micro_timeout_spread_threshold_pct`
- `micro_timeout_vol_multiplier`, `micro_timeout_regime_sensitivity`, `micro_timeout_min_wait_sec`

### recovery_skew_* (2 fields)
- `recovery_skew_buy_multiplier`, `recovery_skew_sell_multiplier`

### cross_venue_* (14 fields)
`cross_venue_ws_url` と `cross_venue_reconnect_interval_sec` は init-time (WebSocket 接続パラメータ) のため除外。
- `cross_venue_enabled`, `cross_venue_lag_threshold_ms`, `cross_venue_lag_max_ms`
- `cross_venue_alpha`, `cross_venue_lag_alpha`, `cross_venue_stale_ms`
- `cross_venue_weight`, `cross_venue_max_weight`, `cross_venue_microprice_blend`
- `cross_venue_spread_factor`, `cross_venue_lead_lag_window`, `cross_venue_correlation_threshold`
- `cross_venue_min_samples`, `cross_venue_regime_adaptive`

### macro_regime_* (3 fields)
- `macro_regime_crisis_dd_threshold`, `macro_regime_crisis_vol_multiplier`, `macro_regime_stressed_dd_threshold`

### regime_* (9 fields)
- `regime_vol_high_threshold`, `regime_vol_extreme_threshold`
- `regime_trend_strong_threshold`, `regime_trend_extreme_threshold`
- `regime_ranging_max_trend`, `regime_ranging_max_vol`
- `regime_momentum_alpha`, `regime_adx_period`, `regime_adx_threshold`

### microprice_* (4 fields)
- `microprice_alpha`, `microprice_weight`, `microprice_max_adjustment`, `microprice_enabled`

### sell/unknown offset boost (2 fields)
- `sell_offset_boost`, `unknown_offset_boost`

### ranging_sell_* (2 fields)
- `ranging_sell_offset_floor`, `ranging_sell_vol_cap`

### vpin_vol_sync_* (3 fields)
- `vpin_vol_sync_enabled`, `vpin_vol_sync_lookback`, `vpin_vol_sync_threshold`

### bayesian_regime_* (3 fields)
- `bayesian_regime_prior_weight`, `bayesian_regime_transition_smoothing`, `bayesian_regime_min_confidence`

### glft_dynamic_k_* (2 fields)
- `glft_dynamic_k_min`, `glft_dynamic_k_max`

### sigma_clustering (5 fields)
`sigma_clustering_enabled` はコンポーネント再構築が必要なため除外。
- `sigma_clustering_n_clusters`, `sigma_clustering_lookback`
- `sigma_clustering_vol_threshold`, `sigma_clustering_trend_threshold`, `sigma_clustering_update_interval`

### hm_*/halt (5 fields)
- `hm_restart_threshold`, `hm_restart_cooldown`, `hm_kill_switch_threshold`
- `halt_price_change_threshold`, `halt_recovery_wait_sec`

## 意図的に除外した 48 フィールド

以下のカテゴリは init-time のみ使用、またはインフラ設定のため hot-reload 対象外:

| カテゴリ | 主なフィールド | 理由 |
|----------|---------------|------|
| WebSocket / API | `cross_venue_ws_url`, `cross_venue_reconnect_interval_sec`, `api_request_timeout` 等 | 接続パラメータ。変更時は再起動必須 |
| ロギング / パス | `log_level`, `log_dir`, `model_dir`, `skip_gate_model_path*` 等 | ファイルハンドル/モデルロード |
| circuit_breaker | `cb_*` (12 fields) | SafeCircuitBreaker の __init__ で設定 |
| sigma_clustering_enabled | 1 field | コンポーネント有効/無効切替はビルドに関わる |
| 構造/初期化 | `version`, `exchange`, `pair`, `fill_test_mode` 等 | アプリケーション同一性に関わる |

## テスト修正

- `test_yaml_micro_timeout_defaults_disabled`: 496# で変更された YAML 値（`wait_sec: 15.0`, `max_requote: 4`）に assertion を合わせた

## コミット

- `30e1c1f9e`: 498# hot-reload横展開: 61フィールド追加 (348→409) + micro_timeoutテスト修正
