"""
FillTestConfig — fill_test 設定データクラス + サイクル内部データクラス.

119# God Object 分割: run_fill_test.py から設定定義を分離.
設定の構造 (FillTestConfig) と YAML→dataclass マッピング (from_yaml) を管理.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class FillTestConfig:
    """Fill test runner の設定.

    優先順位: CLI引数 > YAML > dataclass defaults.

    ╔══════════════════════════════════════════════════════════════╗
    ║  ⚠ GOD OBJECT 化 禁止 — AI コーディングエージェント向け警告  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  from_yaml() は 163# で 479→139 行に分割済み。             ║
    ║  セクション別 @staticmethod パーサー構造:                   ║
    ║    _parse_trading_features()  — 取引パラメータ              ║
    ║    _parse_skip_gate_section() — SkipGate 設定               ║
    ║    _parse_stale_vg_section()  — Stale/VG パラメータ         ║
    ║    _parse_stopgap_section()   — 止血 (Loss Control)         ║
    ║    _parse_infra_section()     — インフラ設定                ║
    ║  新セクション追加時は新しい _parse_*() @staticmethod を作成 ║
    ║  し from_yaml() から呼び出すこと。from_yaml() に直接書かない║
    ║  from_yaml() 行数上限: 150 行。                            ║
    ║  フィールド追加は適切なセクションパーサーに追加。           ║
    ║  クラス全体の行数上限: 1000 行。超過時はモジュール分割。    ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    symbol: str = "btc_jpy"
    order_quantity: float = 0.001  # 初期ロット (Coincheck BTC 最小)
    cycle_interval_sec: float = 120.0  # サイクル間隔
    order_timeout_sec: float = 90.0  # 注文タイムアウト (096# 300→90)
    order_timeout_sec_sell: float | None = None  # 155# S-3: sell 専用 timeout (None=共通値)
    poll_interval_sec: float = 5.0  # ポーリング間隔
    post_fill_wait_sec: float = 30.0  # 約定後 PnL 計測待ち
    post_fill_wait_sec_sell: float | None = None  # 168# §4.1 #1: sell 専用 PnL 計測待ち (None=共通値)
    results_dir: str = "results/v460/fill_test"
    # 044# 連続 preflight 失敗上限 (buy/sell 両方不足で無限スキップ防止)
    max_preflight_skip: int = 10
    # 開始サイド: JPY 残高不足時は "sell" で開始すると自己資金循環できる
    start_side: str = "buy"
    # CM-1: スプレッド比例オフセット (post_only リジェクト防止)
    spread_offset_ratio: float = 0.05  # 031# 0.2→0.05: AS低減のため保守化
    min_offset_jpy: float = 1.0  # 最小オフセット (JPY)
    # CM-2: 注文失敗リトライ
    max_order_retries: int = 2  # 084# 1→2: api_error 28% 対策 (計3回試行)
    retry_delay_sec: float = 2.0  # リトライ初回間隔 (指数バックオフ適用)
    # CM-3: AS 判定デッドゾーン (bps)
    as_deadzone_bps: float = 2.5  # 052# ±2.5 bps 以内の逆行は AS と判定しない
    # 031# スプレッドフィルター
    min_spread_jpy: float = 0.0  # 0 = フィルタなし
    # 保存
    batch_size: int = 10  # バッチ保存のサイクル数
    batch_flush_interval_sec: float = 600.0  # 079# 時間ベース定期flush (秒)
    max_save_retries: int = 3  # 保存リトライ上限
    save_fail_threshold: int = 3  # 緊急ダンプ発動の連続失敗回数
    # 158# P1-5: A/B テスト基盤 — variant 識別子 (fill_records に記録)
    ab_test_variant: str = ""  # 空="テストなし", 例: "sell_offset_015", "rescue_enabled"
    # ログ
    progress_log_interval: int = 50  # 進捗ログの出力間隔 (サイクル数)
    heartbeat_interval_sec: float = 900.0  # 079# heartbeat 間隔 (秒, time_filter 抑制中)
    log_max_bytes: int = 10 * 1024 * 1024  # ログファイル上限 (10 MB)
    log_backup_count: int = 5  # ログバックアップ世代数
    # 方策 A: パラメータ適応
    enable_auto_adapt: bool = False
    adapt_interval_cycles: int = 50
    min_adapt_samples: int = 50  # 方策 A/B 発動の最小サンプル数
    # 方策 B: 動的ロットサイジング
    enable_dynamic_lot: bool = False
    max_lot: float = 0.005
    lot_adapt_interval_cycles: int = 50
    recent_pnl_window: int = 50  # 方策 B 直近 PnL 計算ウィンドウ
    # 151# P3-03: AS 確率連動ロットサイジング (confidence_lot)
    enable_confidence_lot: bool = False
    confidence_lot_scale: float = 1.0       # AS prob → lot 縮小の傾斜
    confidence_lot_floor: float = 0.3       # lot 倍率の下限 [0, 1]
    confidence_lot_mode: str = "as"         # "as" only ("pnl" は凍結)
    # 143# R-1b: レジーム別ロット倍率 (regime_name -> multiplier)
    # high_vol: 0.7 (リスク縮小), trending: 1.2 (トレンド追従), ranging: 1.0 (デフォルト)
    regime_lot_multipliers: dict[str, float] = field(default_factory=dict)
    # 144# R-1c: レジーム別 reprice 上限調整 (regime_name -> int オフセット)
    # high_vol: +1 (乗遅れリスクあるが復帰期待), trending: +2 (積極的reprice), ranging: 0 (デフォルト)
    regime_reprice_adjustments: dict[str, int] = field(default_factory=dict)
    # 144# R-1d: レジーム別 timeout 倍率 (regime_name -> float multiplier)
    # high_vol: 0.7 (早めに撤退), trending: 1.3 (トレンドに乗るため待機), ranging: 1.0
    regime_timeout_multipliers: dict[str, float] = field(default_factory=dict)
    # レジーム検知 (035# §4)
    enable_regime: bool = True
    regime_window: int = 20
    regime_trend_threshold_pct: float = 0.5
    regime_high_vol_multiplier: float = 2.0
    regime_hysteresis_count: int = 3
    regime_min_confidence: float = 0.3  # 052# 0.4→0.3
    # 052#: トレンディング時のオフセットブースト (PnL -1.2bps)
    regime_trending_offset_boost: float = 1.5  # トレンディング検出時に offset × 1.5
    # 157# §19: trending offset boost の buy/sell 非対称化
    # trending_up 時の buy は有利方向取引 → boost 不要 (1.0)
    # trending_up 時の sell は逆方向取引 → 通常 boost (1.5)
    regime_trending_offset_boost_buy: float | None = None   # None=共通値使用
    regime_trending_offset_boost_sell: float | None = None  # None=共通値使用
    # 176# B: 方向×サイド別 offset boost (skip_sell_trending=false と併用)
    # trending_up 時: buy は順張り → offset 縮小 (0.7) で積極約定
    # trending_up 時: sell は逆張り → offset 拡大 (1.8) で保守的利確
    # trending_down 時: 逆 (sell 順張り→縮小、buy 逆張り→拡大)
    # None = regime_trending_offset_boost_buy/sell にフォールバック
    trending_up_buy_offset_boost: float | None = None
    trending_up_sell_offset_boost: float | None = None
    trending_down_buy_offset_boost: float | None = None
    trending_down_sell_offset_boost: float | None = None
    # 143# R-1a: レジーム別 offset 調整
    regime_high_vol_offset_boost: float = 1.2   # high_vol 時に offset × 1.2 (+20% 拡張)
    regime_ranging_offset_discount: float = 1.0 # ranging 時に offset × N (1.0=無効, <1.0で縮小)
    # 168# §9.10: 低ボラティリティ offset boost (time_filter 根本対策)
    # vol_ratio < threshold 時に offset を拡大し、低ボラ環境での過剰アグレッシブ発注を抑制
    low_vol_offset_boost_enabled: bool = False
    low_vol_offset_boost: float = 1.4   # 低 vol 時の offset 倍率
    low_vol_threshold: float = 0.75     # vol_ratio がこの値未満で発動 (168# 0.70→0.75: order/fill遅延マージン)
    # 169# B1': ranging_buy at low_vol ハードスキップ (Gemini 10.2-D「休むも相場」)
    # ranging レジーム + buy + vol_ratio < low_vol_threshold → 完全スキップ
    # ranging_buy が全損失の 69% を占める根本対策。offset 調整より clean
    skip_ranging_buy_low_vol: bool = False
    # 041# 時間帯フィルター (AS 高リスク時間帯のスキップ)
    enable_time_filter: bool = False
    skip_utc_hours: list[int] | None = None
    # 073# side 別時間帯フィルター: 指定時は side 固有リストを優先
    skip_utc_hours_buy: list[int] | None = None
    skip_utc_hours_sell: list[int] | None = None
    # 安全設計 (000# §3.9)
    loss_cap_jpy: float = 10_000.0
    loss_cap_warning_ratio: float = 0.7
    # 041# 動的 loss_cap: API 残高から算出
    loss_cap_auto: bool = False
    loss_cap_ratio: float = 0.05  # 残高の 5% をキャップ (hard)
    # 046# soft/hard 二段 loss_cap: soft 超過でロット半減、hard 超過で SAFE_STOP
    soft_loss_cap_ratio: float = 0.02  # 残高の 2% で soft cap (ロット半減)
    # 168# §4.1 #3: 日次ドローダウンガード (UTC 日単位の cumPnL 制限)
    daily_drawdown_enabled: bool = False        # True で日次 PnL 監視有効
    daily_drawdown_hard_limit_bps: float = -50.0  # この bps 以下でサイクル停止
    daily_drawdown_soft_limit_bps: float = -30.0  # この bps 以下でロット半減
    # 049# E3 サンプリング: 全約定ではなくサンプリングで multi-timeframe 計測
    e3_sampling_ratio: float = 1.0  # 0.0-1.0, 1.0=全約定, 0.33=1/3 のみ
    # 049# side 別 offset: buy/sell で独立に offset を設定
    spread_offset_ratio_buy: float | None = None   # None = 共通 offset を使用
    spread_offset_ratio_sell: float | None = None   # None = 共通 offset を使用
    # 049# 即約定防御: queue_wait が閾値以下で負エッジの場合に保守化
    fast_fill_defense_enabled: bool = False
    fast_fill_threshold_sec: float = 5.0   # この秒数以下で「速い約定」と判定 (共通)
    fast_fill_threshold_sec_buy: float | None = None   # 093# buy 側閾値 (None=共通値)
    fast_fill_threshold_sec_sell: float | None = None  # 093# sell 側閾値 (None=共通値)
    fast_fill_offset_boost: float = 2.0    # 防御時の offset 倍率 (共通)
    fast_fill_offset_boost_buy: float | None = None    # 093# buy 側倍率 (None=共通値)
    fast_fill_offset_boost_sell: float | None = None   # 093# sell 側倍率 (None=共通値)
    # 054# S1: Orderbook Imbalance ベース AS 予測フィルター
    imbalance_enabled: bool = False
    imbalance_depth: int = 5               # 板深さ (上位 N 段)
    imbalance_threshold: float = 0.3       # 偏り判定閾値 (|imbalance| > threshold)
    imbalance_offset_boost: float = 1.5    # AS リスク時の offset 倍率
    imbalance_skip_threshold: float = 0.7  # この閾値以上なら注文スキップ
    # 054# S2: Smart Side Selection (機械的交互 → 条件付き)
    smart_side_enabled: bool = False
    smart_side_mode: str = "suppress"      # suppress / follow
    smart_side_max_consecutive: int = 2    # 片側蓄積防止 (000# §3.3)
    # 054# S3: テール損失カット (post-fill早期監視)
    early_exit_enabled: bool = False
    early_exit_threshold_bps: float = 5.0  # 損失閾値 (bps)
    early_exit_monitor_interval_sec: float = 5.0  # 監視刻み (秒)
    early_exit_rapid_interval_sec: float = 10.0   # rapid exit 時 cycle interval
    # 054# S4: Spread-Responsive Offset (スプレッド適応型オフセット)
    spread_adaptive_enabled: bool = False
    narrow_spread_bps: float = 10.0        # 狭スプレッド閾値 (bps)
    narrow_spread_boost: float = 2.0       # 狭い時の offset 倍率 (共通)
    narrow_spread_boost_buy: float | None = None   # 093# buy 側 boost (None=共通値)
    narrow_spread_boost_sell: float | None = None   # 093# sell 側 boost (None=共通値)
    wide_spread_bps: float = 25.0          # 広スプレッド閾値 (bps)
    wide_spread_ratio: float = 0.5         # 広い時の offset 割引
    # 062# S5: SkipGate ML フィルター (AS 分類器ベースの注文スキップ)
    skip_gate_enabled: bool = False
    # 118# A3: side 別有効/無効 (sell 逆選別対策)
    skip_gate_buy_enabled: bool = True
    skip_gate_sell_enabled: bool = True
    skip_gate_mode: str = "as"             # "pnl" or "as" (061# AS 分類器推奨)
    skip_gate_model_path: str = "models/v460/skip_gate_as.pkl"  # モデルファイル
    # 141# P1-01: side 別モデルパス (None=統一モデルにフォールバック)
    skip_gate_model_path_buy: str | None = None
    skip_gate_model_path_sell: str | None = None
    # 188# C-1: ev_weighted SkipGate — 両 horizon モデルで統合判定
    # 副 horizon モデルパス (pnl120→buy用, pnl30→sell用). None=単一モデル判定
    skip_gate_model_path_buy_long: str | None = None   # buy の pnl120 (長期) モデル
    skip_gate_model_path_sell_short: str | None = None  # sell の pnl30 (短期) モデル
    skip_gate_ev_weighted_enabled: bool = False  # True: 両 horizon ev_weighted 判定
    skip_gate_ev_w30: float = 0.4   # ev_weighted の pnl30 重み
    skip_gate_ev_w120: float = 0.6  # ev_weighted の pnl120 重み
    skip_gate_as_threshold: float = 0.52   # 100# AS 確率スキップ閾値 (0.65→0.52)
    skip_gate_pnl_threshold: float = 0.0   # PnL 予測スキップ閾値 (mode=pnl)
    skip_gate_max_skip_rate: float = 0.3   # 連続スキップ率上限 (安全弁)
    # 068# §3.3: side 別閾値 (None は共通 as_threshold を使用)
    skip_gate_as_threshold_buy: Optional[float] = None
    skip_gate_as_threshold_sell: Optional[float] = None
    # 072# OB 特徴量トグル (ph2 通過後に True へ)
    skip_gate_use_ob_features: bool = False
    # 088# 動的閾値較正
    skip_gate_adaptive_threshold: bool = False
    skip_gate_target_skip_rate_buy: float = 0.10
    skip_gate_target_skip_rate_sell: float = 0.20
    skip_gate_adaptive_window: int = 50
    skip_gate_adaptive_min_samples: int = 20
    skip_gate_adaptive_step: float = 0.05  # 100# 0.02→0.05
    skip_gate_adaptive_floor: float = 0.35
    skip_gate_adaptive_ceiling: float = 0.80
    # 138# P1-03: score calibration (isotonic regression)
    skip_gate_score_calibration: bool = False      # True で score 校正を有効化
    skip_gate_calibrator_path: str | None = None   # calibrator pkl パス (None=インメモリ)
    skip_gate_calibrator_min_samples: int = 30     # 校正に必要な最小サンプル数
    skip_gate_calibrator_refit_interval: int = 100 # 自動 refit 間隔 (新規レコード数)
    # 141# P1-04: regime 別 PnL 閾値オーバーライド
    skip_gate_regime_thresholds: dict[str, float] = field(default_factory=dict)
    # 158# P1-6: 時間帯別 skip_gate 閾値調整 (UTC hour → offset bps)
    # 正=厳格化 (skip 増), 負=緩和 (skip 減). PnL mode: threshold += offset, AS mode: threshold -= offset
    skip_gate_hour_offsets: dict[int, float] = field(default_factory=dict)
    # 124# Rule: unknown regime での sell スキップ
    skip_sell_unknown_regime: bool = False
    # 130# unknown regime での buy offset boost (AS 回避)
    unknown_buy_offset_boost: float = 1.0  # 1.0 = 無効, >1.0 で boost (例: 2.0 = VG相当)
    # 165# AS-R1: velocity-based sell/buy skip (SkipGate pre-ML rule)
    sell_velocity_skip_enabled: bool = False
    sell_velocity_skip_threshold_bps: float = 8.0  # price_velocity_60s > this AND sell -> skip
    buy_velocity_skip_enabled: bool = False
    buy_velocity_skip_threshold_bps: float = -8.0  # price_velocity_60s < this AND buy -> skip
    # 183# narrow spread 時の skip_gate 閾値オフセット (逆選択防御)
    # spread < narrow_spread_skip_threshold_jpy のとき threshold に加算。
    # ログ分析: spread<2kでAS32% (全体28%) → 閾値厳格化で AS fill削減
    skip_gate_narrow_spread_threshold_jpy: float = 0.0  # 0.0=無効
    skip_gate_narrow_spread_offset: float = 0.0  # 正=厳格化 (PnLモード)
    # 187# clamp YAML外部化: skip_gate offset の上下限
    skip_gate_offset_floor: float = -0.3   # 最大緩和
    skip_gate_offset_ceil: float = 0.5     # 最大厳格化
    # 156# §16: OB エラー fallback 価格の鮮度閾値 (秒)
    fallback_stale_sec: float = 120.0
    # 094# stale order 検出 & cancel-replace (価格乖離した注文を再発注)
    stale_order_enabled: bool = False
    stale_check_after_sec: float = 30.0    # 発注後この秒数以降で乖離チェック開始
    stale_drift_bps: float = 5.0           # mid price がこの bps 以上乖離したら stale
    stale_max_reprice: int = 2             # 1 サイクル内の最大再発注回数
    stale_cooldown_sec: float = 10.0       # 再発注後チェック猶予（連続 reprice 防止）
    # 096# stale order side 別パラメータ
    stale_check_after_sec_buy: float | None = None   # buy 側乖離チェック開始 (None=共通値)
    stale_check_after_sec_sell: float | None = None  # sell 側乖離チェック開始 (None=共通値)
    stale_drift_bps_buy: float | None = None         # buy 側乖離閾値 (None=共通値)
    stale_drift_bps_sell: float | None = None        # sell 側乖離閾値 (None=共通値)
    stale_max_reprice_buy: int | None = None         # buy 側最大 reprice (None=共通値)
    stale_max_reprice_sell: int | None = None        # sell 側最大 reprice (None=共通値)
    # 158# P1-2: reprice 時の offset 引き締め (1.0=変更なし, 0.85=15% 引き締め→より市場価格に近づける)
    stale_reprice_tighten: float = 1.0
    # 168# P2-C3: reprice SkipGate 閾値緩和 offset
    stale_reprice_skip_gate_offset: float = 0.0
    # 096# adapter recency window (全履歴混合を停止)
    adapt_recency_window: int = 0          # 0=全履歴, >0=直近 N clean records のみ
    # 107# Volatility Guard (リアルタイム急変検知)
    volatility_guard_enabled: bool = False
    volatility_guard_velocity_window_sec: float = 60.0
    volatility_guard_velocity_threshold_bps: float = 15.0
    volatility_guard_vpin_threshold: float = 0.70
    volatility_guard_offset_boost_factor: float = 2.0
    # 168# InvSkew/VG 競合解消: InvSkew 緩和時に VG ブースト上限を制御
    vg_inv_skew_damping_enabled: bool = False
    # 110# 086# デッドロック修正: 連続 both-filtered 上限
    max_086_consecutive_wait: int = 3      # 0 = 無制限 (旧動作), >0 で N 回超過後 alt_side 許可
    # 163# regime 連動動的ゲーティング (107# Phase 3 拡張)
    regime_adaptive_enabled: bool = False
    regime_adaptive_extra_buy: list[int] | None = None   # high_vol 時のみ追加遮断
    regime_adaptive_extra_sell: list[int] | None = None  # high_vol 時のみ追加遮断
    # 088# sell 専用ハードガード
    sell_max_spread_jpy: float = 0.0       # 0 = 無制限, >0 でスプレッド超過時 sell スキップ
    sell_offset_floor: float = 0.0         # 0 = 無制限, >0 で sell offset 最低保証
    # 173# 動的フロア: 在庫 buy 偏重時にフロアを割引 (1.0=割引なし, 0.5=半減, 0.0=フロア無効化)
    sell_offset_floor_inv_discount: float = 0.5
    # ---- 133# P0-08: 残高制約による強制 side 切替時の発注抑制 ----
    skip_balance_forced: bool = False      # True で強制切替時スキップ (平均 -1.98bps の損失回避)
    # 154# C-1/C-2: deadlock 防止 — 片側残高枯渇時は forced でも実行を許可
    balance_forced_deadlock_limit: int = 3  # 連続 forced skip が N 回超過 → 強制実行 (0=無制限)
    # 158# P1-1: balance_forced 救済モード — offset 倍増で安全にポジション解消
    balance_forced_rescue_enabled: bool = False    # True で rescue モード有効 (skip の代わりに高 offset で実行)
    balance_forced_rescue_offset_mult: float = 2.0  # rescue 時の offset 倍率 (2.0 = 通常の 2 倍)
    # ---- 133# P0-09: unknown レジームでの buy スキップ ----
    skip_buy_unknown_regime: bool = False  # True で unknown レジーム時 buy もスキップ (-1.384bps)
    # ---- 155# §9: trending レジームでの sell 抑制 ----
    skip_sell_trending: bool = False  # True で trending 時 sell をスキップ (-0.687bps)
    # 156# D-4: trending 方向別分解 — True なら trending_up のみスキップ (trending_down sell を開放)
    skip_sell_trending_up_only: bool = False
    # 158# §20-B: 連続 trending sell skip 安全弁 — N 回超過で sell を強制許可 (0=無制限)
    max_consecutive_trending_sell_skip: int = 30
    # 171# Guard Paradox 対策: 在庫偏重時に sell ガードを自動緩和
    # InvSkew imbalance > この閾値なら trending_sell_skip + sell_dynamic_kill をバイパス
    sell_guard_inv_bypass_threshold: float = 0.3  # buy偏重 0.3 以上で sell 抑制解除
    # ---- 133# P0-10: sell 動的 kill (rolling PnL ベースの自動停止) ----
    sell_dynamic_kill_enabled: bool = False  # True で sell rolling PnL 監視有効
    sell_dynamic_kill_window: int = 50       # rolling ウィンドウ (fill 数)
    sell_dynamic_kill_threshold_bps: float = -0.5  # この値以下で sell 停止
    sell_dynamic_kill_resume_window: int = 20     # 停止後、N サイクル後に再評価
    # 139# §9-#2: レジーム別閾値 (regime_name -> threshold_bps)
    sell_dynamic_kill_regime_thresholds: dict[str, float] = field(default_factory=dict)
    # ---- 157# §19: buy 動的 kill (rolling PnL ベースの自動停止 — sell との対称性) ----
    buy_dynamic_kill_enabled: bool = False   # True で buy rolling PnL 監視有効
    buy_dynamic_kill_window: int = 50        # rolling ウィンドウ (fill 数)
    buy_dynamic_kill_threshold_bps: float = -0.8  # buy は sell より寛容 (157#: 構造的に buy のほうが AS リスクが低い)
    buy_dynamic_kill_resume_window: int = 10      # 停止後、N サイクル後に再評価
    buy_dynamic_kill_regime_thresholds: dict[str, float] = field(default_factory=dict)
    # ---- 137# P1-08: spread 狭小時の「休む」判定 ----
    narrow_spread_pause_enabled: bool = False     # True で spread 狭小時にサイクルスキップ
    narrow_spread_pause_bps: float = 3.0          # spread < この bps で狭小とみなす
    narrow_spread_pause_sec: float = 5.0          # 狭小検出時のスリープ秒数
    narrow_spread_pause_max_consecutive: int = 3  # 連続スキップ上限 (超過で強行)
    # ---- 138# P1-10: preflight 失敗連続→run pause (dead-cycle 防止) ----
    # ---- 162# Inventory Skewing: 在庫偏重による非対称クオート ----
    inventory_skewing_enabled: bool = False    # True で在庫偏重 offset 補正を有効化
    inventory_skewing_window: int = 100        # 直近 N fill で在庫偏重を計算
    inventory_skewing_max_factor: float = 0.4  # 最大 offset 補正倍率 (0.4 = 40%)
    inventory_skewing_neutral_band: float = 0.1  # |imbalance| < この値なら補正なし
    preflight_pause_enabled: bool = True       # True で SAFE_STOP 前に pause を挟む
    preflight_pause_threshold: int = 5         # この回数で pause 発動 (< max_preflight_skip)
    preflight_pause_sec: float = 300.0         # pause 時のスリープ秒数
    preflight_max_pauses: int = 3              # run 内の最大 pause 回数 (超過で SAFE_STOP)
    # ---- 137# P1-11: PnL fee 控除統一 ----
    pnl_fee_deduction_enabled: bool = False   # True で PnL に fee 控除を適用
    maker_fee_bps: float = 0.0                # maker 手数料 (bps, Coincheck maker=0)
    taker_fee_bps: float = 0.0                # taker 手数料 (bps, 将来の taker 対応用)
    # ---- 102# YAML化: 散在マジックナンバーの設定外部化 ----
    max_offset_ratio: float = 0.30
    min_offset_ratio: float = 0.01
    loss_cap_update_interval: int = 50
    min_loss_cap_jpy: float = 50.0
    mid_trend_validity_sec: float = 300.0
    balance_margin_ratio: float = 1.01
    balance_shrink_consecutive: int = 3
    balance_shrink_divisor: int = 2
    skip_gate_recent_trades_limit: int = 50
    status_unknown_retry_delays: list[float] = field(default_factory=lambda: [2.0, 3.0, 5.0])
    rate_limit_min_backoff_sec: float = 5.0
    save_retry_backoff_sec: float = 0.5
    regime_warmup_multiplier: int = 3
    e3_60s_multiplier: float = 2.0
    e3_120s_multiplier: float = 4.0
    adapt_min_side_samples: int = 20
    # 121# 追加外部化パラメータ
    min_order_btc: float = 0.001           # Coincheck BTC 最小注文数量
    dust_sweep_enabled: bool = True        # 128# 端数BTC一掃: sell時にdust込みで全額売却
    lock_acquire_retries: int = 2          # lockfile 取得リトライ回数
    lock_heartbeat_period_sec: float = 60.0  # 148# lock heartbeat 更新周期 (秒)
    lock_stale_heartbeat_sec: float = 300.0  # 129#/148# lock heartbeat 陳腐化閾値 (秒, >= 3 * period)
    skip_gate_ob_depth: int = 5            # SkipGate 板情報取得深度
    retry_backoff_base: int = 2            # 発注リトライ指数バックオフ底
    soft_loss_cap_lot_divisor: int = 2     # soft_loss_cap ロット半減の除数
    file_log_level: str = "DEBUG"          # ファイルログレベル
    insufficient_funds_patterns: list[str] = field(
        default_factory=lambda: ["所持金額", "足りません"]
    )
    # ---- 158# YAML 外部化: resilience (CircuitBreaker / HealthMonitor) ----
    cb_failure_threshold: int = 5           # API 連続失敗→OPEN
    cb_recovery_timeout: float = 120.0      # OPEN→HALF_OPEN 待機 (秒)
    cb_success_threshold: int = 2           # HALF_OPEN→CLOSE 成功回数
    cb_timeout: float = 30.0               # API タイムアウト (秒)
    hm_rss_warn_mb: float = 1500.0          # RSS 警告閾値 (MB)
    hm_rss_critical_mb: float = 2500.0      # RSS 緊急閾値 (MB)
    hm_disk_free_warn_gb: float = 2.0       # ディスク空き警告 (GB)
    hm_gc_interval_cycles: int = 100        # GC 実行間隔 (サイクル数)
    hm_check_interval_sec: float = 300.0    # ヘルスチェック間隔 (秒)
    # ---- 158# YAML 外部化: tuning 追加 ----
    hot_reload_check_interval_sec: float = 120.0   # SkipGate モデル差替チェック間隔
    records_cache_ttl_sec: float = 10.0             # 適応エンジン キャッシュ TTL
    trades_recorder_fetch_limit: int = 100          # TradesRecorder 取得件数
    balance_freeze_cycles: int = 3                  # 残高不足 side の凍結サイクル数

    def __post_init__(self) -> None:
        """103# バリデーション: YAML 誤設定による本番クラッシュ防止."""
        if self.balance_shrink_divisor < 1:
            raise ValueError(
                f"balance_shrink_divisor must be >= 1, got {self.balance_shrink_divisor}"
            )
        if self.max_offset_ratio <= self.min_offset_ratio:
            raise ValueError(
                f"max_offset_ratio ({self.max_offset_ratio}) must be > "
                f"min_offset_ratio ({self.min_offset_ratio})"
            )
        # 139# §8-#6: 新規パラメータの境界バリデーション
        if self.preflight_pause_threshold < 1:
            raise ValueError(
                f"preflight_pause_threshold must be >= 1, got {self.preflight_pause_threshold}"
            )
        if self.preflight_max_pauses < 0:
            raise ValueError(
                f"preflight_max_pauses must be >= 0, got {self.preflight_max_pauses}"
            )
        if self.preflight_pause_sec < 0:
            raise ValueError(
                f"preflight_pause_sec must be >= 0, got {self.preflight_pause_sec}"
            )
        if self.skip_gate_calibrator_min_samples < 1:
            raise ValueError(
                f"skip_gate_calibrator_min_samples must be >= 1, got {self.skip_gate_calibrator_min_samples}"
            )
        if self.skip_gate_calibrator_refit_interval < 1:
            raise ValueError(
                f"skip_gate_calibrator_refit_interval must be >= 1, got {self.skip_gate_calibrator_refit_interval}"
            )
        # 145# §8-#6: レジーム設定の値域バリデーション
        for k, v in self.regime_timeout_multipliers.items():
            if v <= 0:
                raise ValueError(
                    f"regime_timeout_multipliers['{k}'] must be > 0, got {v}"
                )
        for k, v in self.regime_lot_multipliers.items():
            if v <= 0:
                raise ValueError(
                    f"regime_lot_multipliers['{k}'] must be > 0, got {v}"
                )
        _MAX_REPRICE_ADJ = 10
        for k, v in self.regime_reprice_adjustments.items():
            if abs(v) > _MAX_REPRICE_ADJ:
                raise ValueError(
                    f"regime_reprice_adjustments['{k}'] abs value must be <= {_MAX_REPRICE_ADJ}, got {v}"
                )
        # 151# P3-03: confidence_lot バリデーション (§10 #2 対応)
        if not (0.0 <= self.confidence_lot_floor <= 1.0):
            raise ValueError(
                f"confidence_lot_floor must be in [0, 1], got {self.confidence_lot_floor}"
            )
        if self.confidence_lot_scale < 0:
            raise ValueError(
                f"confidence_lot_scale must be >= 0, got {self.confidence_lot_scale}"
            )
        if self.confidence_lot_mode not in ("as", "pnl"):
            raise ValueError(
                f"confidence_lot_mode must be 'as' or 'pnl', got '{self.confidence_lot_mode}'"
            )
        # §13 #1: enable=True + mode!=as は設定乖離 → fail-fast
        if self.enable_confidence_lot and self.confidence_lot_mode != "as":
            raise ValueError(
                f"confidence_lot_mode must be 'as' when enabled, "
                f"got '{self.confidence_lot_mode}' (mode='pnl' is frozen)"
            )
        # 173# sell_guard_inv_bypass_threshold バリデーション
        if not (0.0 <= self.sell_guard_inv_bypass_threshold <= 1.0):
            raise ValueError(
                f"sell_guard_inv_bypass_threshold must be in [0, 1], "
                f"got {self.sell_guard_inv_bypass_threshold}"
            )
        # 174# daily_drawdown soft/hard limit 順序バリデーション
        if self.daily_drawdown_soft_limit_bps < self.daily_drawdown_hard_limit_bps:
            raise ValueError(
                f"daily_drawdown_soft_limit_bps ({self.daily_drawdown_soft_limit_bps}) "
                f"must be >= daily_drawdown_hard_limit_bps ({self.daily_drawdown_hard_limit_bps}). "
                f"soft=-30, hard=-50 のように soft は hard より緩い値であること"
            )
        # 174# inventory_skewing_window / sell_dynamic_kill_window 境界
        if self.inventory_skewing_window < 0:
            raise ValueError(
                f"inventory_skewing_window must be >= 0, got {self.inventory_skewing_window}"
            )
        if self.sell_dynamic_kill_window < 1:
            raise ValueError(
                f"sell_dynamic_kill_window must be >= 1, got {self.sell_dynamic_kill_window}"
            )
        if self.buy_dynamic_kill_window < 1:
            raise ValueError(
                f"buy_dynamic_kill_window must be >= 1, got {self.buy_dynamic_kill_window}"
            )
        # 174# sell_offset_floor_inv_discount 値域
        if not (0.0 <= self.sell_offset_floor_inv_discount <= 1.0):
            raise ValueError(
                f"sell_offset_floor_inv_discount must be in [0, 1], "
                f"got {self.sell_offset_floor_inv_discount}"
            )


    # ================================================================
    # from_yaml() セクションパーサー (163# God Object 分割)
    # WARNING: 下記メソッドは from_yaml() から呼ばれる補助関数。
    #          新設定キーは対応するセクションパーサーに追加すること。
    # ================================================================

    @staticmethod
    def _parse_trading_features(yaml_cfg: dict) -> dict:
        """049#/054# E3/side_offset/FFD/imbalance/smart_side/early_exit/spread_adaptive."""
        kwargs: dict = {}
        # 049# E3 サンプリング
        e3 = yaml_cfg.get("e3", {})
        if "sampling_ratio" in e3:
            kwargs["e3_sampling_ratio"] = e3["sampling_ratio"]

        # 049# side 別 offset
        side_offset = yaml_cfg.get("side_offset", {})
        if "buy" in side_offset:
            kwargs["spread_offset_ratio_buy"] = side_offset["buy"]
        if "sell" in side_offset:
            kwargs["spread_offset_ratio_sell"] = side_offset["sell"]

        # 049# 即約定防御
        ffd = yaml_cfg.get("fast_fill_defense", {})
        if ffd.get("enabled") is not None:
            kwargs["fast_fill_defense_enabled"] = ffd["enabled"]
        if "threshold_sec" in ffd:
            kwargs["fast_fill_threshold_sec"] = ffd["threshold_sec"]
        if "offset_boost" in ffd:
            kwargs["fast_fill_offset_boost"] = ffd["offset_boost"]
        # 093# side 別 fast_fill_defense
        if "threshold_sec_buy" in ffd:
            kwargs["fast_fill_threshold_sec_buy"] = ffd["threshold_sec_buy"]
        if "threshold_sec_sell" in ffd:
            kwargs["fast_fill_threshold_sec_sell"] = ffd["threshold_sec_sell"]
        if "offset_boost_buy" in ffd:
            kwargs["fast_fill_offset_boost_buy"] = ffd["offset_boost_buy"]
        if "offset_boost_sell" in ffd:
            kwargs["fast_fill_offset_boost_sell"] = ffd["offset_boost_sell"]

        # 054# S1: Orderbook Imbalance
        imb = yaml_cfg.get("imbalance", {})
        if imb.get("enabled") is not None:
            kwargs["imbalance_enabled"] = imb["enabled"]
        imb_map = {
            "depth": "imbalance_depth",
            "threshold": "imbalance_threshold",
            "offset_boost": "imbalance_offset_boost",
            "skip_threshold": "imbalance_skip_threshold",
        }
        for yaml_key, config_key in imb_map.items():
            if yaml_key in imb:
                kwargs[config_key] = imb[yaml_key]

        # 054# S2: Smart Side
        ss = yaml_cfg.get("smart_side", {})
        if ss.get("enabled") is not None:
            kwargs["smart_side_enabled"] = ss["enabled"]
        if "mode" in ss:
            kwargs["smart_side_mode"] = ss["mode"]
        if "max_consecutive_same" in ss:
            kwargs["smart_side_max_consecutive"] = ss["max_consecutive_same"]

        # 054# S3: Early Exit (テール損失カット)
        ee = yaml_cfg.get("early_exit", {})
        if ee.get("enabled") is not None:
            kwargs["early_exit_enabled"] = ee["enabled"]
        ee_map = {
            "threshold_bps": "early_exit_threshold_bps",
            "monitoring_interval_sec": "early_exit_monitor_interval_sec",
            "rapid_exit_interval_sec": "early_exit_rapid_interval_sec",
        }
        for yaml_key, config_key in ee_map.items():
            if yaml_key in ee:
                kwargs[config_key] = ee[yaml_key]

        # 054# S4: Spread Adaptive Offset
        sa = yaml_cfg.get("spread_adaptive", {})
        if sa.get("enabled") is not None:
            kwargs["spread_adaptive_enabled"] = sa["enabled"]
        sa_map = {
            "narrow_spread_bps": "narrow_spread_bps",
            "narrow_spread_boost": "narrow_spread_boost",
            "narrow_spread_boost_buy": "narrow_spread_boost_buy",    # 093#
            "narrow_spread_boost_sell": "narrow_spread_boost_sell",  # 093#
            "wide_spread_bps": "wide_spread_bps",
            "wide_spread_ratio": "wide_spread_ratio",
        }
        for yaml_key, config_key in sa_map.items():
            if yaml_key in sa:
                kwargs[config_key] = sa[yaml_key]

        return kwargs

    @staticmethod
    def _parse_skip_gate_section(yaml_cfg: dict) -> dict:
        """062# S5: SkipGate ML フィルター YAML マッピング."""
        kwargs: dict = {}
        # 062# S5: SkipGate ML フィルター
        sg = yaml_cfg.get("skip_gate", {})
        if sg.get("enabled") is not None:
            kwargs["skip_gate_enabled"] = sg["enabled"]
        sg_map = {
            "mode": "skip_gate_mode",
            "model_path": "skip_gate_model_path",
            # 141# P1-01: side 別モデルパス
            "model_path_buy": "skip_gate_model_path_buy",
            "model_path_sell": "skip_gate_model_path_sell",
            # 188# C-1: ev_weighted SkipGate
            "model_path_buy_long": "skip_gate_model_path_buy_long",
            "model_path_sell_short": "skip_gate_model_path_sell_short",
            "ev_weighted_enabled": "skip_gate_ev_weighted_enabled",
            "ev_w30": "skip_gate_ev_w30",
            "ev_w120": "skip_gate_ev_w120",
            "as_threshold": "skip_gate_as_threshold",
            "pnl_threshold": "skip_gate_pnl_threshold",
            "max_skip_rate": "skip_gate_max_skip_rate",
            # 118# A3: side 別有効/無効
            "buy_enabled": "skip_gate_buy_enabled",
            "sell_enabled": "skip_gate_sell_enabled",
            # 068# §3.3: side 別閾値
            "as_threshold_buy": "skip_gate_as_threshold_buy",
            "as_threshold_sell": "skip_gate_as_threshold_sell",
            # 072# OB トグル
            "use_ob_features": "skip_gate_use_ob_features",
            # 088# 動的閾値較正
            "adaptive_threshold": "skip_gate_adaptive_threshold",
            "target_skip_rate_buy": "skip_gate_target_skip_rate_buy",
            "target_skip_rate_sell": "skip_gate_target_skip_rate_sell",
            "adaptive_window": "skip_gate_adaptive_window",
            "adaptive_min_samples": "skip_gate_adaptive_min_samples",
            "adaptive_step": "skip_gate_adaptive_step",
            "adaptive_floor": "skip_gate_adaptive_floor",
            "adaptive_ceiling": "skip_gate_adaptive_ceiling",
            # 124# Rule: unknown regime sell skip
            "skip_sell_unknown_regime": "skip_sell_unknown_regime",
            # 130# unknown buy offset boost
            "unknown_buy_offset_boost": "unknown_buy_offset_boost",
            # 165# AS-R1: velocity-based skip
            "sell_velocity_skip_enabled": "sell_velocity_skip_enabled",
            "sell_velocity_skip_threshold_bps": "sell_velocity_skip_threshold_bps",
            "buy_velocity_skip_enabled": "buy_velocity_skip_enabled",
            "buy_velocity_skip_threshold_bps": "buy_velocity_skip_threshold_bps",
            # 141# P1-04: regime thresholds
            "regime_thresholds": "skip_gate_regime_thresholds",
            # 138# P1-03: score calibration
            "score_calibration": "skip_gate_score_calibration",
            "calibrator_path": "skip_gate_calibrator_path",
            "calibrator_min_samples": "skip_gate_calibrator_min_samples",
            "calibrator_refit_interval": "skip_gate_calibrator_refit_interval",
            # 183# narrow spread adverse guard
            "skip_gate_narrow_spread_threshold_jpy": "skip_gate_narrow_spread_threshold_jpy",
            "skip_gate_narrow_spread_offset": "skip_gate_narrow_spread_offset",
            # 187# clamp YAML外部化
            "offset_floor": "skip_gate_offset_floor",
            "offset_ceil": "skip_gate_offset_ceil",
        }
        for yaml_key, config_key in sg_map.items():
            if yaml_key in sg and sg[yaml_key] is not None:
                kwargs[config_key] = sg[yaml_key]

        # 158# P1-6: hour_offsets (UTC hour → offset bps)
        hour_offsets_raw = sg.get("hour_offsets", {})
        if hour_offsets_raw:
            kwargs["skip_gate_hour_offsets"] = {
                int(k): float(v) for k, v in hour_offsets_raw.items()
            }

        return kwargs

    @staticmethod
    def _parse_stale_vg_section(yaml_cfg: dict) -> dict:
        """094#/096#/107# Stale order + VG + sell_guard."""
        kwargs: dict = {}
        # 094# stale order 検出 & cancel-replace
        so = yaml_cfg.get("stale_order", {})
        if so.get("enabled") is not None:
            kwargs["stale_order_enabled"] = so["enabled"]
        so_map = {
            "check_after_sec": "stale_check_after_sec",
            "drift_bps": "stale_drift_bps",
            "max_reprice": "stale_max_reprice",
            "cooldown_sec": "stale_cooldown_sec",
            # 096# side-specific
            "check_after_sec_buy": "stale_check_after_sec_buy",
            "check_after_sec_sell": "stale_check_after_sec_sell",
            "drift_bps_buy": "stale_drift_bps_buy",
            "drift_bps_sell": "stale_drift_bps_sell",
            "max_reprice_buy": "stale_max_reprice_buy",
            "max_reprice_sell": "stale_max_reprice_sell",
            # 158# P1-2: reprice offset tightening
            "reprice_tighten": "stale_reprice_tighten",
            "reprice_skip_gate_offset": "stale_reprice_skip_gate_offset",
        }
        for yaml_key, config_key in so_map.items():
            if yaml_key in so:
                kwargs[config_key] = so[yaml_key]

        # 096# adaptation recency window
        adapt = yaml_cfg.get("adaptation", {})
        if adapt.get("recency_window") is not None:
            kwargs["adapt_recency_window"] = adapt["recency_window"]
        # 103# adaptation.min_samples → min_adapt_samples マッピング
        if "min_samples" in adapt:
            kwargs["min_adapt_samples"] = adapt["min_samples"]

        # 107# Volatility Guard
        vg = yaml_cfg.get("volatility_guard", {})
        if vg.get("enabled") is not None:
            kwargs["volatility_guard_enabled"] = vg["enabled"]
        vg_map = {
            "velocity_window_sec": "volatility_guard_velocity_window_sec",
            "velocity_threshold_bps": "volatility_guard_velocity_threshold_bps",
            "vpin_threshold": "volatility_guard_vpin_threshold",
            "offset_boost_factor": "volatility_guard_offset_boost_factor",
            "inv_skew_damping_enabled": "vg_inv_skew_damping_enabled",
        }
        for yaml_key, config_key in vg_map.items():
            if yaml_key in vg:
                kwargs[config_key] = vg[yaml_key]

        # 088# sell 専用ハードガード
        sell_guard = yaml_cfg.get("sell_guard", {})
        if sell_guard.get("max_spread_jpy") is not None:
            kwargs["sell_max_spread_jpy"] = sell_guard["max_spread_jpy"]
        if sell_guard.get("offset_floor") is not None:
            kwargs["sell_offset_floor"] = sell_guard["offset_floor"]
        # 175# sell_offset_floor_inv_discount YAML バインド
        if sell_guard.get("offset_floor_inv_discount") is not None:
            kwargs["sell_offset_floor_inv_discount"] = float(
                sell_guard["offset_floor_inv_discount"]
            )

        return kwargs

    @staticmethod
    def _parse_stopgap_section(yaml_cfg: dict) -> dict:
        """133# 止血施策 + dynamic kill + narrow spread + inventory skewing."""
        kwargs: dict = {}
        # 133# P0-08/09/10: 止血施策
        止血 = yaml_cfg.get("止血", yaml_cfg.get("loss_control", {}))
        if 止血.get("skip_balance_forced") is not None:
            kwargs["skip_balance_forced"] = 止血["skip_balance_forced"]
        # 154# C-1/C-2: deadlock 防止の連続 forced skip 上限
        if 止血.get("balance_forced_deadlock_limit") is not None:
            kwargs["balance_forced_deadlock_limit"] = 止血["balance_forced_deadlock_limit"]
        # 158# P1-1: balance_forced 救済モード
        if 止血.get("balance_forced_rescue_enabled") is not None:
            kwargs["balance_forced_rescue_enabled"] = 止血["balance_forced_rescue_enabled"]
        if 止血.get("balance_forced_rescue_offset_mult") is not None:
            kwargs["balance_forced_rescue_offset_mult"] = float(止血["balance_forced_rescue_offset_mult"])
        if 止血.get("skip_buy_unknown_regime") is not None:
            kwargs["skip_buy_unknown_regime"] = 止血["skip_buy_unknown_regime"]
        # 155# §9: trending sell 抑制
        if 止血.get("skip_sell_trending") is not None:
            kwargs["skip_sell_trending"] = 止血["skip_sell_trending"]
        # 156# D-4: trending 方向別分解
        if 止血.get("skip_sell_trending_up_only") is not None:
            kwargs["skip_sell_trending_up_only"] = 止血["skip_sell_trending_up_only"]
        # 158# §20-B: 連続 trending sell skip 安全弁
        if 止血.get("max_consecutive_trending_sell_skip") is not None:
            kwargs["max_consecutive_trending_sell_skip"] = 止血["max_consecutive_trending_sell_skip"]
        # 171# Guard Paradox 対策: 在庫偏重時の sell ガードバイパス閾値
        if 止血.get("sell_guard_inv_bypass_threshold") is not None:
            kwargs["sell_guard_inv_bypass_threshold"] = float(止血["sell_guard_inv_bypass_threshold"])
        sell_kill = 止血.get("sell_dynamic_kill", {})
        if sell_kill.get("enabled") is not None:
            kwargs["sell_dynamic_kill_enabled"] = sell_kill["enabled"]
        for yk, ck in {
            "window": "sell_dynamic_kill_window",
            "threshold_bps": "sell_dynamic_kill_threshold_bps",
            "resume_window": "sell_dynamic_kill_resume_window",
        }.items():
            if yk in sell_kill:
                kwargs[ck] = sell_kill[yk]
        # 139# §9-#2: regime_thresholds YAML 配線
        if "regime_thresholds" in sell_kill:
            kwargs["sell_dynamic_kill_regime_thresholds"] = sell_kill["regime_thresholds"]

        # 157# §19: buy 動的 kill
        buy_kill = 止血.get("buy_dynamic_kill", {})
        if buy_kill.get("enabled") is not None:
            kwargs["buy_dynamic_kill_enabled"] = buy_kill["enabled"]
        for yk, ck in {
            "window": "buy_dynamic_kill_window",
            "threshold_bps": "buy_dynamic_kill_threshold_bps",
            "resume_window": "buy_dynamic_kill_resume_window",
        }.items():
            if yk in buy_kill:
                kwargs[ck] = buy_kill[yk]
        if "regime_thresholds" in buy_kill:
            kwargs["buy_dynamic_kill_regime_thresholds"] = buy_kill["regime_thresholds"]

        # 137# P1-08: narrow spread pause
        narrow_pause = 止血.get("narrow_spread_pause", {})
        if narrow_pause.get("enabled") is not None:
            kwargs["narrow_spread_pause_enabled"] = narrow_pause["enabled"]
        for yk, ck in {
            "threshold_bps": "narrow_spread_pause_bps",
            "pause_sec": "narrow_spread_pause_sec",
            "max_consecutive": "narrow_spread_pause_max_consecutive",
        }.items():
            if yk in narrow_pause:
                kwargs[ck] = narrow_pause[yk]


        # 162# Inventory Skewing: 在庫偏重による非対称クオート
        inv_skew = 止血.get("inventory_skewing", {})
        if inv_skew.get("enabled") is not None:
            kwargs["inventory_skewing_enabled"] = inv_skew["enabled"]
        for yk, ck in {
            "window": "inventory_skewing_window",
            "max_factor": "inventory_skewing_max_factor",
            "neutral_band": "inventory_skewing_neutral_band",
        }.items():
            if yk in inv_skew:
                kwargs[ck] = inv_skew[yk]

        # 168# §4.1 #3: 日次ドローダウンガード
        dd_guard = 止血.get("daily_drawdown", {})
        if dd_guard.get("enabled") is not None:
            kwargs["daily_drawdown_enabled"] = dd_guard["enabled"]
        if "hard_limit_bps" in dd_guard:
            kwargs["daily_drawdown_hard_limit_bps"] = float(dd_guard["hard_limit_bps"])
        if "soft_limit_bps" in dd_guard:
            kwargs["daily_drawdown_soft_limit_bps"] = float(dd_guard["soft_limit_bps"])

        return kwargs

    @staticmethod
    def _parse_infra_section(yaml_cfg: dict) -> dict:
        """137#/102#/158# PnL fee + preflight + tuning + resilience + A/B test."""
        kwargs: dict = {}
        # 163#: 止血セクション参照 (pnl_fee_deduction, preflight_pause 等のサブキー用)
        止血 = yaml_cfg.get("止血", yaml_cfg.get("loss_control", {}))
        # 137# P1-11: PnL fee 控除
        fee_cfg = 止血.get("pnl_fee_deduction", {})
        if fee_cfg.get("enabled") is not None:
            kwargs["pnl_fee_deduction_enabled"] = fee_cfg["enabled"]
        if "maker_fee_bps" in fee_cfg:
            kwargs["maker_fee_bps"] = fee_cfg["maker_fee_bps"]
        if "taker_fee_bps" in fee_cfg:
            kwargs["taker_fee_bps"] = fee_cfg["taker_fee_bps"]

        # 138# P1-10: preflight pause (dead-cycle 防止)
        pf_pause = 止血.get("preflight_pause", {})
        if pf_pause.get("enabled") is not None:
            kwargs["preflight_pause_enabled"] = pf_pause["enabled"]
        for yk, ck in {
            "threshold": "preflight_pause_threshold",
            "pause_sec": "preflight_pause_sec",
            "max_pauses": "preflight_max_pauses",
        }.items():
            if yk in pf_pause:
                kwargs[ck] = pf_pause[yk]

        # 102# YAML 化: 散在マジックナンバーの設定外部化
        tuning = yaml_cfg.get("tuning", {})
        tuning_map = {
            "max_offset_ratio": "max_offset_ratio",
            "min_offset_ratio": "min_offset_ratio",
            "loss_cap_update_interval": "loss_cap_update_interval",
            "min_loss_cap_jpy": "min_loss_cap_jpy",
            "mid_trend_validity_sec": "mid_trend_validity_sec",
            "balance_margin_ratio": "balance_margin_ratio",
            "balance_shrink_consecutive": "balance_shrink_consecutive",
            "balance_shrink_divisor": "balance_shrink_divisor",
            "skip_gate_recent_trades_limit": "skip_gate_recent_trades_limit",
            "status_unknown_retry_delays": "status_unknown_retry_delays",
            "rate_limit_min_backoff_sec": "rate_limit_min_backoff_sec",
            "save_retry_backoff_sec": "save_retry_backoff_sec",
            "regime_warmup_multiplier": "regime_warmup_multiplier",
            "e3_60s_multiplier": "e3_60s_multiplier",
            "e3_120s_multiplier": "e3_120s_multiplier",
            "adapt_min_side_samples": "adapt_min_side_samples",
            "batch_flush_interval_sec": "batch_flush_interval_sec",
            "heartbeat_interval_sec": "heartbeat_interval_sec",
            # 121# 追加外部化
            "min_order_btc": "min_order_btc",
            "dust_sweep_enabled": "dust_sweep_enabled",  # 128#
            "lock_acquire_retries": "lock_acquire_retries",
            "skip_gate_ob_depth": "skip_gate_ob_depth",
            "retry_backoff_base": "retry_backoff_base",
            "soft_loss_cap_lot_divisor": "soft_loss_cap_lot_divisor",
            "file_log_level": "file_log_level",
            "insufficient_funds_patterns": "insufficient_funds_patterns",
            # 148# §9 #2: heartbeat 設定を YAML から調整可能に
            "lock_heartbeat_period_sec": "lock_heartbeat_period_sec",
            "lock_stale_heartbeat_sec": "lock_stale_heartbeat_sec",
            # 158# YAML 外部化: tuning 追加
            "hot_reload_check_interval_sec": "hot_reload_check_interval_sec",
            "records_cache_ttl_sec": "records_cache_ttl_sec",
            "trades_recorder_fetch_limit": "trades_recorder_fetch_limit",
            "balance_freeze_cycles": "balance_freeze_cycles",
        }
        for yaml_key, config_key in tuning_map.items():
            if yaml_key in tuning:
                kwargs[config_key] = tuning[yaml_key]

        # 158# YAML 外部化: resilience セクション (CircuitBreaker / HealthMonitor)
        resilience = yaml_cfg.get("resilience", {})
        cb = resilience.get("circuit_breaker", {})
        cb_map = {
            "failure_threshold": "cb_failure_threshold",
            "recovery_timeout": "cb_recovery_timeout",
            "success_threshold": "cb_success_threshold",
            "timeout": "cb_timeout",
        }
        for yaml_key, config_key in cb_map.items():
            if yaml_key in cb:
                kwargs[config_key] = cb[yaml_key]
        hm = resilience.get("health_monitor", {})
        hm_map = {
            "rss_warn_mb": "hm_rss_warn_mb",
            "rss_critical_mb": "hm_rss_critical_mb",
            "disk_free_warn_gb": "hm_disk_free_warn_gb",
            "gc_interval_cycles": "hm_gc_interval_cycles",
            "check_interval_sec": "hm_check_interval_sec",
        }
        for yaml_key, config_key in hm_map.items():
            if yaml_key in hm:
                kwargs[config_key] = hm[yaml_key]

        # 158# P1-5: A/B テスト variant 識別子
        ab_test = yaml_cfg.get("ab_test", {})
        if ab_test.get("variant"):
            kwargs["ab_test_variant"] = str(ab_test["variant"])

        return kwargs

    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> "FillTestConfig":
        """YAML dict から FillTestConfig を構築.

        YAML のフラットキー + ネスト (adaptation / lot_sizing / safety) を
        dataclass フィールドにマッピングする.
        """
        kwargs: dict = {}

        # フラットキー (YAML キー == dataclass フィールド名)
        flat_keys = {
            "symbol", "order_quantity", "cycle_interval_sec", "order_timeout_sec",
            "order_timeout_sec_sell",
            "poll_interval_sec", "post_fill_wait_sec", "post_fill_wait_sec_sell",
            "results_dir",
            "max_preflight_skip", "start_side",
            "spread_offset_ratio", "min_offset_jpy",
            "max_order_retries", "retry_delay_sec",
            "as_deadzone_bps", "min_spread_jpy",
            "batch_size", "max_save_retries", "save_fail_threshold",
            "progress_log_interval",
            "log_max_bytes", "log_backup_count",
            "fallback_stale_sec",  # 156# §16
        }
        for key in flat_keys:
            if key in yaml_cfg:
                kwargs[key] = yaml_cfg[key]

        # adaptation セクション → 方策 A
        adapt = yaml_cfg.get("adaptation", {})
        if adapt.get("enabled") is not None:
            kwargs["enable_auto_adapt"] = adapt["enabled"]
        if "interval_cycles" in adapt:
            kwargs["adapt_interval_cycles"] = adapt["interval_cycles"]

        # lot_sizing セクション → 方策 B
        lot = yaml_cfg.get("lot_sizing", {})
        if lot.get("enabled") is not None:
            kwargs["enable_dynamic_lot"] = lot["enabled"]
        if "interval_cycles" in lot:
            kwargs["lot_adapt_interval_cycles"] = lot["interval_cycles"]
        if "max_lot" in lot:
            kwargs["max_lot"] = lot["max_lot"]
        if "recent_pnl_window" in lot:
            kwargs["recent_pnl_window"] = lot["recent_pnl_window"]

        # 151# P3-03: confidence_lot セクション (AS 確率連動ロットサイジング)
        cl = yaml_cfg.get("confidence_lot", {})
        if cl.get("enabled") is not None:
            kwargs["enable_confidence_lot"] = cl["enabled"]
        cl_map = {
            "scale": "confidence_lot_scale",
            "floor": "confidence_lot_floor",
            "mode": "confidence_lot_mode",
        }
        for yaml_key, config_key in cl_map.items():
            if yaml_key in cl:
                kwargs[config_key] = cl[yaml_key]

        # regime セクション → レジーム検知 (035# §4)
        regime = yaml_cfg.get("regime", {})
        if regime.get("enabled") is not None:
            kwargs["enable_regime"] = regime["enabled"]
        regime_map = {
            "window": "regime_window",
            "trend_threshold_pct": "regime_trend_threshold_pct",
            "high_vol_multiplier": "regime_high_vol_multiplier",
            "hysteresis_count": "regime_hysteresis_count",
            "min_confidence": "regime_min_confidence",
            "trending_offset_boost": "regime_trending_offset_boost",
            "trending_offset_boost_buy": "regime_trending_offset_boost_buy",    # 157# §19
            "trending_offset_boost_sell": "regime_trending_offset_boost_sell",   # 157# §19
            # 176# B: 方向×サイド別 offset boost
            "trending_up_buy_offset_boost": "trending_up_buy_offset_boost",
            "trending_up_sell_offset_boost": "trending_up_sell_offset_boost",
            "trending_down_buy_offset_boost": "trending_down_buy_offset_boost",
            "trending_down_sell_offset_boost": "trending_down_sell_offset_boost",
            "high_vol_offset_boost": "regime_high_vol_offset_boost",       # 143# R-1a
            "ranging_offset_discount": "regime_ranging_offset_discount",   # 143# R-1a
            "low_vol_offset_boost_enabled": "low_vol_offset_boost_enabled", # 168#
            "low_vol_offset_boost": "low_vol_offset_boost",               # 168#
            "low_vol_threshold": "low_vol_threshold",                     # 168#
            "skip_ranging_buy_low_vol": "skip_ranging_buy_low_vol",       # 169# B1'
        }
        for yaml_key, config_key in regime_map.items():
            if yaml_key in regime:
                kwargs[config_key] = regime[yaml_key]
        # 143# R-1b: レジーム別 lot 倍率
        if "lot_multipliers" in regime and isinstance(regime["lot_multipliers"], dict):
            kwargs["regime_lot_multipliers"] = {
                str(k): float(v) for k, v in regime["lot_multipliers"].items()
            }
        # 144# R-1c: レジーム別 reprice 上限調整
        if "reprice_adjustments" in regime and isinstance(regime["reprice_adjustments"], dict):
            kwargs["regime_reprice_adjustments"] = {
                str(k): int(v) for k, v in regime["reprice_adjustments"].items()
            }
        # 144# R-1d: レジーム別 timeout 倍率
        if "timeout_multipliers" in regime and isinstance(regime["timeout_multipliers"], dict):
            kwargs["regime_timeout_multipliers"] = {
                str(k): float(v) for k, v in regime["timeout_multipliers"].items()
            }

        # safety セクション → 損失キャップ
        safety = yaml_cfg.get("safety", {})
        if "loss_cap_jpy" in safety:
            kwargs["loss_cap_jpy"] = safety["loss_cap_jpy"]
        if "loss_cap_warning_ratio" in safety:
            kwargs["loss_cap_warning_ratio"] = safety["loss_cap_warning_ratio"]
        # 041# 動的 loss_cap
        if safety.get("loss_cap_auto") is not None:
            kwargs["loss_cap_auto"] = safety["loss_cap_auto"]
        if "loss_cap_ratio" in safety:
            kwargs["loss_cap_ratio"] = safety["loss_cap_ratio"]
        # 046# soft/hard 二段 loss_cap
        if "soft_loss_cap_ratio" in safety:
            kwargs["soft_loss_cap_ratio"] = safety["soft_loss_cap_ratio"]

        # 041# 時間帯フィルター
        tf = yaml_cfg.get("time_filter", {})
        if tf.get("enabled") is not None:
            kwargs["enable_time_filter"] = tf["enabled"]
        if "skip_utc_hours" in tf:
            kwargs["skip_utc_hours"] = tf["skip_utc_hours"]
        # 073# side 別時間帯フィルター
        if "skip_utc_hours_buy" in tf:
            kwargs["skip_utc_hours_buy"] = tf["skip_utc_hours_buy"]
        if "skip_utc_hours_sell" in tf:
            kwargs["skip_utc_hours_sell"] = tf["skip_utc_hours_sell"]
        # 110# 086# デッドロック修正
        if "max_086_consecutive_wait" in tf:
            kwargs["max_086_consecutive_wait"] = tf["max_086_consecutive_wait"]

        # 163# regime 連動動的ゲーティング
        if tf.get("regime_adaptive_enabled"):
            kwargs["regime_adaptive_enabled"] = True
        if "regime_adaptive_extra_buy" in tf:
            kwargs["regime_adaptive_extra_buy"] = tf["regime_adaptive_extra_buy"]
        if "regime_adaptive_extra_sell" in tf:
            kwargs["regime_adaptive_extra_sell"] = tf["regime_adaptive_extra_sell"]

        # 163# ステージ抽出: trading features
        kwargs.update(cls._parse_trading_features(yaml_cfg))

        # 163# ステージ抽出: skip_gate ML filter
        kwargs.update(cls._parse_skip_gate_section(yaml_cfg))

        # 163# ステージ抽出: stale order + VG + sell_guard
        kwargs.update(cls._parse_stale_vg_section(yaml_cfg))

        # 163# ステージ抽出: 止血 + dynamic kill + narrow spread + inventory skewing
        kwargs.update(cls._parse_stopgap_section(yaml_cfg))

        # 163# ステージ抽出: PnL fee + preflight + tuning + resilience + A/B test
        kwargs.update(cls._parse_infra_section(yaml_cfg))

        return cls(**kwargs)


# ======================================================================
# 113# R1: run_single_cycle 分割用 内部データクラス
# ======================================================================

@dataclass
class SkipGateResult:
    """SkipGate ML 判定結果 (run_single_cycle 内部)."""

    skipped: Optional[bool] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    model_used: Optional[str] = None
    as_prob: Optional[float] = None
    threshold_used: Optional[float] = None
    # 158# P1-6: 時間帯別閾値調整のオフセット
    hour_offset: float = 0.0
    # 165# AS-R1: velocity logging
    price_velocity_60s: Optional[float] = None
    early_return_record: Optional[FillRecord] = None


@dataclass
class FillMonitorResult:
    """約定監視結果 (run_single_cycle 内部)."""

    filled: bool = False
    fill_price: Optional[float] = None
    t_fill: Optional[float] = None
    cancel_reason: Optional[str] = None
    queue_wait: float = 0.0
    reprice_count: int = 0
    # 158# P1-3: reprice 累積 drift (bps)
    reprice_drift_bps: float = 0.0
    final_order_price: float = 0.0
    # 145# §9-#2: regime 調整済みの実効タイムアウト (cancel_reason 判定で使用)
    effective_timeout: float = 0.0
    # 166# C.7: cancel 失敗後に約定を検出した場合のフラグ (Bug11)
    cancel_failed_likely_filled: bool = False


@dataclass
class PnlMeasurement:
    """PnL 計測結果 (run_single_cycle 内部)."""

    mid_at_fill: Optional[float] = None
    mid_30s_after: Optional[float] = None
    mid_60s_after: Optional[float] = None
    mid_120s_after: Optional[float] = None
    post_fill_pnl: Optional[float] = None
    post_fill_60s_pnl: Optional[float] = None
    post_fill_120s_pnl: Optional[float] = None
    adverse_selected: Optional[bool] = None
    adverse_selected_raw: Optional[bool] = None
    actual_measurement_sec: Optional[float] = None
    # 120# PnlMeasurer: early_exit_triggered を戻り値に含める
    early_exit_triggered: bool = False
    # 120# A4-2: EE 発動時の中断時点 PnL (post_fill_pnl は常に固定30s)
    pnl_at_exit_bps: Optional[float] = None