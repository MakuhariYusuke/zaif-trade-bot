"""
FillTestConfig — fill_test 設定データクラス.

119# God Object 分割: run_fill_test.py から設定定義を分離.
設定の構造 (FillTestConfig) と YAML→dataclass マッピング (from_yaml) を管理.

329# Step 1: Result dataclasses (SkipGateResult, FillMonitorResult,
PnlMeasurement, compute_ev_offset_multiplier) を fill_config_results.py に分離.
後方互換のため本モジュールからも re-export する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

# 329# re-export: 既存の import パスを維持
from scripts.v460.lib.fill_config_results import (  # noqa: F401
    FillMonitorResult,
    PnlMeasurement,
    SkipGateResult,
    compute_ev_offset_multiplier,
)


# ======================================================================
# Configuration
# ======================================================================


@lru_cache(maxsize=1)
def _resolve_fill_config_yaml_parser():
    """Resolve the split YAML parser once while keeping circular imports lazy."""
    from scripts.v460.lib.fill_config_parser import parse_fill_config_yaml

    return parse_fill_config_yaml

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
    # 209# M4: sleep 乗数積み重ねの上限 (0=無制限)
    # soft_dd_mult × loss_cooldown × one_sided_mult の積で interval が過大になるのを防止
    max_cycle_sleep_sec: float = 600.0  # 10分上限 (0=無制限)
    # 276# halt_sleep_multiplier: halt/HALT 系 skip の sleep 倍率
    # 理論的根拠: Brunnermeier & Pedersen (2009) — 流動性スパイラル発生時は
    # 取引再開までの待機時間を通常サイクルの N 倍に延長し、価格衝撃減衰を待つ。
    # デフォルト 5.0 はサイクル間隔 120s × 5 = 600s (10分) の halt 周期を意味する。
    halt_sleep_multiplier: float = 5.0
    # 277# phantom 検出後の sleep 倍率 (Avellaneda-Stoikov §3.2)
    # 在庫不一致検出時は逆選択リスクが高いため、halt ほど長くない中間的待機。
    # デフォルト 3.0 → cycle_interval × 3.0 = 360s (6分)。
    phantom_detection_sleep_multiplier: float = 3.0
    # 277# halt 中の state/record 保存間隔 (halt イテレーション数)
    # halt_sleep (600s) × 10 = 6000s (100分) 毎に state 保存。再起動時の巻き戻し防止。
    halt_persist_interval: int = 10
    # 277# fill_rate / avg_pnl30 停止条件モニターの実行間隔 (サイクル数)
    # 30 cycles × 120s = 3600s (1h) — 過剰な計算を避けつつ市場変化に追従。
    stop_condition_check_interval: int = 30
    # 277# 停止条件発動時の fallback 持続時間 (秒)
    # fill_rate 低下/avg_pnl30 悪化時に CycleStrategy を fallback に切替える期間。
    # 理論的根拠: Kyle (1985) — 情報非対称性下では 1h の冷却期間で price discovery が収束。
    fallback_duration_sec: float = 3600.0
    # 242# Quiescence (233# P1: No Trade = 正常系)
    # 連続 gate block が閾値を超えたら quiescence (静止) 状態と認定し、
    # sleep 上限を max_cycle_sleep_sec → quiescence_sleep_sec に引き上げる。
    # 市場理論的根拠: Glosten-Milgrom — 逆選択リスク極大時は流動性供給を撤退し、
    # 市場構造の変化を待つのが最適戦略。数時間の No Trade は異常ではなく正常系。
    quiescence_gate_blocks_threshold: int = 20  # 0=無効  # 連続ゲートブロック → quiescence
    quiescence_sleep_sec: float = 1800.0  # quiescence 時 sleep 上限 (30分, 0=無効)
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
    regime_min_confidence: float = 0.2  # 052# 0.4→0.3, 152# 0.3→0.2, 343# default sync
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
    # 397# mid-confidence paradox guard: confidence [0.7,0.9) の逆転ゾーン対策
    # 395# 実証: SHA-fenced でも再現した唯一の構造的問題
    # confidence [0.7,0.9) は全 SHA で paradoxical underperformance (−0.734 bps, WR=46%)
    regime_mid_confidence_offset_boost: float = 1.0  # 1.0=無効, >1.0 で offset 拡大
    regime_mid_confidence_lo: float = 0.7   # 対象 confidence 帯域の下限
    regime_mid_confidence_hi: float = 0.9   # 対象 confidence 帯域の上限
    # 143# R-1a: レジーム別 offset 調整
    regime_high_vol_offset_boost: float = 1.2   # high_vol 時に offset × 1.2 (+20% 拡張)
    regime_ranging_offset_discount: float = 1.0 # ranging 時に offset × N (1.0=無効, <1.0で縮小)
    # 440# ranging offset の buy/sell 非対称化 (432# buy+ranging PF=0.766 対策)
    # None=共通値(regime_ranging_offset_discount)使用
    # buy: 1.15 推奨 (現行 0.90 は逆効果 — offset 縮小が AS リスク増大)
    # sell: 0.85 推奨 (near-breakeven — fill_rate 向上優先)
    regime_ranging_offset_discount_buy: float | None = None
    regime_ranging_offset_discount_sell: float | None = None
    # 303# C: none レジーム Passive MM フォールバック
    # regime 未確定 (warmup/欠損) 時に 13 段パイプラインをバイパスし固定 offset で指値
    # AS 43% の根本対策: 情報不足時はパッシブに待機
    # 318# F5-1: "unknown" も対象 (旧実装は "none" のみで事実上死んでいた)
    none_regime_passive_mm_enabled: bool = False
    none_regime_fixed_offset_bps: float = 2.0  # 固定 offset (bps of mid_price)
    # 227# C1: Ranging × OBI (Order Book Imbalance) 方向別非対称 offset
    # AS理論: ranging市場ではOBIがmean-reversion方向を予測
    ranging_obi_asymmetry_factor: float = 0.3  # 344#: 0.0→0.3 (ranging OBI方向シグナル)
    ranging_obi_threshold: float = 0.1         # |imbalance| がこの値以下では中立扱い
    # 168# §9.10: 低ボラティリティ offset boost (time_filter 根本対策)
    # vol_ratio < threshold 時に offset を拡大し、低ボラ環境での過剰アグレッシブ発注を抑制
    low_vol_offset_boost_enabled: bool = False
    low_vol_offset_boost: float = 1.4   # 低 vol 時の offset 倍率 (最大値 / 固定値)
    low_vol_threshold: float = 0.75     # vol_ratio がこの値未満で発動 (168# 0.70→0.75: order/fill遅延マージン)
    # 200# C: 低ボラ boost 比例モード (vol_ratio に応じた段階的 boost)
    low_vol_boost_proportional: bool = False  # True=比例スケーリング, False=固定 boost
    low_vol_boost_min: float = 1.0             # 比例モード時の最小 boost (vol_ratio=threshold でこの値)
    # 169# B1': ranging_buy at low_vol ハードスキップ (Gemini 10.2-D「休むも相場」)
    # ranging レジーム + buy + vol_ratio < low_vol_threshold → 完全スキップ
    # ranging_buy が全損失の 69% を占める根本対策。offset 調整より clean
    skip_ranging_buy_low_vol: bool = False
    # 195# B1' ソフト化: hard skip → maker_price low_vol_offset_boost に委譲
    # enabled 時、ranging_buy_low_vol のハードスキップを無効化し offset boost のみで対応
    ranging_buy_low_vol_as_offset: bool = False
    # 189# D: Macro Regime 統合
    enable_macro_regime: bool = False  # MacroRegimeDetector 有効化
    macro_regime_bucket_sec: float = 30.0  # 時間バケットサイズ
    macro_regime_slope_threshold: float = 1.0  # bps/min — trending 判定閾値
    macro_regime_strong_threshold: float = 3.0  # bps/min — strong trending 閾値
    macro_regime_conflict_action: str = "log"  # "log" or "downgrade" — 矛盾時動作
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
    # 200# 10-A/10-E: soft ドローダウン時の interval 乗数 (YAML 外部化)
    soft_drawdown_interval_multiplier: float = 3.0
    # 205# §9.5: 片側 DD Halt — サイド別累積損失超過で片側取引停止
    per_side_dd_enabled: bool = False           # True で片側 DD ガードを有効化
    per_side_dd_hard_limit_bps: float = -50.0   # 片側累積 PnL がこの bps 以下でそのサイドを封鎖 (364# TUNE-2)
    per_side_dd_halt_cycles: int = 10           # 封鎖サイクル数 (364# TUNE-2: 既定20分, 0=UTC 日替わりまで永続封鎖)
    # 224# B1: halt解除後ソフトリカバリ — lot 縮小で段階的復帰
    per_side_dd_recovery_cycles: int = 5        # リカバリ期間サイクル数 (0=無効)
    per_side_dd_recovery_lot_scale: float = 0.5 # リカバリ期間中の lot 倍率
    # 269# per-side halt PnL リアンカー: release 時に side PnL を部分リセット
    # release 後は「過去の負債」ではなく「release 後の追加損失」で再 halt 判定
    per_side_dd_reanchor_budget_bps: float = -25.0  # release 後にこの追加損失で再 halt (364# TUNE-2, -25bps)
    # 225# 市場理論補強: regime-aware recovery lot ペナルティ
    recovery_trending_penalty: float = 0.7  # trending 時のリカバリ lot 追加縮小倍率
    recovery_high_vol_penalty: float = 0.8  # high_vol 時のリカバリ lot 追加縮小倍率
    # 246# DD halt cooldown release: 集約 halt 後 N 秒で lot 縮小付き再開
    dd_cooldown_release_sec: float = 0.0       # 0=無効, 例: 7200=2h後に部分解除
    dd_cooldown_release_lot_scale: float = 0.3 # cooldown release 中の lot 倍率
    # 249# DD halt cooldown re-arm: release 後の追加損失で再 halt
    dd_cooldown_rearm_budget_bps: float = -10.0  # release 後にこの bps 以下で再 halt
    # 303# B: DD soft lot reduction の side 分離
    # True 時: soft_triggered で当該 side のみ lot 縮小 (他 side は据え置き)
    daily_drawdown_soft_lot_side_aware: bool = False
    # 268# DD 日付リセット TZ: 0=UTC, 9=JST。JST 運用なら 9.0 推奨 (halt 最大時間が ~22h→14h に短縮)
    dd_day_reset_utc_offset_hours: float = 9.0
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
    # 230# H-1: Layer 2 deadzone — 正常 spread cost を誤検知しない閾値 (bps)
    ffd_l2_deadzone_bps: float = 3.0
    # 230# H-2: boost 解除に必要な連続正常 fill 数 (Kyle 1985)
    ffd_boost_release_streak: int = 3
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
    # ---- 306# L2: Microprice Side Selection ----
    # microprice = (Pb·Qa + Pa·Qb) / (Qa + Qb)
    # microprice > mid → sell 優先 (買い圧力強), microprice < mid → buy 優先
    # 300# §2.3 Smart Side がレジーム不感症 → microprice で構造的 AS 低減
    microprice_side_enabled: bool = False    # True で microprice ベース side 選択を有効化
    microprice_side_threshold: float = 0.3  # microprice 偏向度 (bps) がこの値以上で side 切替
    # 366# M1: Multi-level microprice (Gatheral 2018)
    microprice_depth: int = 5               # microprice で使用する板の段数 (1=従来互換, 最大5)
    microprice_min_qty: float = 0.0001      # 有効レベルの最小数量 (これ未満の段はスキップ)
    # 310# C: L2 Safety Mode 再有効化ガードレール (308# 盲点1 設計改修)
    # 無条件有効化ではなくスプレッド・レジーム条件付きで制御
    microprice_side_min_spread_bps: float = 15.0  # spread がこの値 (bps) 以上の時のみ有効
    microprice_side_regime_gate: list[str] = field(
        default_factory=lambda: ["ranging"]  # これらの regime でのみ有効 (空=全レジーム)
    )
    # ---- 306# L1: Dynamic Cycle Interval ----
    # cycle_interval_sec ∝ 1/σ: 高σ時に短周期 (素早い在庫回転), 低σ時に長周期 (API節約)
    # 300# §1.3 VG は量のフィルタ → σ 連動で質も改善
    dynamic_cycle_interval_enabled: bool = False  # True で σ 連動 interval を有効化
    dynamic_cycle_interval_min_sec: float = 60.0   # interval 下限 (秒)
    dynamic_cycle_interval_max_sec: float = 300.0  # interval 上限 (秒)
    dynamic_cycle_interval_sigma_ref: float = 0.0005  # σ 基準値 (この σ で cycle_interval_sec)
    # ---- 306# O1: Queue Position Estimation ----
    # 発注時の same_side_depth_ahead を記録し fill probability を推定
    # 低確率注文の早期 cancel で機会コスト削減
    queue_position_tracking_enabled: bool = False  # True で QPE を有効化
    queue_position_early_cancel_prob: float = 0.05  # fill prob がこの値以下で早期 cancel
    # ---- 306# S1: Offset Stage 寄与記録 (301# F6, 300# T0-1) ----
    # offset パイプライン各ステージの寄与量を FillRecord に記録 → 定量分析基盤
    offset_stage_recording_enabled: bool = False  # True で stage-by-stage 記録を有効化
    # ---- 310# A: Sell AS Time-of-Day Offset Boost (307# F3, 306# H5) ----
    # 306# 実証: UTC 08h AS=63%, 13-14h AS=42-43%, 16h AS=61% (sell)
    # sell 時間帯別 offset multiplier — Ho-Stoll (1981): 情報非対称の時間変動を反映
    # key=UTC hour, value=offset multiplier (1.0=無効, >1.0=拡大)
    sell_hour_offset_boost: dict[int, float] = field(default_factory=dict)
    # ---- 306# T1-3: Max Offset Ratio 天井 (300# §2.1 構造的矛盾 #2) ----
    # 全ステージが offset を拡大する方向にのみ作用 → 天井で toxic fill only trap を防止
    offset_ceiling_ratio: float = 0.0  # 0.0=無効, >0 で offset 上限 (e.g. 0.15)
    # 320# C-1: サイド別 ceiling (sell floor(0.30) > ceiling(0.15) 矛盾解消)
    # None=共通値(offset_ceiling_ratio)使用、>0 でサイド別上限
    offset_ceiling_ratio_buy: float | None = None
    offset_ceiling_ratio_sell: float | None = None
    # ---- 421# P0: Execution Final Clamp (416#/417# review: post-ceiling leak) ----
    # maker_price ceiling 後の executor 側 multiplier chain が ceiling を迂回する問題の修正。
    # 全 multiplier 適用後に ceiling を再適用し、offset ratio の暴走を防止。
    execution_final_clamp_enabled: bool = True   # Final Clamp 有効化 (安全のためデフォルト有効)
    execution_final_clamp_hard_skip_mult: float = 0.0  # >0: pre-clamp が ceiling×この倍率を超えたら hard skip (0=無効)

    def resolve_offset_ceiling(self, side: str) -> float:
        """421# DRY: サイド別 offset ceiling を解決する共通ヘルパー.

        maker_price.py L1015 / fill_cycle_executor.py Final Clamp で
        同一パターンが3重複していたため統一。
        Returns: ceiling 値 (0.0 = 無効)。
        """
        ceil = self.offset_ceiling_ratio
        if side == "buy" and self.offset_ceiling_ratio_buy is not None:
            ceil = self.offset_ceiling_ratio_buy
        elif side == "sell" and self.offset_ceiling_ratio_sell is not None:
            ceil = self.offset_ceiling_ratio_sell
        return ceil

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
    # 439# 433# §3 / 434# §4.2: cross-venue lead-lag guard
    # BitFlyer 等の高流動 venue を directional override ではなく、
    # adverse-side retreat / veto の補助票として使う。
    cross_venue_lead_lag_enabled: bool = False
    cross_venue_reference_exchange: str = "bitflyer"
    cross_venue_lead_lag_max_age_sec: float = 3.0
    cross_venue_lead_lag_spread_bps_threshold: float = 1.0  # 444#
    cross_venue_lead_lag_velocity_bps_threshold: float = 0.01  # 444#
    cross_venue_lead_lag_offset_boost: float = 1.25
    cross_venue_lead_lag_veto_enabled: bool = False
    cross_venue_lead_lag_veto_threshold_bps: float = 6.0
    # 442# 板深度拡張: L5 microprice + depth imbalance
    cross_venue_reference_ob_depth: int = 5
    cross_venue_microprice_enabled: bool = False
    cross_venue_depth_imbalance_enabled: bool = False
    cross_venue_depth_imbalance_boost: float = 1.15
    # 445# EMA 平滑化 + confidence scoring
    cross_venue_ema_alpha: float = 0.3
    cross_venue_min_confidence: float = 0.2
    cross_venue_confidence_reference_spread_bps: float = 3.0
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
    # 190# A: ev_weighted 連続 skip 安全弁 (0=無効, N回連続skipで強制PASS)
    # 193#: 安全弁は廃止予定。ev_as_offset_enabled=True 時は無視される。
    skip_gate_ev_max_consecutive_skip: int = 0
    # 190# B: 片側 balance 時の ev_weighted threshold 緩和シフト (bps)
    # 193#: ev_as_offset_enabled=True 時は無視される。
    skip_gate_ev_one_sided_threshold_shift: float = 0.0
    # 193#: ev_weighted → offset 修飾子モード (192# §5.2 + Gemini §9.4)
    # True: ev_weighted は PASS/SKIP ゲートではなく offset 乗数として機能
    skip_gate_ev_as_offset_enabled: bool = False
    # ev_score → offset 乗数の感度: mult = 1.0 + sensitivity × ev_score
    # sensitivity=0.05, ev=-3.0 → mult=0.85 (15%保守的), ev=+2.0 → mult=1.10
    skip_gate_ev_offset_sensitivity: float = 0.05
    # offset 乗数のクランプ範囲
    skip_gate_ev_offset_min_mult: float = 0.5
    skip_gate_ev_offset_max_mult: float = 1.5
    # 緊急スキップ: ev_score がこの値未満なら依然ハードスキップ
    skip_gate_ev_emergency_skip_threshold: float = -8.0
    # 200# M: ev warning zone — emergency と通常の間の中間段階
    # warning zone: emergency < ev_score < warning → offset を追加縮小
    skip_gate_ev_warning_threshold: float = -4.0  # この値未満で warning zone 発動
    skip_gate_ev_warning_offset_factor: float = 0.7  # warning zone での追加 offset 乗数
    skip_gate_as_threshold: float = 0.52   # 100# AS 確率スキップ閾値 (0.65→0.52)
    skip_gate_pnl_threshold: float = 0.0   # PnL 予測スキップ閾値 (mode=pnl)
    skip_gate_max_skip_rate: float = 0.3   # 連続スキップ率上限 (安全弁)
    # 068# §3.3: side 別閾値 (None は共通 as_threshold を使用)
    skip_gate_as_threshold_buy: float | None = None
    skip_gate_as_threshold_sell: float | None = None
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
    # 205# §9.4: 時間帯 Hard Skip — 最悪時間帯は取引完全停止 (Kyle proxy)
    # UTC hour のリスト。これらの時間帯では soft offset ではなくサイクル全停止
    hard_skip_utc_hours: list[int] = field(default_factory=list)
    # 124# Rule: unknown regime での sell スキップ
    skip_sell_unknown_regime: bool = False
    # 130# unknown regime での buy offset boost (AS 回避)
    unknown_buy_offset_boost: float = 1.0  # 1.0 = 無効, >1.0 で boost (例: 2.0 = VG相当)
    # 440# unknown regime sell offset boost (440# §4.3)
    # sell+unknown PnL=-0.39, AS=52.2% → buy ほど深刻ではないが要対策
    unknown_sell_offset_boost: float = 1.0  # 1.0 = 無効, >1.0 で boost
    # 165# AS-R1: velocity-based sell/buy skip (SkipGate pre-ML rule)
    sell_velocity_skip_enabled: bool = False
    sell_velocity_skip_threshold_bps: float = 8.0  # price_velocity_bps > this AND sell -> skip
    buy_velocity_skip_enabled: bool = False
    buy_velocity_skip_threshold_bps: float = -8.0  # price_velocity_bps < this AND buy -> skip
    # 195# velocity_skip ソフト化: hard skip → offset boost
    # enabled 時、閾値超過でもスキップせずに offset を boost して保守的価格で発注
    velocity_skip_as_offset_enabled: bool = False
    velocity_offset_boost_factor: float = 1.5  # 197# 2.0→1.5 (boost 1.0-1.5帯 PnL+0.47 vs 2.0帯 PnL-0.37)
    # 196# velocity offset 段階的 boost: 閾値超過量に比例した乗数
    velocity_offset_proportional: bool = False  # True=比例, False=固定
    velocity_offset_max_mult: float = 4.0  # 比例モード時の上限乗数
    # 183# narrow spread 時の skip_gate 閾値オフセット (逆選択防御)
    # spread < narrow_spread_skip_threshold_jpy のとき threshold に加算。
    # ログ分析: spread<2kでAS32% (全体28%) → 閾値厳格化で AS fill削減
    skip_gate_narrow_spread_threshold_jpy: float = 0.0  # 0.0=無効
    skip_gate_narrow_spread_offset: float = 0.0  # 正=厳格化 (PnLモード)
    # 187# clamp YAML外部化: skip_gate offset の上下限
    skip_gate_offset_floor: float = -0.3   # 最大緩和
    skip_gate_offset_ceil: float = 0.5     # 最大厳格化
    # 343# skip_gate/kill 連携: kill 解除直後は skip_gate を緩和して過剰抑制を防止
    # kill 中は skip_gate が実行されず adaptive data が stale になるため、
    # 解除後 N サイクルは offset を負方向 (緩和) にシフトする。
    skip_gate_kill_release_grace_cycles: int = 3    # kill 解除後の緩和サイクル数
    skip_gate_kill_release_offset: float = -0.1     # 緩和 offset (負=緩和)
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
    # 292# P1: reprice deadband — 価格差がこの値未満なら queue 保護のためスキップ
    stale_reprice_min_delta_jpy: float = 0.0
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
    # ---- 257# VPIN Continuous Modulator: バイナリ → 連続スケーリング ----
    # VPIN が min から threshold の間でも段階的に boost を適用し、
    # 情報非対称性リスクを滑らかに offset に反映する。
    vg_vpin_continuous_enabled: bool = False  # True で VPIN 連続スケーリング有効化
    vg_vpin_continuous_min: float = 0.40     # 連続スケーリング開始の VPIN 下限
    # 353# VPIN 非対称 buy boost: buy 側の VPIN boost を追加増幅
    # 351# 盲点1「Ranging ≠ 対称」対応: buy は sell より VPIN リスクが構造的に高い
    vg_vpin_buy_extra_mult: float = 1.0      # 1.0=対称(従来), >1.0=buy 側追加増幅
    # ---- 366# M5: Volume-Sync VPIN (Easley 2012) ----
    # 出来高バケットベースの VPIN。低出来高時の安定性向上。
    vpin_vol_sync_enabled: bool = False      # True で volume-sync VPIN 有効化
    vpin_vol_sync_bucket_btc: float = 0.05   # 1 バケットあたりの出来高 (BTC)
    vpin_vol_sync_n_buckets: int = 50        # 平均に使うバケット数
    # ---- 366# M2: Bayesian Regime Filter (Hamilton 1989) ----
    # 閾値ベース regime 分類をベイズ事後確率で補完。confidence を精緻化。
    bayesian_regime_enabled: bool = False    # True で Bayesian filter 有効化
    bayesian_regime_stickiness: float = 0.90 # 遷移行列の対角要素割合 (状態持続性)
    bayesian_regime_emission_lr: float = 0.01  # emission パラメータのオンライン学習率
    # ---- 366# M3: σ-Clustering (Vol Regime Classification) ----
    # ボラティリティを 4 段階に分類し、adaptation の step_ratio を動的調整。
    sigma_clustering_enabled: bool = False   # True で σ-clustering 有効化
    sigma_clustering_low_threshold: float = 0.6   # LOW ↔ MID 境界 (vol_ratio)
    sigma_clustering_high_threshold: float = 1.5  # MID ↔ HIGH 境界
    sigma_clustering_extreme_threshold: float = 3.0  # HIGH ↔ EXTREME 境界
    # ---- 366# M4: GLFT Fill Probability — 動的 k 推定 ----
    # fill_records から到着率 k を回帰推定し、AS δ* の静的 k を動的置換。
    glft_dynamic_k_enabled: bool = False     # True で動的 k 推定有効化
    glft_dynamic_k_min_samples: int = 20     # 推定に必要な最小サンプル数
    # ---- 374# Phase 3.1: SAC Sidecar Proportional Boost ----
    # SAC 連続出力 [-1,+1] を比例的にオフセットへ変換 (v1 の離散分類を置換)。
    # 375#/376# 安全制約: max_boost_bps ≤ 0.20 (hard ceiling)。
    # 377# ladder 検証計画: 0.10 → 0.15 → 0.20 bps step-up。
    sidecar_enabled: bool = True             # sidecar offset 注入の有効/無効
    sidecar_max_boost_bps: float = 0.15      # 最大ブースト (bps), 375# hard ceiling=0.20
    sidecar_dead_zone: float = 0.10          # |bias| ≤ この値で offset=0 (ノイズ除去)
    sidecar_shaping: str = "linear"          # "linear" | "quadratic" | "sigmoid"
    sidecar_use_v2: bool = True              # True=v2(比例), False=v1(離散) — A/B切替用
    # 211# P1-B: Micro Circuit Breaker (短期価格急変の自動検知・防御)
    mcb_enabled: bool = False
    mcb_caution_sigma: float = 1.0
    mcb_warning_sigma: float = 1.5
    mcb_halt_sigma: float = 2.0
    mcb_halt_cooldown_sec: float = 300.0
    mcb_warning_offset_mult: float = 1.5
    mcb_warning_interval_mult: float = 2.0
    # 211# P1-C: Spread Anomaly Detector (流動性枯渇検知)
    sad_enabled: bool = False
    sad_wide_ratio: float = 2.0
    sad_dry_ratio: float = 4.0
    sad_frozen_ratio: float = 8.0
    sad_baseline_window_sec: float = 3600.0
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
    # 348# balance_forced 撤廃: skip_balance_forced 以下の関連設定を削除
    # 202# A: 単一サイクル大損失クールダウン — 大損後の即連鎖を防止
    loss_cooldown_threshold_bps: float = -10.0  # この PnL 以下で次サイクルの interval を延長
    loss_cooldown_interval_mult: float = 2.0    # 損失後のインターバル乗数 (1サイクル限定)
    # 204# I: Per-fill loss offset boost — 大損後に次回 offset を一時的に拡大
    # loss_cooldown (interval延長) + toxic_veto (side封鎖) に加え、offset も防御拡大
    loss_boost_offset_mult: float = 1.5  # 大損後の offset 乗数 (1.0=無効)
    # 226# T1: loss_boost 指数減衰 τ (秒) — AS理論に基づく情報非対称性リスク減衰
    # τ=300s → 5分で boost 63%減衰, 10分で86%減衰, 15分で95%減衰
    loss_boost_decay_tau_sec: float = 300.0
    # 205# §9.2: Toxic Fill 同一サイド拒否 — 大損後に同一方向を N サイクル完全封鎖
    # loss_cooldown (202# A) は interval 2x 延長のみで不十分。同一サイドの連鎖損失を遮断
    toxic_fill_veto_threshold_bps: float = -5.0  # この PnL 以下で同一サイド拒否発動
    toxic_fill_veto_cycles: int = 3              # 拒否サイクル数 (0=無効)
    # 202# B: 片側残高枯渇時にも rescue offset を適用 (通常の rescue は deadlock 用)
    one_sided_balance_rescue_offset: bool = True  # True で one_sided_balance 時も offset 保護
    # 207# §4: one-sided 連続実行制限 — 片側残高枯渇での連続強制実行を制限
    # 205# §4.2 Codex: offset だけでは不十分、interval 延長 + 連続制限が必要
    one_sided_consecutive_limit: int = 5  # 片側強制取引の連続上限 (0=無制限)
    one_sided_consecutive_interval_mult: float = 3.0  # 上限到達時の interval 乗数
    # 234# one-sided エスカレーション: limit超過後の段階的強化
    one_sided_escalation_cooldown_offset: int = 2   # limit+N で cooldown (skip N cycles)
    one_sided_escalation_cooldown_cycles: int = 2   # cooldown 時スキップするサイクル数
    one_sided_escalation_freeze_offset: int = 4     # limit+N で freeze (当該side N cycles凍結)
    one_sided_escalation_freeze_cycles: int = 3     # freeze 時のサイクル数
    # 234# 縮退清算モード — kill gate blocked 時の安全実行
    # 348# balance_forced 撤廃: inventory_escape 経由でのみ縮退清算を発動
    # min lot + wide offset で安全に縮退清算する (Gemini 233# / Codex 232# 共同提言)
    degraded_liquidation_enabled: bool = True       # 縮退清算モードの有効/無効
    degraded_liquidation_lot_mult: float = 0.2      # 通常 lot の 20% (min lot 相当)
    degraded_liquidation_offset_mult: float = 3.0   # offset を通常の 3 倍 (wide offset)
    degraded_liquidation_duty_cycle: int = 3        # N サイクルに 1 回のみ実行 (dutyCycle=3 → 33%)
    # 269# P0: Inventory Escape Mode — per-side halt 時のデッドロック解消
    # Codex 269# §4.1 / Gemini 270# Action A: 在庫過多で JPY 不足、反対 side は halt
    # → 完全停止ではなく、halt を一時的に貫通して縮退清算 (degraded liquidation パラメータを流用)
    inventory_escape_enabled: bool = True           # Inventory Escape の有効/無効
    inventory_escape_duty_cycle: int = 5            # N サイクルに 1 回のみ実行 (halt 貫通は控えめ)
    # 277# unknown regime 連続ブロックのバイパス閾値 (Hamilton 1989 regime-switching)
    # unknown が N 回連続したら regime 判定不能として gate を強制バイパス。
    # 10 × cycle_interval(120s) = 20 分 — regime 再評価の猶予期間。
    unknown_regime_max_consecutive: int = 5   # 336# drift fix: YAML=5 (321# M-3)
    # ---- 133# P0-09: unknown レジームでの buy スキップ ----
    skip_buy_unknown_regime: bool = False  # True で unknown レジーム時 buy もスキップ (-1.384bps)
    # ---- 155# §9: trending レジームでの sell 抑制 ----
    skip_sell_trending: bool = False  # True で trending 時 sell をスキップ (-0.687bps)
    # 156# D-4: trending 方向別分解 — True なら trending_up のみスキップ (trending_down sell を開放)
    skip_sell_trending_up_only: bool = False
    # 251# Sell Asymmetric Mode (248# P1):
    #   Glosten-Milgrom の情報非対称下、情報劣位者 (MM) の sell は
    #   trending_up と同様に high_vol でも逆選択リスクが高い。
    #   「No Trade = 正常」(242# 思想) を high_vol にも拡張。
    sell_asymmetric_high_vol_enabled: bool = False
    # 196# trending_sell ソフト化: hard skip → offset boost
    # enabled 時、trending sell をスキップせず offset を boost して保守的価格で sell 発注
    trending_sell_as_offset_enabled: bool = False
    trending_sell_offset_boost_factor: float = 1.5  # 336# drift fix: YAML=1.5 (320# C-1)
    # 253# 削除完了: balance_forced_apply_trending_offset (234# dead config → 235# TODO 解消)
    # 158# §20-B: 連続 trending sell skip 安全弁 — N 回超過で sell を強制許可 (0=無制限)
    max_consecutive_trending_sell_skip: int = 30
    # 171# Guard Paradox 対策: 在庫偏重時に sell ガードを自動緩和
    # 344# 342#B: 0.3→0.0 ステップ関数廃止 → inv_relaxation max_bps拡大で gradual 化
    sell_guard_inv_bypass_threshold: float = 0.0
    # ---- 133# P0-10: sell 動的 kill (rolling PnL ベースの自動停止) ----
    sell_dynamic_kill_enabled: bool = False  # True で sell rolling PnL 監視有効
    sell_dynamic_kill_window: int = 50       # rolling ウィンドウ (fill 数)
    sell_dynamic_kill_threshold_bps: float = -0.5  # 364# TUNE-3: -0.3→-0.5 (SDK kill主因, 361# F7制約: ewma据置)
    sell_dynamic_kill_resume_window: int = 10     # 336# drift fix: YAML=10 (156# D-5)
    # 139# §9-#2: レジーム別閾値 (regime_name -> threshold_bps)
    sell_dynamic_kill_regime_thresholds: dict[str, float] = field(default_factory=dict)
    # 243# 242# YAML 配線: toxic_kill_stale_multiplier
    sell_dynamic_kill_toxic_stale_mult: int = 10   # 242# probe interval 延長倍率
    # 269# probe/force-release YAML 露出 (250# 廃止検討対応)
    sell_dynamic_kill_max_stale_cycles: int = 0    # 336# drift fix: YAML=0 (269# probe無効)
    sell_dynamic_kill_max_force_probes: int = 0    # 336# drift fix: YAML=0 (269# force-release無効)
    # 273# kill 時間上限 (268# I5: Pattern B kill↔halt 相互ロック防止)
    sell_dynamic_kill_max_duration_sec: float = 1800.0  # 336# drift fix: YAML=1800 (273#)
    sell_dynamic_kill_ewma_alpha: float = 0.05  # 344# 342#D: EWMA α (0=無効)
    sell_dynamic_kill_ewma_time_decay_tau_sec: float = 0.0  # 353# EWMA 時間減衰 τ (0=無効)
    # ---- 157# §19: buy 動的 kill (rolling PnL ベースの自動停止 — sell との対称性) ----
    buy_dynamic_kill_enabled: bool = False   # True で buy rolling PnL 監視有効
    buy_dynamic_kill_window: int = 50        # rolling ウィンドウ (fill 数)
    buy_dynamic_kill_threshold_bps: float = -0.8  # 341# revert: 340#符号修正後の正常値
    buy_dynamic_kill_resume_window: int = 10      # 停止後、N サイクル後に再評価
    buy_dynamic_kill_regime_thresholds: dict[str, float] = field(default_factory=dict)
    buy_dynamic_kill_toxic_stale_mult: int = 10    # 242# probe interval 延長倍率
    buy_dynamic_kill_max_stale_cycles: int = 0     # 336# drift fix: YAML=0 (269# probe無効)
    buy_dynamic_kill_max_force_probes: int = 0     # 336# drift fix: YAML=0 (269# force-release無効)
    # 273# kill 時間上限 (268# I5)
    buy_dynamic_kill_max_duration_sec: float = 1800.0  # 336# drift fix: YAML=1800 (273#)
    buy_dynamic_kill_ewma_alpha: float = 0.05  # 344# 342#D: EWMA α (0=無効)
    buy_dynamic_kill_ewma_time_decay_tau_sec: float = 0.0  # 353# EWMA 時間減衰 τ (0=無効)
    # 286# 283# P1-4: 在庫連動の buy_dynamic_kill 閾値緩和 (Ho & Stoll 1981)
    # 在庫偏重時 (BTC 不足) に buy kill を緩和して在庫リバランスを促進。
    # 閾値オフセット = min(|imbalance| × scale, max_bps)
    # 例: imbalance=-0.5 (buy寄り=BTC不足), scale=0.5, max=0.3 → offset=+0.25bps → 緩和
    buy_dynamic_kill_inv_relaxation_enabled: bool = False  # True で在庫連動緩和を有効化
    buy_dynamic_kill_inv_relaxation_scale: float = 0.5     # |imbalance| → offset 変換スケール
    buy_dynamic_kill_inv_relaxation_max_bps: float = 0.3   # 341# revert: 340#符号修正後の正常値
    # 337# 在庫連動 sell_dynamic_kill 緩和 (Ho & Stoll 1981 対称性)
    sell_dynamic_kill_inv_relaxation_enabled: bool = False  # True で在庫連動緩和を有効化
    sell_dynamic_kill_inv_relaxation_scale: float = 0.4     # buy(0.5)より保守的
    sell_dynamic_kill_inv_relaxation_max_bps: float = 0.5   # 344# 342#B: 0.3→0.5 inv_bypass廃止補完
    # 348# balance_forced 撤廃: forced_fill_pnl_downweight, forced KPI/delay 設定を削除
    # 249# dual_kill_bypass → quiescence: 両方 kill 時は休止 (242# "No Trade = normal")
    dual_kill_quiescence_enabled: bool = False  # True で dual_kill_bypass を無効化 → 静観
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
    # 249# Regime-aware inv skewing: trending 時は在庫偏重補正を無効化
    inv_skew_regime_gate_enabled: bool = False  # True で trending 時の inv_skew を停止
    # 228# C2: 在庫偏重の時間減衰 — 古い fill 履歴の影響を指数関数的に減衰
    inv_decay_tau_sec: float = 1800.0          # 344#: 0→1800 (30分 τ 古いfill履歴減衰)
    # ---- 257# AS Reservation Price: Avellaneda-Stoikov 在庫×ボラ連動 offset ----
    # 在庫リスクをボラティリティに応じて非対称 offset 補正する理論的基盤。
    # 既存 inv_skew (線形) を σ²·τ で補完し、高ボラ時に在庫リバランスを加速。
    as_reservation_enabled: bool = False     # True で AS reservation shift を有効化
    as_reservation_gamma: float = 0.1        # γ: リスク回避度 (0=中立, 高い=保守的)
    as_reservation_tau_sec: float = 120.0    # τ: 時間ホライゾン (秒)
    # ---- 266# GLFT τ動的化: Guéant-Lehalle-Fernandez-Tapia (2013) ----
    # ボラティリティに応じて τ を動的調整。高ボラ時は τ 短縮 (素早い在庫調整)、
    # 低ボラ時は τ 延長 (緩やかな調整)。τ_eff = τ_base / vol_ratio。
    as_tau_dynamic_enabled: bool = False     # True で τ 動的化を有効化
    as_tau_dynamic_min_sec: float = 30.0     # τ 下限 (秒, 過度な短縮防止)
    as_tau_dynamic_max_sec: float = 600.0    # τ 上限 (秒, 過度な延長防止)
    # ---- 266# AS δ*: 最適スプレッド幅 (Avellaneda-Stoikov 2008 §4) ----
    # δ* = γσ²τ + (2/γ)ln(1 + γ/k) — fill rate k に基づく理論的 offset 下限。
    as_delta_star_enabled: bool = False      # True で δ* 下限を有効化
    as_delta_star_fill_rate_k: float = 1.5   # fill rate κ (注文到着強度, 要チューニング)
    # ---- 266# Kyle λ: 価格インパクト係数 (Kyle 1985) ----
    # λ_est = spread / (2·depth_volume) — 自己注文の市場インパクト → offset 安全マージン。
    kyle_lambda_enabled: bool = False        # True で Kyle λ offset 補正を有効化
    kyle_lambda_impact_mult: float = 0.5     # λ×lot を offset に反映する倍率
    kyle_lambda_max_add_ratio: float = 0.05  # offset 加算上限 (ratio 単位)
    # ---- 266# Amihud ILLIQ: 非流動性比率 (Amihud 2002) ----
    # ILLIQ = |ΔP/P| / Volume — 非流動性指標 → spread_adaptive 閾値を動的調整。
    amihud_illiq_enabled: bool = False       # True で ILLIQ 補正を有効化
    amihud_illiq_baseline: float = 0.001     # ILLIQ ベースライン (正規化基準)
    # ---- 305# Parkinson σ推定器: Parkinson (1980) High-Low Volatility Estimator ----
    # Roll proxy σ = spread/(2·mid) は薄い板で極めてノイジー。
    # Parkinson: σ_P = ln(H/L) / (2·√(ln2)) — rolling window 内の max/min mid から推定。
    # AS δ*, inv_skew, VG, Kyle λ の全段精度を向上させる。
    sigma_parkinson_enabled: bool = False     # True で Parkinson σ を使用 (False: Roll proxy)
    sigma_parkinson_window_sec: float = 300.0 # high/low 追跡ウィンドウ (秒)
    # 330# σ floor: σ=0 は AS δ*/Kyle λ/Amihud ILLIQ を完全無効化するシステミックリスク。
    # spread=0 (tight book) は AS 理論上 maker が最も vulnerable なタイミング。
    sigma_floor: float = 1e-6               # σ 推定の最小下限 (0 防止)
    # 330# vol_ratio floor: vol_ratio が極小正値の場合に σ が膨張するのを防止。
    vol_ratio_floor: float = 0.1            # RegimeDetector vol_ratio の下限
    # ---- 286# 283# P1-6: Buy-side AS Guard — microprice 急落時の buy offset 拡大 ----
    # Glosten-Milgrom (1985): 価格急落時は情報非対称性リスクが急上昇し、
    # buy 側 maker は逆選択を被りやすい。velocity が閾値を超えたら
    # buy offset を強制拡大して AS 損失を抑制する。
    amihud_illiq_max_mult: float = 1.5       # ILLIQ 由来の offset 倍率上限
    # ---- 286# 283# P1-6: Buy-side AS Guard — microprice 急落時の buy offset 拡大 ----
    # Glosten-Milgrom (1985): 価格急落時は情報非対称性リスクが急上昇し、
    # buy 側 maker は逆選択を被りやすい。velocity が閾値を超えたら
    # buy offset を強制拡大して AS 損失を抑制する。
    buy_as_guard_enabled: bool = False        # True で buy AS guard を有効化
    buy_as_guard_velocity_threshold_bps: float = -5.0  # velocity がこの値以下で発動
    buy_as_guard_offset_mult: float = 1.5     # 発動時の買い offset 倍率
    buy_as_guard_max_offset_ratio: float = 0.5  # offset 拡大の上限 ratio
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
    # 227# C3: velocity EMA smoothing — bid-ask bounce noise filter
    # α=1.0 でフィルタなし (raw velocity), α=0.3 で適度な平滑化
    velocity_ema_alpha: float = 0.3  # 344#: 1.0→0.3 (bid-ask bounce 抑制)
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
    hm_check_interval_sec: float = 60.0     # ヘルスチェック間隔 (秒)
    # ---- 158# YAML 外部化: tuning 追加 ----
    hot_reload_check_interval_sec: float = 120.0   # SkipGate モデル差替チェック間隔
    records_cache_ttl_sec: float = 10.0             # 適応エンジン キャッシュ TTL
    trades_recorder_fetch_limit: int = 100          # TradesRecorder 取得件数
    balance_freeze_cycles: int = 3                  # 残高不足 side の凍結サイクル数

    def __post_init__(self) -> None:
        """103# バリデーション: YAML 誤設定による本番クラッシュ防止.

        329# Step 2: バリデーションロジックを fill_config_validation.py に分離。
        """
        from scripts.v460.lib.fill_config_validation import validate_fill_config

        validate_fill_config(self)


    # ================================================================
    # from_yaml() — 329# Step 3: fill_config_parser.py に分離
    # ================================================================

    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> "FillTestConfig":
        """YAML dict から FillTestConfig を構築 (parser に委譲)."""
        return _resolve_fill_config_yaml_parser()(yaml_cfg)
