"""136# P1-03: DynamicKillManager — side 別動的 kill の抽出・強化.

run_fill_test.py 内の rolling PnL kill ロジックを
単体テスト可能なクラスとして抽出し、以下を追加:
  - テレメトリ記録 (rolling stats を JSON 返却)
  - レジーム別閾値サポート (regime_thresholds)
  - 統計情報 (kill 回数、累計 cooldown サイクル数)
  - 157# §19: side パラメータで buy/sell 共用 (DRY)

Usage:
    from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

    mgr = SellDynamicKillManager(SellKillConfig(
        window=50, threshold_bps=-0.5, resume_window=20,
    ))
    mgr.track(pnl_bps=0.3)
    killed, info = mgr.check_kill()
"""

from __future__ import annotations

import enum
import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 354# 型安全ヘルパ — import_state 用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _get_int(state: dict[str, object], key: str, default: int = 0) -> int:
    """dict[str, object] から int を安全に取り出す."""
    v = state.get(key, default)
    if isinstance(v, (int, float, str)):
        return int(v)
    return default


def _get_float(
    state: dict[str, object], key: str, default: float | None = None
) -> float | None:
    """dict[str, object] から float を安全に取り出す (None 許容)."""
    v = state.get(key, default)
    if isinstance(v, (int, float, str)):
        return float(v)
    return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 240# Toxicity Budget (232# §2.2 Glosten-Milgrom)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToxicityLevel(enum.Enum):
    """逆選択リスクの段階 (Glosten-Milgrom adverse-selection tiers).

    GREEN  = 正常: そのまま参加
    YELLOW = 警戒: スプレッドを広げて参加
    ORANGE = 要注意: 確率的に 1/N 参加 + スプレッド拡大
    KILL   = 危険: 完全停止 (従来の binary kill)
    """

    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    KILL = "kill"


@dataclass(frozen=True, slots=True)
class ToxicityAssessment:
    """Toxicity budget 評価結果 (副作用なし).

    Attributes:
        level: 4段階の逆選択リスクレベル
        score: 正規化 toxicity スコア [0, ∞) — 0=安全, 1.0=kill 閾値
        offset_mult: 推奨 offset 乗数 (1.0=通常)
        participation_rate: 推奨参加率 (1.0=全参加, 0.0=全停止)
        threshold_used: 使用された kill 閾値 (bps)
        rolling_mean: 直近 rolling PnL 平均 (bps), None=データ不足
    """

    level: ToxicityLevel
    score: float
    offset_mult: float
    participation_rate: float
    threshold_used: float
    rolling_mean: float | None

@dataclass
class DynamicKillConfig:
    """side 共用 dynamic kill 設定 (157# §19 DRY 化)."""

    enabled: bool = True
    window: int = 50
    threshold_bps: float = -0.5
    resume_window: int = 20
    #: レジーム別閾値 (レジーム名 → threshold_bps)。
    #: キーが一致すれば基本閾値の代わりに使用。
    regime_thresholds: dict[str, float] = field(default_factory=dict)
    #: 218# anti-stagnation: kill 発動中に track() が呼ばれずに
    #: この回数 check_kill() が True を返し続けたら 1 サイクルだけ許可して
    #: 新鮮な PnL データを取得するプローブサイクルを発動する。
    #: 0 = 無効 (従来互換)。
    #: 219# 30→10 に短縮 (60分→20分)。
    #:
    #: 250# 廃止検討 (247# P1-6):
    #:   probe は 242#「No Trade = 正常」思想と矛盾する。
    #:   Glosten-Milgrom の情報非対称下では、kill 中の強制取引 (probe) は
    #:   情報優位者への無防備な流動性供給となり、逆選択コストを増大させる。
    #:   242# toxic_kill_stale_multiplier で大幅緩和済みだが、
    #:   将来的に 0 (無効) をデフォルトとすることを検討。
    max_stale_kill_cycles: int = 10
    #: 219# progressive probe: 連続 probe 時に interval を半減する際の下限。
    min_probe_interval: int = 2
    #: 219# force release: この回数 consecutive probe が発火したら
    #: new data が来るまで kill を強制解除する。0 = 無効。
    max_force_release_probes: int = 5
    # ---- 242# Liveness Constraint Relaxation (233# P1) ----
    #: toxicity_budget が KILL でデータ裏付けありの場合、probe interval を
    #: この倍率で延長する。1 = 従来互換 (延長なし)。10 = 約 3.3 時間間隔。
    #: 市場理論的根拠: Glosten-Milgrom の逆選択リスクが KILL 水準のとき、
    #: probe (強制取引) は損失を追加するだけ。長期休止を正常系として許容する。
    toxic_kill_stale_multiplier: int = 10
    # ---- 240# Toxicity Budget (232# §2.2 Glosten-Milgrom) ----
    #: True で toxicity budget を有効化 (段階的応答)。
    #: False なら従来の binary kill のみ。
    toxicity_budget_enabled: bool = False
    #: YELLOW ゾーン開始点 (正規化スコア)。
    #: score = rolling_mean / threshold (0=安全, 1.0=kill)。
    #: warn_level=0.3 かつ threshold=-0.5 → rolling_mean < -0.15 で YELLOW。
    toxicity_warn_level: float = 0.3
    #: ORANGE ゾーン開始点。
    #: caution_level=0.7 かつ threshold=-0.5 → rolling_mean < -0.35 で ORANGE。
    toxicity_caution_level: float = 0.7
    #: YELLOW ゾーン入口での offset 乗数。
    #: ゾーン内で線形補間: warn_level→1.0, caution_level→caution_offset_mult。
    toxicity_warn_offset_mult: float = 1.0
    #: ORANGE ゾーン入口での offset 乗数。
    #: ゾーン内で線形補間: caution_level→caution_offset_mult, 1.0→kill_offset_mult。
    toxicity_caution_offset_mult: float = 2.0
    #: KILL 直前 (score=1.0) での offset 乗数。
    toxicity_kill_offset_mult: float = 3.0
    #: ORANGE ゾーン最悪時の最低参加率 (0.0-1.0)。
    #: ゾーン内で線形補間: caution_level→1.0, 1.0→min_participation。
    toxicity_caution_min_participation: float = 0.33
    # ---- 273# Kill Time Limit (268# I5: kill 過剰持続防止) ----
    #: kill 発動から自動解除するまでの最大秒数。0 = 無制限 (従来互換)。
    #: 268# 分析: sell_dynamic_kill が 92 分持続し、per-side halt との相互ロック
    #: (Pattern B) を引き起こした。時間上限を設けることで自動解除する。
    #: 市場理論的根拠: kill が長期化すると rolling PnL ウィンドウが stale 化し、
    #: 市場条件が変化しても反映されない。時間上限は情報の鮮度を保証する。
    max_kill_duration_sec: float = 0.0
    # ---- 344# 342#D: EWMA モード (RiskMetrics 1996) ----
    #: EWMA α ∈ (0, 1]。0.0 = 無効 (count-based rolling mean を使用)。
    #: α=0.05 で effective window ≈ 20、レジーム遷移への高速応答。
    #: 市場理論的根拠: J.P.Morgan RiskMetrics — 新しい情報ほど重要。
    ewma_alpha: float = 0.0
    # ---- 353# EWMA 時間減衰 (351# 盲点2: stale EWMA 対策) ----
    #: kill 中 (fill なし) に EWMA を中立 (0) へ指数減衰させる時定数 (秒)。
    #: 0.0 = 無効。τ=600 で半減期 ≈ 7 分 (ln2×600=416s)。
    #: 市場理論的根拠: 古い PnL 情報は時間経過で失効する (Hull 2006)。
    #: kill が情報劣化を反映せず永続する問題 (351# 盲点2) を
    #: TIME LIMIT のような hard cutoff でなく連続的減衰で解決する。
    ewma_time_decay_tau_sec: float = 0.0

    def __post_init__(self) -> None:
        """173# バリデーション + 241# S-4 toxicity config 制約チェック."""
        if self.window < 1:
            raise ValueError(f"DynamicKillConfig.window must be >= 1, got {self.window}")
        if self.resume_window < 0:
            raise ValueError(
                f"DynamicKillConfig.resume_window must be >= 0, got {self.resume_window}"
            )
        # 243# toxic_kill_stale_multiplier バリデーション (244# SR-5: >= 1 に強化)
        if self.toxic_kill_stale_multiplier < 1:
            raise ValueError(
                f"DynamicKillConfig.toxic_kill_stale_multiplier must be >= 1, "
                f"got {self.toxic_kill_stale_multiplier}"
            )
        # 241# S-4: toxicity budget 設定バリデーション
        if self.toxicity_budget_enabled:
            if not (0.0 <= self.toxicity_warn_level < self.toxicity_caution_level <= 1.0):
                raise ValueError(
                    f"DynamicKillConfig: must satisfy "
                    f"0 <= warn_level < caution_level <= 1.0, "
                    f"got warn={self.toxicity_warn_level}, "
                    f"caution={self.toxicity_caution_level}"
                )
            if self.toxicity_warn_offset_mult < 1.0:
                raise ValueError(
                    f"DynamicKillConfig.toxicity_warn_offset_mult must be >= 1.0, "
                    f"got {self.toxicity_warn_offset_mult}"
                )
            if self.toxicity_caution_offset_mult < self.toxicity_warn_offset_mult:
                raise ValueError(
                    f"DynamicKillConfig.toxicity_caution_offset_mult must be >= "
                    f"warn_offset_mult ({self.toxicity_warn_offset_mult}), "
                    f"got {self.toxicity_caution_offset_mult}"
                )
            if self.toxicity_kill_offset_mult < self.toxicity_caution_offset_mult:
                raise ValueError(
                    f"DynamicKillConfig.toxicity_kill_offset_mult must be >= "
                    f"caution_offset_mult ({self.toxicity_caution_offset_mult}), "
                    f"got {self.toxicity_kill_offset_mult}"
                )
            if not (0.0 < self.toxicity_caution_min_participation <= 1.0):
                raise ValueError(
                    f"DynamicKillConfig.toxicity_caution_min_participation "
                    f"must be in (0, 1.0], got {self.toxicity_caution_min_participation}"
                )
        # 344# 342#D: EWMA α バリデーション
        if not (0.0 <= self.ewma_alpha <= 1.0):
            raise ValueError(
                f"DynamicKillConfig.ewma_alpha must be in [0, 1], "
                f"got {self.ewma_alpha}"
            )
        # 353# EWMA time decay τ バリデーション
        if self.ewma_time_decay_tau_sec < 0:
            raise ValueError(
                f"DynamicKillConfig.ewma_time_decay_tau_sec must be >= 0, "
                f"got {self.ewma_time_decay_tau_sec}"
            )

# 後方互換エイリアス
SellKillConfig = DynamicKillConfig

@dataclass
class DynamicKillTelemetry:
    """テレメトリ情報."""

    killed: bool
    cooldown_remaining: int
    rolling_mean: float | None
    rolling_count: int
    threshold_used: float
    regime: str | None
    total_kills: int
    total_cooldown_cycles: int
    side: str = ""
    probe_fired: bool = False       # 223# probe 発動フラグ
    force_release_fired: bool = False  # 223# force release 発動フラグ

# 後方互換エイリアス
SellKillTelemetry = DynamicKillTelemetry

class DynamicKillManager:
    """side 共用動的 kill マネージャ (157# §19).

    133# P0-10 のロジックを独立クラスに抽出し、
    テレメトリ + レジーム別閾値をサポート。
    side パラメータで buy/sell 両方に対応 (DRY)。
    """

    __slots__ = (
        "_config",
        "_pnl_history",
        "_cooldown",
        "_total_kills",
        "_total_cooldown_cycles",
        "_side",
        "_stale_counter",
        "_total_probe_cycles",
        "_consecutive_probes",  # 219#
        "_force_released",       # 219#
        "_kill_activated_at",    # 273# kill 発動タイムスタンプ
        "_ewma_value",           # 344# 342#D EWMA 状態
        "_last_track_time",      # 353# EWMA time decay 基準時刻
    )

    def __init__(self, config: DynamicKillConfig | None = None, *, side: str = "sell") -> None:
        self._config = config or DynamicKillConfig()
        self._pnl_history: list[float] = []
        self._cooldown: int = 0
        self._total_kills: int = 0
        self._total_cooldown_cycles: int = 0
        self._side = side
        self._stale_counter: int = 0  # 218# anti-stagnation
        self._total_probe_cycles: int = 0
        self._consecutive_probes: int = 0  # 219# progressive probe
        self._force_released: bool = False  # 219# force release active
        self._kill_activated_at: float | None = None  # 273# kill 発動時刻
        self._ewma_value: float | None = None  # 344# 342#D EWMA 累積値
        self._last_track_time: float | None = None  # 353# 最終 track() 時刻

    @property
    def config(self) -> DynamicKillConfig:
        return self._config

    def track(self, pnl_bps: float) -> None:
        """fill の PnL (bps) を追跡."""
        self._pnl_history.append(pnl_bps)
        # 344# 342#D: EWMA 更新
        # 352#: 初回シードは当該値そのもの (track() と _rebuild_ewma_from_history() の整合性確保)
        # (349# P1 の history 平均シードは 350# F1 / 351# Q1 で look-ahead bias と指摘 → 撤回)
        alpha = self._config.ewma_alpha
        if alpha > 0:
            if self._ewma_value is None:
                self._ewma_value = pnl_bps
            else:
                self._ewma_value = alpha * pnl_bps + (1.0 - alpha) * self._ewma_value
        self._last_track_time = time.time()  # 353# 最終 track 時刻更新
        self._stale_counter = 0  # 218# 新データ投入 → stale リセット
        self._consecutive_probes = 0  # 219# 新データ → probe 連続カウンタリセット
        if self._force_released:
            self._force_released = False
            logger.info(f"[219#] {self._side} force release ended — new data received")
        # 273# 新データ投入 → kill timestamp リセット
        # (新しい PnL が入ったので kill 判定は再評価される)
        self._kill_activated_at = None
        # メモリ制限: 最大 window*3
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

    def _rebuild_ewma_from_history(self) -> None:
        """349# P0: pnl_history から EWMA 値を再構築.

        import_state() で ewma_value が欠落している場合のフォールバック。
        352#: history[0] を seed とし history[1:] を replay する厳密式に修正。
        (350# F1 / 351# Q1: 平均 seed + 全履歴 replay は look-ahead bias / double counting)
        """
        alpha = self._config.ewma_alpha
        if alpha <= 0 or not self._pnl_history:
            self._ewma_value = None
            return
        # 352#: track() と同一の更新式で厳密に再構築
        ewma = self._pnl_history[0]  # 初回 seed = 最初の観測値
        for v in self._pnl_history[1:]:
            ewma = alpha * v + (1.0 - alpha) * ewma
        self._ewma_value = ewma
        logger.info(
            f"[352# P0] {self._side} EWMA rebuilt from {len(self._pnl_history)} records: "
            f"ewma={ewma:.4f}"
        )

    def _get_rolling_mean(self) -> float | None:
        """344# 342#D: EWMA or count-based rolling mean を返す.

        EWMA モード (ewma_alpha > 0): _ewma_value を返す (window 未満でも可)。
        Count-based モード (ewma_alpha == 0): 従来の直近 window 件の算術平均。
        データ不足時は None。

        353# EWMA 時間減衰: ewma_time_decay_tau_sec > 0 の場合、
        最終 track() からの経過時間で EWMA を 0 方向へ指数減衰させる。
        kill 中に fill が来ず EWMA が stale 化する問題 (351# 盲点2) を解決。
        _ewma_value 自体は変更せず、返値のみ減衰 (副作用なし)。
        """
        if self._config.ewma_alpha > 0:
            ewma = self._ewma_value
            # 353# time decay: kill 中 (fill なし) に EWMA を中立方向へ減衰
            tau = self._config.ewma_time_decay_tau_sec
            if (
                ewma is not None
                and tau > 0
                and self._last_track_time is not None
            ):
                elapsed = time.time() - self._last_track_time
                if elapsed > 0:
                    ewma = ewma * math.exp(-elapsed / tau)
            return ewma
        window = self._config.window
        if len(self._pnl_history) < window:
            return None
        recent = self._pnl_history[-window:]
        return sum(recent) / len(recent)

    def is_kill_active(self) -> tuple[bool, float | None, int]:
        """224# B2: 副作用なしで kill 状態を検査.

        check_kill() は cooldown デクリメント等の副作用があるため、
        日替わり境界など副作用を避けたい場面で使用する。

        Returns:
            (is_active, rolling_mean, rolling_count)
        """
        if not self._config.enabled or self._force_released:
            return False, None, len(self._pnl_history)
        if self._cooldown > 0:
            return True, None, len(self._pnl_history)
        rolling_mean = self._get_rolling_mean()
        if rolling_mean is None:
            return False, None, len(self._pnl_history)
        threshold = self._config.threshold_bps
        return rolling_mean < threshold, rolling_mean, len(self._pnl_history)

    def assess_toxicity(self, regime: str | None = None) -> ToxicityAssessment:
        """240# 副作用なしで toxicity budget を評価.

        Glosten-Milgrom の逆選択リスクを正規化スコアで定量化し、
        4 段階 (GREEN/YELLOW/ORANGE/KILL) にマッピングする。

        スコア = max(0, rolling_mean / threshold) (threshold < 0 の場合)
        - 0.0: 完全に安全
        - warn_level (0.3): YELLOW ゾーン開始 → offset 拡大
        - caution_level (0.7): ORANGE ゾーン開始 → 確率的参加
        - 1.0: KILL 閾値到達

        Args:
            regime: レジーム名 (regime_thresholds 参照用)

        Returns:
            ToxicityAssessment (immutable, 副作用なし)
        """
        cfg = self._config

        # 閾値決定 (レジーム別)
        threshold = cfg.threshold_bps
        if regime and regime in cfg.regime_thresholds:
            threshold = cfg.regime_thresholds[regime]

        # データ不足 or 無効 or force release → GREEN
        if (
            not cfg.enabled
            or not cfg.toxicity_budget_enabled
            or self._force_released
        ):
            return ToxicityAssessment(
                level=ToxicityLevel.GREEN, score=0.0,
                offset_mult=1.0, participation_rate=1.0,
                threshold_used=threshold, rolling_mean=None,
            )

        rolling_mean = self._get_rolling_mean()
        if rolling_mean is None:
            return ToxicityAssessment(
                level=ToxicityLevel.GREEN, score=0.0,
                offset_mult=1.0, participation_rate=1.0,
                threshold_used=threshold, rolling_mean=None,
            )

        # 正規化スコア: 0=安全, 1.0=kill 閾値
        # threshold は負なので rolling_mean/threshold → 正の値が危険
        if threshold >= 0:
            # threshold が 0 以上 (異常設定) → 安全扱い
            score = 0.0
        else:
            score = max(0.0, rolling_mean / threshold)

        # cooldown 中は kill 確定
        if self._cooldown > 0:
            return ToxicityAssessment(
                level=ToxicityLevel.KILL, score=max(score, 1.0),
                offset_mult=cfg.toxicity_kill_offset_mult,
                participation_rate=0.0,
                threshold_used=threshold, rolling_mean=rolling_mean,
            )

        # 段階判定
        warn = cfg.toxicity_warn_level
        caution = cfg.toxicity_caution_level

        if score >= 1.0:
            # KILL ゾーン
            return ToxicityAssessment(
                level=ToxicityLevel.KILL, score=score,
                offset_mult=cfg.toxicity_kill_offset_mult,
                participation_rate=0.0,
                threshold_used=threshold, rolling_mean=rolling_mean,
            )
        elif score >= caution:
            # ORANGE ゾーン: 確率的参加 + offset 拡大
            # 線形補間: caution → 1.0
            t = (score - caution) / (1.0 - caution) if caution < 1.0 else 0.0
            offset_m = cfg.toxicity_caution_offset_mult + t * (
                cfg.toxicity_kill_offset_mult - cfg.toxicity_caution_offset_mult
            )
            participation = 1.0 - t * (1.0 - cfg.toxicity_caution_min_participation)
            return ToxicityAssessment(
                level=ToxicityLevel.ORANGE, score=score,
                offset_mult=offset_m,
                participation_rate=participation,
                threshold_used=threshold, rolling_mean=rolling_mean,
            )
        elif score >= warn:
            # YELLOW ゾーン: offset 拡大のみ
            # 線形補間: warn → caution
            t = (score - warn) / (caution - warn) if caution > warn else 0.0
            offset_m = cfg.toxicity_warn_offset_mult + t * (
                cfg.toxicity_caution_offset_mult - cfg.toxicity_warn_offset_mult
            )
            return ToxicityAssessment(
                level=ToxicityLevel.YELLOW, score=score,
                offset_mult=offset_m,
                participation_rate=1.0,
                threshold_used=threshold, rolling_mean=rolling_mean,
            )
        else:
            # GREEN ゾーン: 通常
            return ToxicityAssessment(
                level=ToxicityLevel.GREEN, score=score,
                offset_mult=1.0, participation_rate=1.0,
                threshold_used=threshold, rolling_mean=rolling_mean,
            )

    def check_kill(self, regime: str | None = None, *, threshold_offset_bps: float = 0.0) -> tuple[bool, DynamicKillTelemetry]:
        """kill すべきか判定.

        Args:
            regime: 現在のマーケットレジーム名。
                regime_thresholds にキーがあれば、その閾値を使用。
            threshold_offset_bps: 286# 283# P1-4: 在庫連動の閾値調整 (bps)。
                正値 → 緩和 (閾値をより負側へ、kill されにくくなる)、負値 → 厳格化。
                Ho & Stoll (1981) 在庫リスクモデルに基づき、在庫偏重度に
                応じて動的 kill 閾値を段階的に緩和する。

        Returns:
            (killed, telemetry)
        """
        if not self._config.enabled:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        # 219# force release: 連続 probe 超過で強制解除中
        if self._force_released:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        # 273# Kill Time Limit (268# I5 / Pattern B): 時間上限による自動解除
        # kill が max_kill_duration_sec 以上持続した場合、rolling PnL が改善して
        # いなくても強制解除する。halt 中に新データが入らず kill が永続する
        # (268# 92分持続) パターンを防止。
        _max_dur = self._config.max_kill_duration_sec
        if (
            _max_dur > 0
            and self._kill_activated_at is not None
            and (time.time() - self._kill_activated_at) >= _max_dur
        ):
            # 349# P2: TIME LIMIT 解除時に EWMA を decay して即再 kill を防止
            # 根拠: kill 中は新データが入らず EWMA が stale 化している。
            # threshold 付近まで decay させることで、次の fill で改善の余地を与える。
            # 352#: effective_threshold (regime + inv_relaxation) を使用
            #   (351# Q2: base threshold のみだと inv_relaxation 適用後にリセット先が
            #    kill 圏内に入り TIME LIMIT 即再 kill のデッドロック復活リスク)
            old_ewma = self._ewma_value
            if self._config.ewma_alpha > 0 and self._ewma_value is not None:
                effective_threshold = self._config.threshold_bps
                if regime and regime in self._config.regime_thresholds:
                    effective_threshold = self._config.regime_thresholds[regime]
                # 在庫連動緩和も反映 (check_kill と同一のロジック)
                if threshold_offset_bps != 0.0:
                    effective_threshold -= threshold_offset_bps
                # EWMA を effective_threshold * 0.8 にリセット (kill 閾値の少し上)
                # これにより即再 kill は避けつつ、次の悪い fill では再 kill される
                reset_target = effective_threshold * 0.8
                self._ewma_value = reset_target
                logger.info(
                    f"[352# P2] {self._side} EWMA decay on TIME LIMIT: "
                    f"{old_ewma:.3f} → {reset_target:.3f}bps "
                    f"(effective_threshold={effective_threshold}, offset={threshold_offset_bps})"
                )
            logger.warning(
                f"[273#] {self._side} kill TIME LIMIT expired: "
                f"active for {time.time() - self._kill_activated_at:.0f}s "
                f"(limit={_max_dur:.0f}s) — auto-releasing"
            )
            self._cooldown = 0
            self._kill_activated_at = None
            self._stale_counter = 0
            return False, self._make_telemetry(
                killed=False, rolling_mean=None,
                threshold=self._config.threshold_bps, regime=regime,
            )

        # 218#/219# anti-stagnation: stale probe check + progressive interval
        # 242# Liveness relaxation: toxicity KILL + データ裏付けあり → interval 延長
        # 250# 廃止検討注記 (247# P1-6):
        #   probe は逆選択環境で情報劣位者が穴を開ける行為であり、
        #   MM 理論 (Glosten-Milgrom 1985) 上は「No Quote」が最適解。
        #   242# toxic_kill_stale_multiplier で実質的に無効化されるが、
        #   max_stale_kill_cycles=0 で明示的に無効化可能。
        effective_max_stale = self._effective_probe_interval()
        _toxic_mult = self._toxic_kill_multiplier(regime)
        if _toxic_mult > 1:
            effective_max_stale *= _toxic_mult
        if effective_max_stale > 0 and self._stale_counter >= effective_max_stale:
            self._stale_counter = 0
            self._total_probe_cycles += 1
            self._consecutive_probes += 1  # 219#

            # 219# force release 判定
            max_fr = self._config.max_force_release_probes
            if max_fr > 0 and self._consecutive_probes >= max_fr:
                self._force_released = True
                logger.warning(
                    f"[219#] {self._side} FORCE RELEASE: "
                    f"{self._consecutive_probes} consecutive probes without recovery — "
                    f"kill disabled until new data (total_probes={self._total_probe_cycles})"
                )
                self._cooldown = 0
                _telem = self._make_telemetry(
                    killed=False, rolling_mean=None,
                    threshold=self._config.threshold_bps, regime=regime,
                )
                _telem.force_release_fired = True  # 223#
                return False, _telem

            logger.warning(
                f"[219#] {self._side} dynamic kill probe: "
                f"stale for {effective_max_stale} cycles — "
                f"allowing 1 probe cycle "
                f"(consecutive={self._consecutive_probes}, "
                f"total_probes={self._total_probe_cycles})"
            )
            self._cooldown = 0  # cooldown もリセット
            _telem = self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )
            _telem.probe_fired = True  # 223#
            return False, _telem

        # cooldown 中
        if self._cooldown > 0:
            self._cooldown -= 1
            self._total_cooldown_cycles += 1
            self._stale_counter += 1  # 218#
            return True, self._make_telemetry(
                killed=True, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        window = self._config.window
        rolling_mean = self._get_rolling_mean()
        if rolling_mean is None:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        # レジーム別閾値
        threshold = self._config.threshold_bps
        if regime and regime in self._config.regime_thresholds:
            threshold = self._config.regime_thresholds[regime]

        # 286# 283# P1-4: 在庫連動の閾値緩和 (Ho & Stoll 1981)
        # 340# 符号修正: threshold_offset_bps > 0 は閾値をより負方向に動かす (kill されにくくなる)
        # kill 条件: rolling_mean < threshold
        # 例: threshold=-1.0, offset=+0.3 → effective=-1.3 (緩和: より深い損失まで許容)
        if threshold_offset_bps != 0.0:
            threshold -= threshold_offset_bps

        if rolling_mean < threshold:
            self._cooldown = self._config.resume_window
            self._total_kills += 1
            self._stale_counter += 1  # 218#
            # 273# kill 発動時刻を記録 (既存 kill の延長でなければ)
            if self._kill_activated_at is None:
                self._kill_activated_at = time.time()
            logger.warning(
                f"[157# §19] {self._side} dynamic kill activated: "
                f"rolling{window} mean={rolling_mean:.3f}bps < {threshold}bps, "
                f"regime={regime or 'default'}, "
                f"cooldown={self._config.resume_window}, "
                f"total_kills={self._total_kills}"
            )
            return True, self._make_telemetry(
                killed=True, rolling_mean=rolling_mean, threshold=threshold, regime=regime
            )

        return False, self._make_telemetry(
            killed=False, rolling_mean=rolling_mean, threshold=threshold, regime=regime
        )

    def _make_telemetry(
        self,
        killed: bool,
        rolling_mean: float | None,
        threshold: float,
        regime: str | None,
    ) -> DynamicKillTelemetry:
        return DynamicKillTelemetry(
            killed=killed,
            cooldown_remaining=self._cooldown,
            rolling_mean=rolling_mean,
            rolling_count=len(self._pnl_history),
            threshold_used=threshold,
            regime=regime,
            total_kills=self._total_kills,
            total_cooldown_cycles=self._total_cooldown_cycles,
            side=self._side,
        )

    def _effective_probe_interval(self) -> int:
        """219# progressive probe: 連続 probe 回数に応じて interval を半減.

        Returns:
            現在の effective max_stale_kill_cycles。
            0 なら probe 無効。
        """
        base = self._config.max_stale_kill_cycles
        if base <= 0 or self._consecutive_probes <= 0:
            return base
        # 各 consecutive probe で半減: 10 → 5 → 3 → 2 → 2
        interval = base
        for _ in range(self._consecutive_probes):
            interval = max(self._config.min_probe_interval, (interval + 1) // 2)
        return interval

    def _toxic_kill_multiplier(self, regime: str | None) -> int:
        """242# toxicity KILL + データ裏付けありなら probe interval 延長倍率を返す.

        Glosten-Milgrom の逆選択リスクが KILL 水準のとき、probe (強制取引) は
        損失を追加するだけ。長期休止 (No Trade) を正常系として許容する。

        Returns:
            toxic_kill_stale_multiplier (データ裏付け KILL 時) or 1 (通常時)
        """
        mult = self._config.toxic_kill_stale_multiplier
        if mult <= 1:
            return 1
        if not self._config.toxicity_budget_enabled:
            return 1
        tox = self.assess_toxicity(regime=regime)
        # rolling_mean is not None = 十分なデータがある → kill は data-backed
        if tox.level == ToxicityLevel.KILL and tox.rolling_mean is not None:
            return mult
        return 1

    def reset(self) -> None:
        """状態リセット."""
        self._pnl_history.clear()
        self._cooldown = 0
        self._total_kills = 0
        self._total_cooldown_cycles = 0
        self._stale_counter = 0
        self._total_probe_cycles = 0
        self._consecutive_probes = 0
        self._force_released = False
        self._kill_activated_at = None  # 273#
        self._ewma_value = None  # 349# P0

    # ------------------------------------------------------------------
    # 209# H4: 状態永続化 — export / import
    # ------------------------------------------------------------------
    def export_state(self) -> dict[str, object]:
        """永続化用に内部状態を dict にエクスポート.

        Returns:
            pnl_history, cooldown, total_kills, total_cooldown_cycles, side を含む dict。
        """
        return {
            "pnl_history": list(self._pnl_history),
            "cooldown": self._cooldown,
            "total_kills": self._total_kills,
            "total_cooldown_cycles": self._total_cooldown_cycles,
            "side": self._side,
            "stale_counter": self._stale_counter,
            "total_probe_cycles": self._total_probe_cycles,
            "consecutive_probes": self._consecutive_probes,
            "force_released": self._force_released,
            "kill_activated_at": self._kill_activated_at,  # 273#
            "ewma_value": self._ewma_value,  # 349# P0: EWMA 状態永続化
            "last_track_time": self._last_track_time,  # 353# EWMA time decay
        }

    def import_state(self, state: dict[str, object]) -> None:
        """export_state() で保存した dict から内部状態を復元.

        Args:
            state: export_state() の戻り値。キーが欠落している場合は
                   デフォルト値 (空リスト / 0) にフォールバック。
        """
        raw_history = state.get("pnl_history", [])
        if isinstance(raw_history, list):
            self._pnl_history = [float(v) for v in raw_history]
        else:
            self._pnl_history = []
        self._cooldown = _get_int(state, "cooldown")
        self._total_kills = _get_int(state, "total_kills")
        self._total_cooldown_cycles = _get_int(state, "total_cooldown_cycles")
        self._stale_counter = _get_int(state, "stale_counter")
        self._total_probe_cycles = _get_int(state, "total_probe_cycles")
        self._consecutive_probes = _get_int(state, "consecutive_probes")
        self._force_released = bool(state.get("force_released", False))
        # 273# kill_activated_at 復元
        self._kill_activated_at = _get_float(state, "kill_activated_at")
        # 353# last_track_time 復元
        self._last_track_time = _get_float(state, "last_track_time")
        # 349# P0: EWMA 状態復元 — 欠落している場合は history から再構築
        _ewma = _get_float(state, "ewma_value")
        if _ewma is not None:
            self._ewma_value = _ewma
        elif self._config.ewma_alpha > 0 and self._pnl_history:
            self._rebuild_ewma_from_history()
        else:
            self._ewma_value = None
        # side は import しない (コンストラクタで固定)
        # メモリ制限: window*3 に収める
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

    @property
    def side(self) -> str:
        """管理対象 side."""
        return self._side

# 後方互換エイリアス (136# の import を破壊しない)
SellDynamicKillManager = DynamicKillManager

class BuyDynamicKillManager(DynamicKillManager):
    """157# §19: buy 側動的 kill マネージャ.

    DynamicKillManager(side="buy") のコンビニエンスサブクラス。
    """

    def __init__(self, config: DynamicKillConfig | None = None) -> None:
        super().__init__(config, side="buy")
