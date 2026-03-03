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
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    max_stale_kill_cycles: int = 10
    #: 219# progressive probe: 連続 probe 時に interval を半減する際の下限。
    min_probe_interval: int = 2
    #: 219# force release: この回数 consecutive probe が発火したら
    #: new data が来るまで kill を強制解除する。0 = 無効。
    max_force_release_probes: int = 5
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

    def __post_init__(self) -> None:
        """173# バリデーション + 241# S-4 toxicity config 制約チェック."""
        if self.window < 1:
            raise ValueError(f"DynamicKillConfig.window must be >= 1, got {self.window}")
        if self.resume_window < 0:
            raise ValueError(
                f"DynamicKillConfig.resume_window must be >= 0, got {self.resume_window}"
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

    @property
    def config(self) -> DynamicKillConfig:
        return self._config

    def track(self, pnl_bps: float) -> None:
        """fill の PnL (bps) を追跡."""
        self._pnl_history.append(pnl_bps)
        self._stale_counter = 0  # 218# 新データ投入 → stale リセット
        self._consecutive_probes = 0  # 219# 新データ → probe 連続カウンタリセット
        if self._force_released:
            self._force_released = False
            logger.info(f"[219#] {self._side} force release ended — new data received")
        # メモリ制限: 最大 window*3
        max_keep = self._config.window * 3
        if len(self._pnl_history) > max_keep:
            self._pnl_history = self._pnl_history[-max_keep:]

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
        window = self._config.window
        if len(self._pnl_history) < window:
            return False, None, len(self._pnl_history)
        recent = self._pnl_history[-window:]
        rolling_mean = sum(recent) / len(recent)
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

        window = cfg.window
        if len(self._pnl_history) < window:
            return ToxicityAssessment(
                level=ToxicityLevel.GREEN, score=0.0,
                offset_mult=1.0, participation_rate=1.0,
                threshold_used=threshold, rolling_mean=None,
            )

        recent = self._pnl_history[-window:]
        rolling_mean = sum(recent) / len(recent)

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

    def check_kill(self, regime: str | None = None) -> tuple[bool, DynamicKillTelemetry]:
        """kill すべきか判定.

        Args:
            regime: 現在のマーケットレジーム名。
                regime_thresholds にキーがあれば、その閾値を使用。

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

        # 218#/219# anti-stagnation: stale probe check + progressive interval
        effective_max_stale = self._effective_probe_interval()
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
        if len(self._pnl_history) < window:
            return False, self._make_telemetry(
                killed=False, rolling_mean=None, threshold=self._config.threshold_bps, regime=regime
            )

        recent = self._pnl_history[-window:]
        rolling_mean = sum(recent) / len(recent)

        # レジーム別閾値
        threshold = self._config.threshold_bps
        if regime and regime in self._config.regime_thresholds:
            threshold = self._config.regime_thresholds[regime]

        if rolling_mean < threshold:
            self._cooldown = self._config.resume_window
            self._total_kills += 1
            self._stale_counter += 1  # 218#
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
        self._cooldown = int(state.get("cooldown", 0))
        self._total_kills = int(state.get("total_kills", 0))
        self._total_cooldown_cycles = int(state.get("total_cooldown_cycles", 0))
        self._stale_counter = int(state.get("stale_counter", 0))
        self._total_probe_cycles = int(state.get("total_probe_cycles", 0))
        self._consecutive_probes = int(state.get("consecutive_probes", 0))
        self._force_released = bool(state.get("force_released", False))
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
