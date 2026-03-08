"""
PnL モンテカルロシミュレータ — fill_test 実測データから月次 PnL 信頼区間を推定.

014# T5: 012# §3 #4「PnL モンテカルロ化」の実装.

実測した fill_records (JSONL) からスプレッド・adverse selection・fill rate の
経験分布を構築し、n=10,000 の月次シミュレーションで PnL レンジを推定する。

設計ポイント:
  - n が少なくても動作する (bootstrap resampling)
  - n が増えれば精度が自動的に向上・分布も安定する
  - 000# §3.3 の G1.1 判定指標メトリクスを同時出力
  - 時間帯別・レジーム別の層化サンプリングに拡張可能 (将来)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 027# 型統合: FillRecord は fill_quality の単一定義を使用
from ztb.metrics.fill_quality import (
    FillRecord,  # noqa: F401 (re-export)
    FillMetrics,
    compute_fill_metrics,
)

logger = logging.getLogger(__name__)

_VECTORIZED_MC_MAX_SAMPLES = 8_000_000

@dataclass
class MonteCarloConfig:
    """シミュレーション設定."""
    n_simulations: int = 10_000
    cycles_per_day: int = 720      # 120s 間隔 = 1日 720 サイクル
    days_per_month: int = 30
    lot_size_btc: float = 0.001    # BTC per cycle
    btc_price_jpy: float = 10_300_000.0  # approx
    maker_fee_rate: float = 0.0    # Coincheck maker fee = 0%
    random_seed: int = 42
    confidence_levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

@dataclass
class MonteCarloResult:
    """シミュレーション結果."""
    # Input summary
    n_records: int
    n_filled: int
    n_cancelled: int
    observed_fill_rate: float
    observed_pnl_mean_bps: float
    observed_pnl_std_bps: float
    observed_as_ratio: float

    # Simulation summary
    n_simulations: int
    cycles_per_month: int

    # Monthly PnL distribution (JPY)
    pnl_mean_jpy: float
    pnl_std_jpy: float
    pnl_percentiles_jpy: dict[str, float]  # "5%": xxx, "25%": xxx, ...

    # Monthly PnL distribution (bps, per-cycle average)
    pnl_mean_bps: float
    pnl_std_bps: float

    # Risk metrics
    var_95_jpy: float             # 95% VaR (neg = loss)
    cvar_95_jpy: float            # Conditional VaR
    prob_loss: float              # P(monthly PnL < 0)
    prob_profit: float            # P(monthly PnL > 0)

    # G1.1 criteria pass/fail (000# §3.3)
    g11_fill_rate: float
    g11_cancel_ratio: float
    g11_queue_wait_median: float
    g11_pnl_mean_bps: float
    g11_as_ratio: float
    g11_pass: bool

    # Break-even analysis
    breakeven_fill_rate: float    # fill rate needed for PnL ≥ 0
    breakeven_pnl_bps: float     # per-cycle bps needed for PnL ≥ 0

    raw_monthly_pnls: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict に変換 (027# 型統合: MonteCarloResult 自身に to_dict 追加)."""
        return {
            "n_records": self.n_records,
            "n_filled": self.n_filled,
            "n_cancelled": self.n_cancelled,
            "observed_fill_rate": self.observed_fill_rate,
            "observed_pnl_mean_bps": self.observed_pnl_mean_bps,
            "observed_pnl_std_bps": self.observed_pnl_std_bps,
            "observed_as_ratio": self.observed_as_ratio,
            "n_simulations": self.n_simulations,
            "cycles_per_month": self.cycles_per_month,
            "pnl_mean_jpy": self.pnl_mean_jpy,
            "pnl_std_jpy": self.pnl_std_jpy,
            "pnl_percentiles_jpy": self.pnl_percentiles_jpy,
            "pnl_mean_bps": self.pnl_mean_bps,
            "pnl_std_bps": self.pnl_std_bps,
            "var_95_jpy": self.var_95_jpy,
            "cvar_95_jpy": self.cvar_95_jpy,
            "prob_loss": self.prob_loss,
            "prob_profit": self.prob_profit,
            "g11_fill_rate": self.g11_fill_rate,
            "g11_cancel_ratio": self.g11_cancel_ratio,
            "g11_queue_wait_median": self.g11_queue_wait_median,
            "g11_pnl_mean_bps": self.g11_pnl_mean_bps,
            "g11_as_ratio": self.g11_as_ratio,
            "g11_pass": self.g11_pass,
            "breakeven_fill_rate": self.breakeven_fill_rate,
            "breakeven_pnl_bps": self.breakeven_pnl_bps,
        }

# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class PnLMonteCarloSimulator:
    """fill_records から月次 PnL 分布を推定するモンテカルロシミュレータ.

    使用例::

        records = PnLMonteCarloSimulator.load_fill_records("results/v460/fill_test/")
        sim = PnLMonteCarloSimulator(records)
        result = sim.run()
        sim.print_report(result)
    """

    def __init__(
        self,
        records: Sequence[FillRecord],
        config: MonteCarloConfig | None = None,
    ) -> None:
        if not records:
            raise ValueError("No fill records provided")
        self.records = list(records)
        self.config = config or MonteCarloConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_fill_records(
        path: str | Path,
        glob_pattern: str = "fill_records_*.jsonl",
    ) -> list[FillRecord]:
        """Load fill records from JSONL file(s) in a directory or single file.

        027# 型統合: fill_quality.load_fill_records / load_fill_records_glob に委譲.
        """
        from ztb.metrics.fill_quality import load_fill_records, load_fill_records_glob

        p = Path(path)
        if p.is_file():
            return load_fill_records(p)
        elif p.is_dir():
            return load_fill_records_glob(p)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _extract_filled_pnl_bps(self) -> np.ndarray:
        """約定済みレコードから PnL(bps) 配列を抽出する."""
        pnl_values = [
            float(r.post_fill_30s_pnl)
            for r in self.records
            if r.filled and r.post_fill_30s_pnl is not None
        ]
        if not pnl_values:
            return np.array([0.0], dtype=np.float64)
        return np.asarray(pnl_values, dtype=np.float64)

    def _simulate_monthly_pnls(
        self,
        *,
        fill_rate: float,
        pnl_bps: np.ndarray,
        cycles_per_month: int,
        jpy_per_bps: float,
    ) -> np.ndarray:
        """1条件分の月次PnL分布をシミュレーションする."""
        monthly_pnls = np.zeros(self.config.n_simulations, dtype=np.float64)
        fills_per_sim = self._rng.binomial(
            cycles_per_month,
            fill_rate,
            size=self.config.n_simulations,
        )
        # PnL 値が定数のときは sampling を省いて高速化できる。
        if pnl_bps.size == 1:
            return fills_per_sim.astype(np.float64) * float(pnl_bps[0]) * jpy_per_bps
        max_fills = int(fills_per_sim.max(initial=0))
        if (
            max_fills > 0
            and self.config.n_simulations * max_fills <= _VECTORIZED_MC_MAX_SAMPLES
        ):
            sampled = self._rng.choice(
                pnl_bps,
                size=(self.config.n_simulations, max_fills),
                replace=True,
            )
            sampled_cumsum = np.cumsum(sampled, axis=1, dtype=np.float64)
            positive = fills_per_sim > 0
            if np.any(positive):
                fill_idx = fills_per_sim[positive] - 1
                monthly_pnls[positive] = sampled_cumsum[positive, fill_idx]
            return monthly_pnls * jpy_per_bps

        for i, fills_this_month in enumerate(fills_per_sim):
            fills_this_month = int(fills_this_month)
            if fills_this_month <= 0:
                continue
            sampled_pnl = self._rng.choice(pnl_bps, size=fills_this_month, replace=True)
            monthly_pnls[i] = float(np.sum(sampled_pnl * jpy_per_bps))
        return monthly_pnls

    def _sample_monthly_pnl_bps(
        self,
        *,
        fill_rate: float,
        pnl_bps: np.ndarray,
        cycles_per_month: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """月次 PnL(bps) と fill count を同時にサンプリングする."""
        fills_per_sim = self._rng.binomial(
            cycles_per_month,
            fill_rate,
            size=self.config.n_simulations,
        )
        monthly_pnl_bps = np.zeros(self.config.n_simulations, dtype=np.float64)
        if pnl_bps.size == 1:
            monthly_pnl_bps = fills_per_sim.astype(np.float64) * float(pnl_bps[0])
            return monthly_pnl_bps, fills_per_sim

        max_fills = int(fills_per_sim.max(initial=0))
        if (
            max_fills > 0
            and self.config.n_simulations * max_fills <= _VECTORIZED_MC_MAX_SAMPLES
        ):
            sampled = self._rng.choice(
                pnl_bps,
                size=(self.config.n_simulations, max_fills),
                replace=True,
            )
            sampled_cumsum = np.cumsum(sampled, axis=1, dtype=np.float64)
            positive = fills_per_sim > 0
            if np.any(positive):
                fill_idx = fills_per_sim[positive] - 1
                monthly_pnl_bps[positive] = sampled_cumsum[positive, fill_idx]
            return monthly_pnl_bps, fills_per_sim

        for i, fills_this_month in enumerate(fills_per_sim):
            fills_this_month = int(fills_this_month)
            if fills_this_month <= 0:
                continue
            sampled_pnl = self._rng.choice(pnl_bps, size=fills_this_month, replace=True)
            monthly_pnl_bps[i] = float(np.sum(sampled_pnl))
        return monthly_pnl_bps, fills_per_sim

    @staticmethod
    def _compute_breakeven(
        *,
        fill_rate: float,
        pnl_mean_bps: float,
    ) -> tuple[float, float]:
        """損益分岐の補助指標を返す."""
        breakeven_pnl_bps = 0.0
        if pnl_mean_bps > 0:
            breakeven_fill_rate = 0.0
        elif pnl_mean_bps < 0:
            breakeven_fill_rate = 1.0
        else:
            breakeven_fill_rate = fill_rate
        return breakeven_fill_rate, breakeven_pnl_bps

    @staticmethod
    def _passes_g11(
        *,
        fill_rate: float,
        cancel_ratio: float,
        queue_wait_median: float,
        pnl_mean_bps: float,
        as_ratio: float,
    ) -> bool:
        """G1.1 条件の PASS/FAIL を返す."""
        return (
            fill_rate >= 0.90
            and cancel_ratio <= 0.30
            and queue_wait_median <= 60.0
            and pnl_mean_bps >= 0.0
            and as_ratio <= 0.20
        )

    def run(self) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        cfg = self.config

        # --- Observed statistics (fill_quality.compute_fill_metrics に委譲) ---
        metrics = compute_fill_metrics(self.records)
        n_total = metrics.total_orders
        n_filled = metrics.filled_orders
        n_cancelled = metrics.cancelled_orders
        fill_rate = n_filled / n_total if n_total > 0 else 0.0
        cancel_ratio = metrics.cancel_ratio
        pnl_mean = metrics.post_fill_30s_pnl_mean
        as_ratio = metrics.adverse_selection_ratio
        queue_wait_median = metrics.queue_wait_median_sec

        # PnL distribution (bps) — bootstrap sampling 用の生配列
        pnl_bps = self._extract_filled_pnl_bps()

        pnl_std = float(np.std(pnl_bps, ddof=1)) if len(pnl_bps) > 1 else 0.0

        # --- Monte Carlo ---
        cycles_per_month = cfg.cycles_per_day * cfg.days_per_month
        notional = cfg.lot_size_btc * cfg.btc_price_jpy  # JPY per cycle
        jpy_per_bps = 1e-4 * notional
        monthly_pnls = self._simulate_monthly_pnls(
            fill_rate=fill_rate,
            pnl_bps=pnl_bps,
            cycles_per_month=cycles_per_month,
            jpy_per_bps=jpy_per_bps,
        )

        # --- Percentiles ---
        pct_levels = [p * 100 for p in cfg.confidence_levels]
        pct_values = np.percentile(monthly_pnls, pct_levels)
        percentiles = {
            f"{p * 100:.0f}%": float(v)
            for p, v in zip(cfg.confidence_levels, pct_values)
        }

        # VaR / CVaR
        var_95 = float(np.percentile(monthly_pnls, 5))  # 5th percentile (loss side)
        cvar_95 = float(np.mean(monthly_pnls[monthly_pnls <= var_95])) if np.any(monthly_pnls <= var_95) else var_95

        # P(loss) / P(profit)
        prob_loss = float(np.mean(monthly_pnls < 0))
        prob_profit = float(np.mean(monthly_pnls > 0))

        # --- Break-even analysis ---
        # What fill_rate would make expected PnL = 0?
        # E[monthly PnL] = cycles * fill_rate * E[pnl_bps] * 1e-4 * notional
        # = 0  when fill_rate = 0 or E[pnl_bps] = 0
        breakeven_fill_rate, breakeven_pnl_bps = self._compute_breakeven(
            fill_rate=fill_rate,
            pnl_mean_bps=pnl_mean,
        )

        # --- G1.1 pass/fail (000# §3.3) ---
        g11_pass = self._passes_g11(
            fill_rate=fill_rate,
            cancel_ratio=cancel_ratio,
            queue_wait_median=queue_wait_median,
            pnl_mean_bps=pnl_mean,
            as_ratio=as_ratio,
        )

        return MonteCarloResult(
            n_records=n_total,
            n_filled=n_filled,
            n_cancelled=n_cancelled,
            observed_fill_rate=fill_rate,
            observed_pnl_mean_bps=pnl_mean,
            observed_pnl_std_bps=pnl_std,
            observed_as_ratio=as_ratio,
            n_simulations=cfg.n_simulations,
            cycles_per_month=cycles_per_month,
            pnl_mean_jpy=float(np.mean(monthly_pnls)),
            pnl_std_jpy=float(np.std(monthly_pnls)),
            pnl_percentiles_jpy=percentiles,
            pnl_mean_bps=pnl_mean,
            pnl_std_bps=pnl_std,
            var_95_jpy=var_95,
            cvar_95_jpy=cvar_95,
            prob_loss=prob_loss,
            prob_profit=prob_profit,
            g11_fill_rate=fill_rate,
            g11_cancel_ratio=cancel_ratio,
            g11_queue_wait_median=queue_wait_median,
            g11_pnl_mean_bps=pnl_mean,
            g11_as_ratio=as_ratio,
            g11_pass=g11_pass,
            breakeven_fill_rate=breakeven_fill_rate,
            breakeven_pnl_bps=breakeven_pnl_bps,
            raw_monthly_pnls=monthly_pnls,
        )

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        fill_rates: Sequence[float] | None = None,
        pnl_adjustments_bps: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """fill_rate / pnl_mean を変えた感度分析.

        Args:
            fill_rates: テストする fill_rate の一覧 (default: 0.5~1.0)
            pnl_adjustments_bps: pnl に加算する値の一覧 (default: -2~+2)

        Returns:
            各シナリオの {fill_rate, pnl_adj, mean_jpy, var95_jpy, prob_loss} リスト
        """
        if fill_rates is None:
            fill_rates = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        if pnl_adjustments_bps is None:
            pnl_adjustments_bps = [-2.0, -1.0, 0.0, 1.0, 2.0]

        cfg = self.config
        pnl_bps = self._extract_filled_pnl_bps()

        notional = cfg.lot_size_btc * cfg.btc_price_jpy
        jpy_per_bps = 1e-4 * notional
        cycles_per_month = cfg.cycles_per_day * cfg.days_per_month
        results: list[dict[str, Any]] = []

        for fr in fill_rates:
            base_monthly_bps, fills_per_sim = self._sample_monthly_pnl_bps(
                fill_rate=float(fr),
                pnl_bps=pnl_bps,
                cycles_per_month=cycles_per_month,
            )
            fills_per_sim_f = fills_per_sim.astype(np.float64, copy=False)
            for adj in pnl_adjustments_bps:
                adj_value = float(adj)
                if adj_value == 0.0:
                    monthly = base_monthly_bps * jpy_per_bps
                else:
                    monthly = (base_monthly_bps + fills_per_sim_f * adj_value) * jpy_per_bps

                results.append({
                    "fill_rate": fr,
                    "pnl_adj_bps": adj_value,
                    "mean_jpy": float(np.mean(monthly)),
                    "std_jpy": float(np.std(monthly)),
                    "var_95_jpy": float(np.percentile(monthly, 5)),
                    "prob_loss": float(np.mean(monthly < 0)),
                    "p50_jpy": float(np.median(monthly)),
                })

        return results

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, result: MonteCarloResult) -> str:
        """Pretty-print report, also returns as string."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  PnL Monte Carlo Report (014# T5)")
        lines.append("=" * 60)
        lines.append("")

        # Observed data
        lines.append("▶ Observed Data")
        lines.append(f"  Records: {result.n_records} (filled={result.n_filled}, cancelled={result.n_cancelled})")
        lines.append(f"  Fill rate: {result.observed_fill_rate:.1%}")
        lines.append(f"  PnL mean:  {result.observed_pnl_mean_bps:+.3f} bps")
        lines.append(f"  PnL stdev: {result.observed_pnl_std_bps:.3f} bps")
        lines.append(f"  Adverse selection: {result.observed_as_ratio:.1%}")
        lines.append(f"  Queue wait median: {result.g11_queue_wait_median:.1f}s")
        lines.append("")

        # Monthly simulation
        lines.append(f"▶ Monthly Simulation ({result.n_simulations:,} paths, {result.cycles_per_month:,} cycles/mo)")
        lines.append(f"  E[PnL]:  {result.pnl_mean_jpy:+,.0f} JPY/mo")
        lines.append(f"  σ[PnL]:  {result.pnl_std_jpy:,.0f} JPY/mo")
        for k, v in result.pnl_percentiles_jpy.items():
            lines.append(f"    {k}: {v:+,.0f} JPY")
        lines.append("")

        # Risk
        lines.append("▶ Risk Metrics")
        lines.append(f"  VaR 95%:  {result.var_95_jpy:+,.0f} JPY")
        lines.append(f"  CVaR 95%: {result.cvar_95_jpy:+,.0f} JPY")
        lines.append(f"  P(loss):  {result.prob_loss:.1%}")
        lines.append(f"  P(profit): {result.prob_profit:.1%}")
        lines.append("")

        # G1.1
        lines.append("▶ G1.1 Criteria (000# §3.3)")
        g = "✅" if result.g11_fill_rate >= 0.90 else "❌"
        lines.append(f"  {g} fill_rate:    {result.g11_fill_rate:.1%} (≥90%)")
        g = "✅" if result.g11_cancel_ratio <= 0.30 else "❌"
        lines.append(f"  {g} cancel_ratio: {result.g11_cancel_ratio:.1%} (≤30%)")
        g = "✅" if result.g11_queue_wait_median <= 60.0 else "❌"
        lines.append(f"  {g} queue_wait:   {result.g11_queue_wait_median:.1f}s (≤60s)")
        g = "✅" if result.g11_pnl_mean_bps >= 0 else "❌"
        lines.append(f"  {g} pnl_mean:     {result.g11_pnl_mean_bps:+.3f} bps (≥0)")
        g = "✅" if result.g11_as_ratio <= 0.20 else "❌"
        lines.append(f"  {g} AS_ratio:     {result.g11_as_ratio:.1%} (≤20%)")
        verdict = "PASS" if result.g11_pass else "FAIL"
        lines.append(f"  → G1.1 = **{verdict}** (n={result.n_records}, 統計的意味は n≥200 で確定)")
        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    # 027# 旧 to_dict(self, result) は MonteCarloResult.to_dict() に統合済み。削除。
