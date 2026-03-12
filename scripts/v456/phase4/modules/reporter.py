"""Walk-Forward Reporter: 結果集約と報告"""

import json
import logging
from pathlib import Path

import numpy as np

from .result import WalkForwardResult

logger = logging.getLogger(__name__)


class WalkForwardReporter:
    """結果集約と報告"""

    def __init__(self, result: WalkForwardResult):
        self.result = result

    def report(self):
        """結果報告"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 Walk-Forward Analysis Results")
        logger.info("=" * 70)
        
        # ウィンドウ別結果
        logger.info(f"\nWindow-by-Window Performance:")
        for perf in self.result.performances:
            logger.info(
                f"  Window {perf.window_id}: "
                f"Val ROI {perf.val_roi:.4f} | "
                f"Test ROI {perf.test_roi:.4f} | "
                f"Sharpe {perf.sharpe_ratio:.4f}"
            )
        
        # 平均性能
        logger.info(f"\nAggregate Performance:")
        logger.info(f"  Average Val ROI: {self.result.average_val_roi:.4f}")
        logger.info(f"  Average Test ROI: {self.result.average_test_roi:.4f}")
        logger.info(f"  Test ROI Std Dev: {self.result.test_roi_std:.4f}")
        logger.info(f"  Average Sharpe: {self.result.average_sharpe:.4f}")
        logger.info(f"  Sharpe Consistency: {self.result.sharpe_consistency:.4f}")
        logger.info(f"  Average Win Rate: {self.result.average_win_rate:.4f}")
        logger.info(f"  Overfitting Ratio: {self.result.overfitting_ratio:.4f}")
        
        logger.info("=" * 70)

    def save_results(self, output_path: Path):
        """結果をJSON保存"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            "windows": len(self.result.windows),
            "average_val_roi": self.result.average_val_roi,
            "average_test_roi": self.result.average_test_roi,
            "test_roi_std": self.result.test_roi_std,
            "average_sharpe": self.result.average_sharpe,
            "sharpe_consistency": self.result.sharpe_consistency,
            "average_win_rate": self.result.average_win_rate,
            "overfitting_ratio": self.result.overfitting_ratio,
            "performances": [
                {
                    "window_id": p.window_id,
                    "val_roi": p.val_roi,
                    "test_roi": p.test_roi,
                    "sharpe_ratio": p.sharpe_ratio,
                    "win_rate": p.win_rate,
                    "trades": p.trades,
                }
                for p in self.result.performances
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_path}")
