# v452 Remaining Tasks

## 1. Dynamic Threshold Auto-Tuning (Architecture Improvement A)
*   **Status:** Not Started.
*   **Description:** Create a script to analyze past backtest data and determine the optimal threshold multipliers for each market regime. Currently, values like `0.8` (for range) or `1.3` (for trend) are manually set heuristics.
*   **Goal:** Automate the optimization of these multipliers to adapt to changing market conditions over the long term.

## 2. Transfer Learning / Fine-tuning Pipeline (Architecture Improvement B)
*   **Status:** Not Started.
*   **Description:** Build a pipeline to fine-tune the base model (v451) using only the most recent data (e.g., last 2 weeks).
*   **Goal:** Allow the model to adapt to the *current* specific market texture without forgetting general principles learned from long-term data.

## 3. Comprehensive Backtesting of v452 Changes
*   **Status:** Pending.
*   **Description:** Run a full backtest comparing v451 (Optimized) vs v452 (Range Scalping + Advanced Regime Detection) to quantify the improvement in Profit Factor and Sharpe Ratio.

## 🔧 Repo maintenance notes (LFS migration)
* **Status:** Completed (2025-12-13)
* **Details:** Large historic artifacts were migrated to Git LFS and the repository history was rewritten to remove oversized blobs. The repo pack size is now ~35 MiB.
* **Action for developers:** Please re-clone or run `git fetch origin --prune && git reset --hard origin/main` and then `git lfs install` and `git lfs pull --all` to ensure you have all LFS objects and updated history.
