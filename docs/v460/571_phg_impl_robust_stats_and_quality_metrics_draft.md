# 571# [phg] [impl] ロバスト統計ユーティリティの実装と執行品質比較分析ドラフト

> **ステータス**: 実装ドラフト・セルフレビュー (Gemini 担当)  
> **作成日**: 2026-03-23  
> **参照**: 570# (ロバスト入力設計), 569# (I2/I3計測)

---

## 1. Task B: `ztb/utils/robust_stats.py` 実装ドラフト

eDRC への入力となる $\sigma$ や $OFI$ をノイズから保護するための高速・頑健な計算モジュール。

```python
import numpy as np
from typing import Optional, Tuple

class RobustStats:
    """頑健な統計計算ユーティリティ (メモリ効率・低遅延重視)."""

    @staticmethod
    def clip_outliers_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """MAD (Median Absolute Deviation) ベースの外れ値クリッピング."""
        if len(data) == 0:
            return data
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return data
        
        lower = median - threshold * mad
        upper = median + threshold * mad
        return np.clip(data, lower, upper)

    @staticmethod
    def robust_ema(
        current_val: float, 
        prev_ema: float, 
        alpha: float, 
        sigma_limit: Optional[float] = None
    ) -> float:
        """入力クリッピング付き EMA. 急激なスパイクによる指数移動平均の跳ね上がりを抑制."""
        if sigma_limit is not None and abs(current_val - prev_ema) > sigma_limit:
            # 前回の EMA から乖離しすぎている場合は、制限値でクリップしてから更新
            clipped_val = prev_ema + np.sign(current_val - prev_ema) * sigma_limit
            return alpha * clipped_val + (1 - alpha) * prev_ema
        
        return alpha * current_val + (1 - alpha) * prev_ema

    @staticmethod
    def asymmetric_ema(
        current_val: float, 
        prev_ema: float, 
        alpha_up: float, 
        alpha_down: float
    ) -> float:
        """非対称 EMA. 反転（価格逆行）に対する感度を高める設計."""
        alpha = alpha_up if current_val > prev_ema else alpha_down
        return alpha * current_val + (1 - alpha) * prev_ema

    @staticmethod
    def median_filter_fast(buffer: np.ndarray) -> float:
        """過去 N 件の中央値を高速に算出 (OFI 等のノイズ除去用)."""
        return float(np.median(buffer))
```

---

## 2. Task A: `spread_capture_bps` 効果測定スクリプト草案

加法パイプライン（Experimental）が、既存の乗法パイプラインに対してどれだけ「逆選択（AS）」を回避できているかを定量化する。

```python
def section_execution_quality_comparison(records: list[dict[str, Any]]) -> list[str]:
    """571# Task A: 加法 vs 乗法 パイプラインの執行品質比較分析."""
    filled = [r for r in records if r.get("filled")]
    if not filled:
        return ["## Execution Quality Comparison", "  (no fills)", ""]

    # パイプライン・トグルでグループ分け (または git_sha/run_id)
    # 実装例: additive_mode が記録されている前提
    groups = {"multiplicative": [], "additive": []}
    for r in filled:
        mode = "additive" if r.get("execution_additive_enabled") else "multiplicative"
        groups[mode].append(r)

    lines = ["## Execution Quality Analysis (Kissell & Glantz)"]
    for mode, recs in groups.items():
        if not recs: continue
        
        captures = [float(r.get("spread_capture_bps", 0.0)) for r in recs]
        costs = [float(r.get("adverse_selection_cost_bps", 0.0)) for r in recs]
        
        avg_cap = np.mean(captures)
        avg_cost = np.mean(costs)
        net_pnl = avg_cap + avg_cost  # Cost は負値
        
        # 評価指標: 情報反映率 (Information Capture Ratio)
        # 1.0 に近いほど、リスクを回避しつつスプレッドを多く取れている
        icr = avg_cap / abs(avg_cost) if avg_cost != 0 else 0
        
        lines.append(f"  --- {mode.upper()} (n={len(recs)}) ---")
        lines.append(f"    Spread Capture: {avg_cap:+.2f} bps")
        lines.append(f"    AS Cost (90s):  {avg_cost:+.2f} bps")
        lines.append(f"    Net Spread PnL: {net_pnl:+.2f} bps")
        lines.append(f"    Info Capture Ratio: {icr:.3f}")
    
    return lines
```

---

## 3. セルフレビュー (Self-Review)

### 3.1 数値的安定性と精度
- **MAD クリッピング**: `mad == 0`（全データが同一値）の場合のゼロ割り回避を実装済み。
- **$\epsilon$ 床**: スプレッド逆数計算（eDRC）における 1.0 bps 床により、タイトスプレッド時の天井発散を防止。

### 3.2 パフォーマンスとメモリ
- **NumPy 依存**: 純粋な Python ループを避け、ベクトル化された NumPy 関数を使用することで Numba 導入なしでも十分な速度を確保。
- **ステートレス設計**: `RobustStats` は静的メソッドとして提供し、副作用（メモリリーク）のリスクを最小化。

### 3.3 論証の死角
- **窓の同期問題**: Sell 90s / Buy 30s の不一致は `analyze_fill_logs` 側で「窓の明示」を行うことで是正。
- **パラメータ $\alpha, \beta$ の敏感さ**: 初期値は安全側に振っているが、実稼働後の I2 計測による「再キャリブレーション」が不可欠である。

---

## 4. 結論

本ドラフトにより、執行エンジンの刷新（Additive Pipeline & eDRC）を支える **「頑健な計算機」** と **「厳格な評価者」** の双方が定義された。Copilot は本コードをベースに実装を完了されたし。

---
