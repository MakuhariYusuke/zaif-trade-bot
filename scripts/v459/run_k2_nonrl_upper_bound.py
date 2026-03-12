#!/usr/bin/env python3
"""
K2: 非RL上限テスト — 同一特徴量で XGBoost / Logistic の方向予測力を検証

SAC と同じ 8 特徴量 (RSI×7 + ReturnStdDev) で BTC/JPY 1min の
次ステップ符号を予測。Walk-forward 5 窓で IC・accuracy を算出。

IC ≈ 0 → 特徴量情報量不足 (v460 で特徴量改革必要)
IC > 0.02 → 特徴量に情報あり、SAC の学習器が問題

Usage:
  python scripts/v459/run_k2_nonrl_upper_bound.py
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 定数
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "btc_jpy_1m_v451_optimized_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "results" / "k2_nonrl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["RSI", "RSI_D1", "RSI_H1", "RSI_H4", "RSI_M1", "RSI_M15", "RSI_M5", "ReturnStdDev"]
TRAIN_END_INDEX = 973544  # 80% split (same as SAC)

# Walk-forward設定
N_FOLDS = 5
TRAIN_RATIO = 0.80  # 各fold内のtrain/test比率


# ============================================================================
# データ前処理
# ============================================================================

def load_and_prepare() -> pd.DataFrame:
    """データ読み込み、ターゲット生成、欠損除去。"""
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {df.shape}")

    # 特徴量をfloat32に統一 (float16→float32)
    for col in FEATURE_COLS:
        df[col] = df[col].astype(np.float32)

    # ターゲット: 次ステップの符号 (1=up, 0=down/flat)
    df["price_change"] = df["close"].shift(-1) - df["close"]
    df["target"] = (df["price_change"] > 0).astype(int)

    # 欠損除去
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    print(f"After dropna: {df.shape}")
    print(f"Target balance: up={df['target'].mean():.4f}")
    return df


# ============================================================================
# Walk-forward評価
# ============================================================================

def walk_forward_eval(
    df: pd.DataFrame,
    model_name: str,
    model_factory: Any,
) -> List[Dict[str, Any]]:
    """Walk-forward N窓で評価。"""
    n = len(df)
    fold_size = n // N_FOLDS
    results = []

    for fold_i in range(N_FOLDS):
        fold_start = fold_i * fold_size
        fold_end = min((fold_i + 1) * fold_size, n)
        fold_data = df.iloc[fold_start:fold_end]

        train_size = int(len(fold_data) * TRAIN_RATIO)
        train = fold_data.iloc[:train_size]
        test = fold_data.iloc[train_size:]

        if len(test) < 100:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train["target"].values
        X_test = test[FEATURE_COLS].values
        y_test = test["target"].values
        price_changes = test["price_change"].values

        # スケーリング
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # モデル訓練
        model = model_factory()
        model.fit(X_train_s, y_train)

        # 予測
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]  # P(up)

        # 指標計算
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # IC: P(up) を [0,1] → [-1,1] に変換し、実 price_change との Spearman
        signal = y_prob * 2 - 1  # [-1, 1]
        ic_result = stats.spearmanr(signal, price_changes)
        ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
        ic_p = float(ic_result.pvalue) if not np.isnan(ic_result.pvalue) else 1.0

        # 方向一致率 (予測符号 × 実符号)
        direction_correct = np.mean((y_pred == y_test))

        # 確信度別 IC
        high_conf_mask = np.abs(signal) > 0.3
        n_high = high_conf_mask.sum()
        if n_high > 50:
            ic_high = float(stats.spearmanr(signal[high_conf_mask], price_changes[high_conf_mask]).correlation)
        else:
            ic_high = None

        result = {
            "fold": fold_i,
            "model": model_name,
            "train_size": len(train),
            "test_size": len(test),
            "accuracy": round(acc, 6),
            "f1_macro": round(f1, 6),
            "ic_spearman": round(ic, 6),
            "ic_pvalue": round(ic_p, 6),
            "ic_high_conf": round(ic_high, 6) if ic_high is not None else None,
            "n_high_conf": int(n_high),
            "target_rate": round(float(y_test.mean()), 4),
        }
        results.append(result)

        print(f"  [{model_name}] fold={fold_i}: acc={acc:.4f} ic={ic:.6f} p={ic_p:.4f} f1={f1:.4f}")

    return results


def oos_eval(
    df: pd.DataFrame,
    model_name: str,
    model_factory: Any,
) -> Dict[str, Any]:
    """In-sample で訓練し、OOS (train_end_index 以降) で評価。"""
    train = df.iloc[:TRAIN_END_INDEX]
    test = df.iloc[TRAIN_END_INDEX:]

    # 欠損除去（全データで既にdropna済みだが念のため）
    train = train.dropna(subset=FEATURE_COLS + ["target"])
    test = test.dropna(subset=FEATURE_COLS + ["target"])

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values
    price_changes = test["price_change"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    signal = y_prob * 2 - 1
    ic_result = stats.spearmanr(signal, price_changes)
    ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
    ic_p = float(ic_result.pvalue) if not np.isnan(ic_result.pvalue) else 1.0

    result = {
        "model": model_name,
        "mode": "OOS",
        "train_size": len(train),
        "test_size": len(test),
        "accuracy": round(acc, 6),
        "f1_macro": round(f1, 6),
        "ic_spearman": round(ic, 6),
        "ic_pvalue": round(ic_p, 6),
        "target_rate": round(float(y_test.mean()), 4),
    }

    print(f"  [{model_name}] OOS: acc={acc:.4f} ic={ic:.6f} p={ic_p:.4f} f1={f1:.4f} n={len(test)}")
    return result


# ============================================================================
# モデルファクトリ
# ============================================================================

def make_logistic():
    return LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=42)


def make_xgboost():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=1,
    )


# ============================================================================
# メイン
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("  K2: Non-RL Upper Bound Test")
    print("  Same 8 features as SAC, XGBoost/Logistic walk-forward")
    print("=" * 60)

    df = load_and_prepare()

    models = [("Logistic", make_logistic)]
    if HAS_XGBOOST:
        models.append(("XGBoost", make_xgboost))
    else:
        print("⚠️ XGBoost not installed, skipping")

    all_wf_results: List[Dict] = []
    all_oos_results: List[Dict] = []

    for name, factory in models:
        print(f"\n--- {name}: Walk-Forward ({N_FOLDS} folds) ---")
        wf = walk_forward_eval(df, name, factory)
        all_wf_results.extend(wf)

        print(f"\n--- {name}: OOS (train_end={TRAIN_END_INDEX}) ---")
        oos = oos_eval(df, name, factory)
        all_oos_results.append(oos)

    # 集約
    summary = _build_summary(all_wf_results, all_oos_results)

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"k2_results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 結果表示
    _print_summary(summary)
    print(f"\nSaved: {out_path}")


def _build_summary(
    wf_results: List[Dict], oos_results: List[Dict]
) -> Dict[str, Any]:
    """結果を構造化。"""
    per_model: Dict[str, Dict] = {}

    for name in set(r["model"] for r in wf_results):
        model_wf = [r for r in wf_results if r["model"] == name]
        model_oos = [r for r in oos_results if r["model"] == name]

        accs = [r["accuracy"] for r in model_wf]
        ics = [r["ic_spearman"] for r in model_wf]

        per_model[name] = {
            "walk_forward": {
                "n_folds": len(model_wf),
                "accuracy_mean": round(float(np.mean(accs)), 6),
                "accuracy_std": round(float(np.std(accs)), 6),
                "ic_mean": round(float(np.mean(ics)), 6),
                "ic_std": round(float(np.std(ics)), 6),
                "ic_all_positive": all(ic > 0 for ic in ics),
                "ic_significant": sum(1 for r in model_wf if r["ic_pvalue"] < 0.05),
                "folds": model_wf,
            },
            "oos": model_oos[0] if model_oos else None,
        }

    # Gate判定
    best_model = max(per_model.items(), key=lambda x: abs(x[1]["walk_forward"]["ic_mean"]))
    best_ic = best_model[1]["walk_forward"]["ic_mean"]
    best_acc = best_model[1]["walk_forward"]["accuracy_mean"]
    oos_ic = best_model[1]["oos"]["ic_spearman"] if best_model[1]["oos"] else 0
    oos_acc = best_model[1]["oos"]["accuracy"] if best_model[1]["oos"] else 0.5

    has_edge = abs(best_ic) > 0.02 and best_acc > 0.51
    oos_edge = abs(oos_ic) > 0.01 and oos_acc > 0.505

    return {
        "experiment": "K2 Non-RL Upper Bound",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {
            "path": str(DATA_PATH),
            "features": FEATURE_COLS,
            "n_features": len(FEATURE_COLS),
            "train_end_index": TRAIN_END_INDEX,
        },
        "per_model": per_model,
        "gate": {
            "best_model": best_model[0],
            "best_wf_ic": best_ic,
            "best_wf_acc": best_acc,
            "oos_ic": oos_ic,
            "oos_acc": oos_acc,
            "condition_ic": "|IC| > 0.02",
            "condition_acc": "accuracy > 51%",
            "has_wf_edge": has_edge,
            "has_oos_edge": oos_edge,
            "verdict": "FEATURES_HAVE_INFO" if has_edge else "FEATURES_NO_INFO",
        },
    }


def _print_summary(summary: Dict) -> None:
    gate = summary["gate"]

    print("\n" + "=" * 60)
    print("  K2 Results Summary")
    print("=" * 60)

    for name, data in summary["per_model"].items():
        wf = data["walk_forward"]
        oos = data["oos"]
        print(f"\n  [{name}]")
        print(f"    Walk-Forward: acc={wf['accuracy_mean']:.4f}±{wf['accuracy_std']:.4f}"
              f"  IC={wf['ic_mean']:.6f}±{wf['ic_std']:.6f}"
              f"  sig={wf['ic_significant']}/{wf['n_folds']}")
        if oos:
            print(f"    OOS:          acc={oos['accuracy']:.4f}"
                  f"  IC={oos['ic_spearman']:.6f}  p={oos['ic_pvalue']:.4f}")

    print(f"\n  Gate ({gate['condition_ic']} & {gate['condition_acc']}):")
    print(f"    Best model: {gate['best_model']}")
    print(f"    WF:  IC={gate['best_wf_ic']:.6f}  acc={gate['best_wf_acc']:.4f}"
          f"  → {'✅ EDGE' if gate['has_wf_edge'] else '❌ NO EDGE'}")
    print(f"    OOS: IC={gate['oos_ic']:.6f}  acc={gate['oos_acc']:.4f}"
          f"  → {'✅ EDGE' if gate['has_oos_edge'] else '❌ NO EDGE'}")
    print(f"\n  Verdict: {gate['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
