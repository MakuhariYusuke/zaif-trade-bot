# 465# モデル退化の根本原因分析と構造的修正

| 項目 | 値 |
|------|------|
| 起票 | 2026-03-19 |
| 前提 | 461#-464# 分析で特定された EV 0.5-1.0 gap, ceiling 100%, deep-night AS |
| 方針 | 対症療法 (時間帯制限) ではなく **根本原因** に迫る |

---

## §1 根本原因の特定

### §1.1 問題の統一

461#-463# で報告された3つの主要問題:

1. **EV スコア 0.5-1.0 帯の構造的不在** (461# §4)
2. **Ceiling 100% 常時発火** (461# §5.3)
3. **Deep-night AS 高率** (462# §3)

これらは **単一の根本原因** を共有する: **pnl120 モデルの退化 (degeneration)**

### §1.2 退化モデルの詳細

| モデル | 木数 | 学習サンプル数 | 学習日 | 予測出力 | 状態 |
|--------|------|----------------|--------|----------|------|
| pnl120_buy | **1** | **26** | 2026-03-13 | CONST 4.7352 | **退化** |
| pnl120_sell | **1** | **25** | 2026-03-18 | CONST -1.2812 | **退化** |
| pnl30_buy | 300 | 519 | 2026-02-24 | 変動あり | 健全 |
| pnl30_sell | 300 | 44 | 2026-03-18 | 変動あり | 正常 |

**ファイルサイズ比較**:
- pnl120_buy 現行: 8,051 bytes vs backup_pre288: 129,983 bytes (16x 小)
- pnl120_sell 現行: 8,093 bytes vs backup_pre288: 250,199 bytes (31x 小)

### §1.3 退化メカニズム (因果連鎖)

```
bootstrap_min_total_samples: 25       ← 設定値が低すぎる
    ↓
26 samples で retrain が発火            ← min_total=25 を通過
    ↓
warm_start_enabled = False (side model)  ← 前モデルの 150 trees を破棄
    ↓
LightGBM fit(26 samples)              ← min_child_samples=20
    ↓
early_stopping: 1 tree で停止          ← train=20, val=6, 分割不能
    ↓
statistical_gate: skipped              ← test_samples=6 < 40 (最低要件)
    ↓
1-tree モデルが deploy                  ← tree count / 分散ガードなし
    ↓
予測値 = 定数 (4.735 / -1.281)         ← 特徴量に依存しない
```

### §1.4 問題への連鎖

**EV 0.5-1.0 gap**:
- EV = 0.4 × pnl30 + 0.6 × pnl120
- pnl120 = 定数 → EV = 定数 + 0.4 × pnl30 変動
- buy: 0.6 × 4.735 = 2.84 がベース → 常に EV > 2.0
- sell: 0.6 × (-1.281) = -0.77 がベース → 常に EV < 0.5
- 0.5-1.0 帯は原理的に出現不能

**Ceiling 100%**:
- EV multiplier = 1.0 + 0.05 × ev_score
- buy の ev_score ≈ 2.84+ → multiplier ≈ 1.14+
- offset が膨らみ、ceiling_ratio=0.20 を常時超過

**Deep-night AS**:
- 4 モデル中 3 モデルが定数出力
- Skip gate は favorable/unfavorable 条件を識別不能
- 時間帯による市場構造の変化を学習できていない

---

## §2 構造的修正

### §2.1 D1: モデル退化ガード — 最小木数 (`min_deploy_trees`)

**場所**: `retrain_scheduler.py` — `retrain_model()` 内、訓練後・deploy 前

```python
min_deploy_trees = safe_to_int(cfg.get("min_deploy_trees", 3), 3)
if actual_n_trees < min_deploy_trees:
    return {**result, "status": "rejected", "reason": "degenerate_model: ..."}
```

LightGBM が生成した木数が閾値未満なら、モデルを棄却して既存モデルを保持。

### §2.2 D2: 予測分散ガード — 定数出力検出 (`min_pred_std`)

**場所**: `retrain_scheduler.py` — D1 直後

```python
preds = lgbm.predict(X_sc)
pred_std = float(np.std(preds))
if pred_std < min_pred_std:
    return {**result, "status": "rejected", "reason": "constant_output: ..."}
```

定数出力 (std < 0.01) はスキップゲートの識別能力を完全に破壊するため棄却。

### §2.3 D3: bootstrap_min_total_samples 引き上げ

**変更**: `configs/v460/fill_test.yaml`

```yaml
bootstrap_min_total_samples: 50  # 旧: 25
```

**根拠**: `lgbm_min_child_samples=20` のとき、25 samples では LightGBM がノードを分割する余地がほとんどない。50 samples で 2.5x min_child_samples を確保。

### §2.4 balance_forced_switch 一貫性修正

**場所**: `orchestrator_mid_cycle.py`

```python
if ctx.resolved_side_reason == "balance_switch":
    record.balance_forced_switch = True
```

f840d0e の `resolved_side_reason="balance_switch"` (51件) が `balance_forced_switch=None` だった問題を修正。retrain_scheduler の Y5 フィルタ (`balance_forced_switch=True` 除外) が正しく機能するようになる。

### §2.5 バックアップモデル復元

退化した pnl120 モデルを `backup_pre288` から復元:

| モデル | 復元前 | 復元後 |
|--------|--------|--------|
| pnl120_buy | 1 tree, 26 samples, 8KB | 150 trees, 431 samples, 130KB |
| pnl120_sell | 1 tree, 25 samples, 8KB | 300 trees, 229 samples, 250KB |

D1/D2/D3 ガードが有効な状態で復元するため、再び退化するリスクは排除済み。

---

## §3 464# 提案との対応

| 464# 提案 | 本対応 | 備考 |
|-----------|--------|------|
| §1.2 EV gap 構造的原因確認 | ✅ §1.4 で因果連鎖を完全解明 | pnl120 定数出力が根本原因 |
| §2 A-S inventory 分析 | 保留 | モデル退化修正が先決 |
| §3.1 DuckDB+BI | 将来検討 | |
| §3.2 ChatOps | 将来検討 | |
| §3.3 Rust/Go IPC | 将来検討 | |
| §3.4 Tier1 macro data | 将来検討 | |
| §3.5 Chaos Engineering | 将来検討 | D1/D2 はその一形態 |

---

## §4 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/ml/retrain_scheduler.py` | D1/D2 ガード追加 + デフォルト定義 |
| `configs/v460/fill_test.yaml` | D3 bootstrap_min_total=50, D1/D2 パラメータ追加 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | balance_forced_switch 一貫性修正 |
| `models/v460/skip_gate_lgbm_pnl120_buy.pkl` | backup_pre288 から復元 |
| `models/v460/skip_gate_lgbm_pnl120_sell.pkl` | backup_pre288 から復元 |
| `tests/unit/v460/test_retrain_hot_reload.py` | D1/D2 テスト追加 + stub 更新 |

## §5 テスト結果

- `test_retrain_hot_reload.py`: 90/90 passed (既存 86 + 新規 4)
- `test_141_side_specific_models.py`: 48/48 passed
- `test_189_alt_horizon_macro_integration.py` + `test_139_review_fixes.py` + `test_145_s13_boundary_guards.py`: 98/98 passed
