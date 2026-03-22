# 545# Phase 4b: OFI→boost変調 / Toxicity→sidecar confidence / δ*→sidecar動的天井

> **Commit**: 545#  
> **Parent**: 544# (`d6ef49061`)  
> **Phase**: phg (フェーズ横断品質)  
> **日付**: 2026-03-23

---

## §1 目的

544# で導入した計測チャネル (OFI rolling mean, δ* bps, Toxicity Budget) を
**固定パラメータの動的置換** に活用し、市場状態への即応性を向上させる。

### 設計原則
- 純関数 (spread_adaptive.py) は**変更しない** — 呼出元で modulation
- 既存パラメータの乗数/減衰として実装 — 新 config 不要
- 制御可能な上下限 (scalar cap) で暴走防止

---

## §2 実装内容

### A) OFI mean → spread_adapt boost 動的変調

**ファイル**: `scripts/v460/lib/maker_price.py` — `_ofi_modulated_boost()`

**理論**: CKS (2014) OFI が示す逆選択方向に対し、spread_adapt boost を拡大して
防御的にスプレッドを広げる。

| 条件 | 動作 |
|------|------|
| Buy 約定時、OFI mean < 0 (売り圧力) | boost UP (最大 ×1.5) |
| Sell 約定時、OFI mean > 0 (買い圧力) | boost UP (最大 ×1.5) |
| 順方向 or OFI=0 | base boost そのまま |

**数式**:
```
adverse = -ofi_mean (buy) or +ofi_mean (sell)
scalar = 1.0 + min(adverse, 1.0) × 0.5
effective_boost = base_boost × scalar
```

### B) Toxicity → sidecar confidence 減衰

**ファイル**: `scripts/v460/lib/cycle_gate_aggregator.py` — `_apply_sidecar_offset()`

**理論**: Glosten-Milgrom (1985) 逆選択レベルが高い局面で SAC sidecar の confidence を
減衰させ、informed trader presence 下での過積極的ポジション構築を抑制する。

| ToxicityLevel | confidence 乗数 |
|---------------|-----------------|
| GREEN | 1.0 (フル信頼) |
| YELLOW | 0.7 (警戒) |
| ORANGE | 0.3 (大幅減衰) |
| KILL | 0.0 (完全無視) |

**実装**: `_TOXICITY_CONFIDENCE_MAP` モジュール定数 → `signal.confidence × attenuation`

### C) δ* → sidecar 動的天井

**ファイル**: `scripts/v460/lib/cycle_gate_aggregator.py` — `_apply_sidecar_offset()`

**理論**: Avellaneda-Stoikov (2008) δ* が実勢スプレッドより広い (ratio > 1.0) とき、
理論上より広いスプレッドが最適 → SAC sidecar に**より大きな裁量**を許容する。

**数式**:
```
if delta_star_ratio > 1.0:
    ceiling_scalar = min(delta_star_ratio, 2.0)
    effective_max_boost = config.max_boost_bps × ceiling_scalar
```

**データフロー**: `orchestrator.maker_price._last_as_delta_star_ratio`
→ `evaluate(delta_star_ratio=...)` → `_apply_sidecar_offset(delta_star_ratio=...)`

---

## §3 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/maker_price.py` | `_ofi_modulated_boost()` 追加、`_apply_spread_adaptive()` で呼出 |
| `scripts/v460/lib/cycle_gate_aggregator.py` | `_TOXICITY_CONFIDENCE_MAP` 追加、`evaluate()` に `delta_star_ratio` 引数追加、`_apply_sidecar_offset()` に toxicity+δ* ロジック追加 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | `evaluate()` 呼出に `delta_star_ratio=` 追加 |

---

## §4 テスト結果

```
tests/unit/v460: 428 passed, 1 failed (test_143 — 他AI変更、既知)
test_sidecar_sac_integration: 77 passed
spread_adapt + ofi: 81 passed
```

---

## §5 動的値フロー図

```
OFI rolling deque (50cyc)
    ↓ mean()
    ↓ adverse direction check
    ↓ × scalar [1.0, 1.5]
    → narrow_spread_boost_buy / _sell (maker_price._apply_spread_adaptive)

ToxicityLevel (GREEN/YELLOW/ORANGE/KILL)
    ↓ _TOXICITY_CONFIDENCE_MAP lookup
    ↓ × signal.confidence
    → effective_confidence (sidecar v1/v2)

δ* (A-S δ* ratio from maker_price)
    ↓ ratio > 1.0 → ceiling_scalar (cap=2.0)
    ↓ × config.max_boost_bps
    → dynamic max_boost (sidecar v2)
```

---

## §6 リスク評価

| リスク | 対策 |
|--------|------|
| OFI boost が過大 → 約定率低下 | scalar cap 1.5, adverse 方向のみ |
| Toxicity attenuation が SAC 学習を阻害 | GREEN=1.0 (通常は影響なし), KILL=0.0 (kill 時は元々注文停止) |
| δ* ceiling が暴走 | cap=2.0 (config × 2倍が上限) |
| 全て乗算的 → 複合時に過剰 | 各段に独立 cap あり、相互独立な制御チャネル |

---

## §7 次ステップ

| 優先度 | 施策 | 根拠 |
|--------|------|------|
| P1 | sidecar hard ceiling 引上げ (0.20→0.30) | SAC ±5.0bps への段階的拡大 |
| P1 | CalibrationMap → sidecar confidence 統合 | 538# §6「第三の道」learned calibration |
| P2 | δ* → executor stage 参照 (execution-level floor) | 現在 pre-order のみ。executor にも δ* 情報を伝搬 |
| P2 | OFI boost 感度 k の YAML config 化 | 現在 k=0.5 ハードコード → 運用中チューニング対応 |
| P3 | drift detection for OFI/Toxicity | 分布シフト監視 (ztb/utils/drift_detection.py 活用) |
