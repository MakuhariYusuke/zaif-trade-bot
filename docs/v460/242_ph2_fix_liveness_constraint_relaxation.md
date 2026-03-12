# 242# Liveness Constraint Relaxation (233# P1)

| 項目       | 内容                                          |
|------------|-----------------------------------------------|
| 種別       | fix (233# P1 対応)                             |
| 親チケット | 232# P1-4 / 233# P1                           |
| 前提       | 240# Toxicity Budget, 218#/219# Probe機構     |
| テスト     | 14 件追加 (3358 passed)                        |

## 1. 背景と問題

233# Gemini レビュー P1 指摘:
> 「システムが数時間スリープすること（No Trade）」を運用上の正常系として許容しろ。

現状の anti-stagnation 機構は、Kill Gate が正当にブロックしている場合でも
probe (強制取引) を発動して損失を追加する **「穴の開いたバケツ」** 問題を抱えていた:

| # | 機構 | 挙動 | 問題 |
|---|------|------|------|
| 218# | Stale probe | 10 cycle 停滞 → 1 trade 強制通過 | toxic 市場で損失追加 |
| 219# | Progressive probe | 10→5→3→2 cycle で加速 | 加速が逆効果 |
| 219# | Force release | 5 probe 連続 → kill 完全解除 | 最悪ケース: 安全弁撤去 |

## 2. 市場理論的根拠

**Glosten-Milgrom (1985)** の逆選択モデル:
- toxicity score が KILL 水準 (rolling PnL < threshold) のとき、
  informed trader の存在確率が高く、全オーダーが逆選択される。
- この状態で probe (強制取引) を行うことは、
  逆選択コストを追加で支払うことと等価。
- **最適戦略**: 市場から撤退し、情報非対称性が解消されるまで待機。

## 3. 実装

### A. DynamicKillManager: `toxic_kill_stale_multiplier`

**ファイル**: `ztb/risk/sell_dynamic_kill.py`

```python
# DynamicKillConfig に追加
toxic_kill_stale_multiplier: int = 10  # default: probe 間隔 ×10
```

**ロジック** (`check_kill()`):
```
effective_max_stale = _effective_probe_interval()
if toxicity_budget_enabled AND assess_toxicity() == KILL AND rolling_mean != None:
    effective_max_stale *= toxic_kill_stale_multiplier  # 10 → 100 cycles
```

**効果**:
- 従来: probe 間隔 ~10 cycles (~20 分) → force release ~50 cycles (~1.5 時間)
- 242#: probe 間隔 ~100 cycles (~3.3 時間) → force release ~500 cycles (~16 時間)
- toxicity GREEN/YELLOW/ORANGE 時は従来通り (multiplier=1)
- `toxicity_budget_enabled=False` 時も従来通り (multiplier=1)

**安全弁**: `toxic_kill_stale_multiplier=1` で完全無効化 (後方互換)。

### B. Orchestrator: Quiescence Sleep Escalation

**ファイル**: `scripts/v460/lib/fill_config.py`

```python
quiescence_gate_blocks_threshold: int = 20  # 連続ゲートブロック → quiescence 認定
quiescence_sleep_sec: float = 1800.0        # quiescence 時 sleep 上限 (30分)
```

**ファイル**: `scripts/v460/lib/fill_loop_orchestrator.py`

- `_effective_sleep()` に `max_override` パラメータ追加
- 連続ゲートブロック ≥ threshold → `quiescence_sleep_sec` を sleep cap として使用
- ログの正常化:
  - < threshold: `[218#] DEADLOCK WARNING` (従来通り)
  - ≥ threshold: `[242#] QUIESCENCE: ... no-trade accepted as normal`

## 4. 設計判断

### なぜ probe を完全禁止しないのか?

probe の完全禁止は、以下のシナリオで永久停止を引き起こす:
- Kill fires → rolling PnL frozen (no new data) → 市場が回復しても検知不可

**代替案**: probe 間隔を延長 (×10) して頻度を下げつつ、最終的に回復パスを維持。
これにより「数時間の No Trade」を実現しつつ、市場回復時の自動復帰も保証。

### Quiescence vs Deadlock

- **Deadlock** (< 20 blocks): 一時的なゲートブロック。通常 10 分以内に解消。
- **Quiescence** (≥ 20 blocks): 市場構造的な要因。No Trade が正常系。
  - Sleep cap 引き上げ (10分 → 30分) で API 負荷削減
  - ログレベル変更 (WARNING → INFO) でアラート疲労回避

## 5. 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `ztb/risk/sell_dynamic_kill.py` | `toxic_kill_stale_multiplier` config + `_toxic_kill_multiplier()` |
| `scripts/v460/lib/fill_config.py` | `quiescence_*` 設定 2 件追加 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | `_effective_sleep(max_override=)` + quiescence 判定 |
| `tests/unit/v460/test_242_liveness_relaxation.py` | 14 件 (probe 延長 9 + config 2 + sleep 3) |

## 6. 後方互換性

- `toxic_kill_stale_multiplier=10` (default): 新動作有効
  - `toxicity_budget_enabled=False` 時は effect なし (multiplier=1)
- `toxic_kill_stale_multiplier=1`: 完全無効化 (従来互換)
- `quiescence_gate_blocks_threshold=20` (default): 新動作有効
- `quiescence_sleep_sec=0`: 無効化 (max_cycle_sleep_sec のみ)
- 既存テスト 3344 件: 全 PASS (回帰なし)
