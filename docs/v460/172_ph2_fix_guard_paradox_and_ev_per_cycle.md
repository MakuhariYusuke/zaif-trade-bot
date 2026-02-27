# 172# Guard Paradox 根本対策 + EV_per_cycle 実装

> **種別**: fix  
> **フェーズ**: ph2  
> **日付**: 2026-02-27  
> **前提**: 170# §10 著者回答 (Guard Paradox 確認), 171# 技術精査  
> **由来**: 170# §10.5 P0 (balance_forced_skip 正フィードバックループ対策), P1 (EV_per_cycle)

---

## §1 背景 — Guard Paradox と正フィードバックループ

170# §10.4.1 および 171# で確認された **Sell Guard Paradox**:

```
trending_sell_skip (⑨)   ── sell ブロック ─→ buy 在庫蓄積
        ↓
sell_dynamic_kill (⑪)    ── trending_up -0.1bps → ほぼ全 sell 遮断
        ↓
balance_forced_skip (⑥)  ── sell balance 枯渇 → forced switch → skip
        ↓
                         ── sell 機会さらに喪失 → ループ強化
```

**定量根拠** (168h ログ):
- balance_forced_skip: 377 件 **100% sell 側**
- sell ガード合計ブロック率: 46.2% (buy 18.1% の **2.16 倍**)
- balance_forced sell: avg PnL **-0.659 bps** (通常 sell -0.449 bps)
- VG-fired sell: **-0.608 bps** (non-VG sell -0.443 bps)

→ ガードが中程度リスクを排除し、最悪ケースのみ通過する「逆選択」。

---

## §2 対策設計 — InvSkew バイパス方式

### 設計思想

ガードを追加するのではなく、**在庫偏重時にのみ sell ガードをバイパス** することで
正フィードバックループを断ち切る。

$\text{bypass} = \begin{cases} \text{true} & \text{if } \texttt{inv\_net\_imbalance} \geq \texttt{sell\_guard\_inv\_bypass\_threshold} \\ \text{false} & \text{otherwise} \end{cases}$

- `inv_net_imbalance` ∈ [-1, 1]: 直近 100 fills の buy/sell 偏り (+1 = 全 buy, -1 = 全 sell)
- `sell_guard_inv_bypass_threshold` = 0.3 (hot-reload 対応)

### 対象ガード

| ガード | バイパス条件 | 効果 |
|--------|------------|------|
| `trending_sell_skip` (⑨) | imbalance ≥ 0.3 | 在庫偏重時に sell を強制実行 |
| `sell_dynamic_kill` (⑪) | imbalance ≥ 0.3 | 同上 |
| `balance_forced_skip` (⑥) | 直接介入なし | ⑨⑪のバイパスで自然に解消 |

---

## §3 実装詳細

### 3.1 新 config フィールド

**`fill_config.py`** — `FillConfig` dataclass に追加:
```python
sell_guard_inv_bypass_threshold: float = 0.3
# 171# Guard Paradox 対策: 在庫偏重時に sell ガードを自動緩和
```

YAML 解析 (`_from_yaml`) + hot-reload whitelist (`config_hot_reload.py`) に追加済み。

### 3.2 trending_sell_skip バイパス (orchestrator L770+)

```python
# --- 171# Guard Paradox: InvSkew bypass ---
_inv_imbalance = getattr(
    self._maker_price, "_inv_net_imbalance", 0.0
)
_inv_bypass_threshold = self._config.sell_guard_inv_bypass_threshold
if (
    _inv_imbalance >= _inv_bypass_threshold
    and next_side == "sell"
):
    # 在庫が buy 偏重 → sell ガードを無視して売り注文を許可
    logger.info(
        "[171# InvSkew bypass] trending_sell_skip を無視 "
        "(inv_imbalance=%.3f >= %.3f)",
        _inv_imbalance, _inv_bypass_threshold,
    )
else:
    # 通常の trending_sell_skip ロジック (既存)
    ...
```

### 3.3 sell_dynamic_kill バイパス (orchestrator L822+)

```python
_inv_bypass_sell_kill = (
    _inv_imbalance >= _inv_bypass_threshold
    and next_side == "sell"
)
# sell_dynamic_kill の条件に `not _inv_bypass_sell_kill` を追加
```

### 3.4 config 調整

| パラメータ | 旧値 | 新値 | 理由 |
|-----------|------|------|------|
| `balance_forced_rescue_offset_mult` | 2.0 | **1.3** | 2.0× では rescue fill が約定不可能 |
| `sell_dynamic_kill.trending_up` | -0.1 | **-0.3** | -0.1 は全 sell をほぼ遮断 |
| `max_consecutive_trending_sell_skip` | 20 | **10** | 安全弁を早期発動 |
| `sell_guard_inv_bypass_threshold` | (新規) | **0.3** | InvSkew バイパス閾値 |

---

## §4 EV_per_cycle 実装 — Codex R3 + Gemini 9.4

### 4.1 定義

$$EV_{\text{per\_cycle}} = P(\text{fill}) \times \overline{\text{PnL}}_{\text{filled}} \quad [\text{bps/cycle}]$$

### 4.2 guard_value (ガード価値指標)

$$\text{guard\_value} = EV_{\text{per\_cycle}} - \overline{\text{hindsight}}_{\text{blocked}}$$

- `guard_value > 0`: ガードがブロックしたものは EV 以下 → ガード有効
- `guard_value < 0`: ガードがブロックしたものの方が良かった → ガード有害

### 4.3 集計軸

| 軸 | 内容 |
|----|------|
| **overall** | 全サイクル |
| **by_regime_side** | regime × side (e.g., `trending_up_sell`) |
| **by_guard** | カテゴリ別 (filled, H1_skip_gate, H5_balance_forced, H8_regime_guard, ...) |

### 4.4 実装場所

`scripts/v460/analysis/hindsight_filter.py`:
- `EvPerCycleSummary` (TypedDict): 集計結果の型
- `EvPerCycleReport` (TypedDict): 全体レポートの型
- `_compute_ev_summary()`: 共通集計ロジック
- `_analyze_ev_per_cycle()`: regime×side / guard カテゴリ別算出
- `_print_report()`: コンソール出力追加
- `main()`: JSON 出力に `ev_per_cycle` キー追加

---

## §5 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/fill_config.py` | `sell_guard_inv_bypass_threshold` フィールド + YAML 解析 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | InvSkew bypass (trending_sell_skip, sell_dynamic_kill) |
| `scripts/v460/lib/config_hot_reload.py` | hot-reload whitelist 追加 |
| `configs/v460/fill_test.yaml` | 4 パラメータ変更 |
| `scripts/v460/analysis/hindsight_filter.py` | EV_per_cycle 分析追加 |
| `tests/unit/v460/test_169_c1_c3_c4_config.py` | 3 テスト追加/修正 |
| `docs/v460/172_ph2_fix_guard_paradox_and_ev_per_cycle.md` | 本文書 |

---

## §6 テスト結果

```
48 passed, 0 failed (test_169_config_hot_reload + test_169_ranging_buy_skip_and_metrics + test_169_c1_c3_c4_config)
```

mypy: 新コードの型エラー 0 件 (既知のモジュール名衝突のみ)。

---

## §7 期待効果

1. **正フィードバックループ断絶**: 在庫偏重時に sell ガードを自動バイパス → balance_forced_skip の sell 100% 偏りが解消
2. **rescue fill の実効化**: offset 2.0× → 1.3× で約定可能性が大幅上昇
3. **sell 機会確保**: trending_sell_skip 安全弁 20→10 + 閾値 -0.1→-0.3 で過剰ブロック緩和
4. **定量的ガード評価**: EV_per_cycle で guard_value を数値化、有害ガードの特定が可能に
5. **Hot-Reload 対応**: `sell_guard_inv_bypass_threshold` は運用中に動的調整可能

---

## §8 170# §10.5 アクションリスト消化状況

| 優先度 | アクション | ステータス |
|--------|-----------|-----------|
| **P0** | balance_forced_skip フィードバックループ対策 | ✅ §2-3 |
| **P0** | Gate Exception-1 ルール 000# 追記 | ❌ 未着手 |
| **P1** | EV_per_cycle 算出ロジック実装 | ✅ §4 |
| **P1** | sell offset floor 動的化検討 | ❌ 未着手 |
| **P1** | DailyDrawdownGuard 機会損失メトリクス | ❌ 未着手 |
| **P2** | 日次レポート 3 系列固定出力 | ❌ 未着手 |
| **P2** | cancel_reason Literal 型化 | ❌ 未着手 |
| **P2** | CircuitBreaker 統合 | ❌ 未着手 |
| **P3** | StatisticalValidator A/B テスト統合 | ❌ 未着手 |
