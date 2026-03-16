# 191# Guard Layer 複雑性分析 + 簡素化提案

> **目的**: 190# 修正後の実稼働ログを深層分析し、7+ 重層ガードの相互作用を解剖。  
> **前提**: 複雑化せずに改善する。削減・統合を優先。  
> **対象データ**: 2026-02-28 13:03–15:54 JST (約2.8時間, 70 cycles)

---

## 1. 現状サマリー

### 1.1 パフォーマンス概要

| 指標 | 値 | 評価 |
|---|---|---|
| 総 Cycle 数 | 70 | — |
| Fill 数 | 12 | — |
| **Fill Rate** | **17.1%** | ❌ 壊滅的に低い |
| Unfill (timeout) | 7 (10.0%) | 許容範囲 |
| Sum PnL | +11.10 bps | 表面上プラス |
| Mean PnL | +0.93 bps | — |
| Median PnL | +0.05 bps | ほぼゼロ |
| Max PnL | +32.06 bps | 外れ値 |
| Min PnL | -15.50 bps | 外れ値 |
| StdDev PnL | ~11.3 bps | ❌ 極めて高分散 |

**PnL 分布**: `[0.13, -3.18, 2.42, -1.85, -15.50, 0.45, -0.03, 0.28, -0.51, 32.06, 3.12, -6.29]`  
→ 2つの外れ値 (+32.06, -15.50) が全 PnL の 80%+ を支配。再現性なし。

### 1.2 ガード発動統計

| ガード | 発動数 | Cycle比 | 根拠 | 導入# |
|---|---|---|---|---|
| **ev_weighted_skip (buy)** | 11 | 15.7% | ML: 常時負スコア | 188# |
| **ev_weighted_skip (sell)** | 16 | 22.9% | ML: 常時負スコア | 188# |
| **B1' ranging_buy_low_vol** | 12 | 17.1% | vol_ratio < 0.75 | 169# |
| velocity_sell_skip | 8 | 11.4% | > 6.0 bps | 165# |
| narrow_spread (< 1000 JPY) | 8 | 11.4% | 実勢 spread 狭小 | 137# |
| sell_guard (spread > 5000) | 6 | 8.6% | 瞬間ワイドスプレッド | 088# |
| velocity_buy_skip | 3 | 4.3% | < -6.0 bps | 183# |
| balance_insufficient | 77 events | — | JPY 枯渇 | 091# |
| safety_valve (190# A) | 2 | 2.9% | 5連続 skip 解除 | 190# |
| stale_order_reprice | 4 | 5.7% | price drift | 096# |

**ev_weighted 合計: 27/70 = 38.6%** — 最大の blocking 要因。

### 1.3 致命的問題: ev_weighted スコア全件負

ev_weighted が評価した全 27 件のスコア:
- **Buy**: -0.142, -3.788, -2.888, -3.099, -0.924, -1.719, -2.189, ... (全て負)
- **Sell**: -1.287, -0.434, -3.166, -1.060, -2.930, -0.061, -0.912, -0.026, ... (全て負)

→ **ev_weighted threshold (0.0/−0.5) を超えた自然なPASSはゼロ**。全PASSはsafety valve (190# A) 経由。

### 1.4 online_monitor 警告

```
pass_mean_pnl=-0.380bps < threshold=-0.3bps [DEGRADED]
buy: pass_pnl=-0.642bps, win_rate=47.6%
sell: pass_pnl=-0.075bps, win_rate=38.9%
skip_precision=96.7%
```
→ skip precision は高い (正しくスキップしている) が、PASS した取引が平均損失。モデルの予測能力が片方向に偏っている。

---

## 2. ガードレイヤー完全マップ

### 2.1 レイヤー構造 (上から評価順)

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Balance Check (fill_loop_orchestrator.py)         │
│    ├─ JPY/BTC insufficient → side switch → balance_forced   │
│    └─ one_sided_balance flag 伝搬                           │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Regime Rules (fill_loop_orchestrator.py)          │
│    ├─ 133# unknown_regime buy skip                          │
│    ├─ 169# B1' ranging_buy_low_vol (vol < 0.75)            │
│    ├─ 155# trending_sell_skip (現在 disabled)               │
│    ├─ 157# buy_dynamic_kill (rolling PnL)                   │
│    └─ 133# sell_dynamic_kill (rolling PnL)                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Spread/Price Guard (fill_cycle_executor.py)       │
│    ├─ 088# sell_guard (spread > max_spread_jpy)             │
│    ├─ 137# narrow_spread_pause (< narrow_spread_pause_bps)  │
│    └─ 158# min_spread_jpy check (spread < 1000 JPY)        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: maker_price rejection (maker_price.py)            │
│    └─ sell_guard: spread > max_spread check                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: SkipGate ML (skip_gate_evaluator.py)              │
│    ├─ 124# rule_skip_unknown_sell                           │
│    ├─ 165# velocity_sell_skip (> 6.0 bps)                   │
│    ├─ 165# velocity_buy_skip (< -6.0 bps)                   │
│    ├─ 158# hour_offset (UTC hour → threshold 調整)          │
│    ├─ 183# narrow_spread_offset (spread < 2000 → 厳格化)    │
│    ├─ ML model prediction (primary: side-specific model)    │
│    └─ 188# ev_weighted (w30×pnl30 + w120×pnl120)           │
│        ├─ 190# A: safety valve (5 consecutive → PASS)       │
│        └─ 190# B: one_sided threshold shift (-1.0)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Post-order (order_monitor.py)                     │
│    └─ stale_order reprice (price drift > threshold)         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 重要な観察

1. **ガード数**: 少なくとも **12の独立したスキップ条件** + reprice
2. **コード分散**: 3ファイル (orchestrator 1309行, executor 829行, evaluator 968行)
3. **設定パラメータ**: 50+ のガード関連 YAML パラメータ
4. **重複カバレッジ**: B1' と ev_weighted は buy side で二重に blocking
5. **逆相関ガード**: sell_guard (spread > 5000) と narrow_spread (spread < 1000) — 中間帯のみ通過

---

## 3. 根本原因分析

### 3.1 問題A: ev_weighted モデルの恒常的負スコア

**原因**: ev_weighted は `w30 * pnl30 + w120 * pnl120` を合成するが、both horizons のモデルが負 PnL を一貫して予測している。  
**含意**: ML モデルが「この市場環境ではどの取引も損失する」と判断している可能性。または、学習データと現在の市場環境の乖離。
**影響**: ev_weighted が実質的に **全取引無条件ブロック層** と化している。

### 3.2 問題B: ガード層の重積による乗算的遮断

各ガードが独立に 10–20% の取引を遮断する場合、6 ガードの通過率:  
$(1 - 0.15)^6 \approx 0.377$ → **理論通過率 37.7%**

実測の **17.1%** はこれを大幅に下回る → ガード間の正の相関（同じ取引を複数ガードが同時遮断）が存在。

### 3.3 問題C: stale_order reprice の逆選択リスク

Cycle 4876: stale_order が 10294148 → 10302100 JPY (8.5bps 上方ドリフト) に追随 reprice → 約定後 -15.50bps  
→ **reprice は市場を追いかけるため、逆選択に晒される**。maker bot として「待つ」べき局面で「追う」行動は本質的にリスキー。

### 3.4 問題D: PnL 分布の高分散 (StdDev 11.3bps)

12 fills のうち 2 つの外れ値 (+32.06, -15.50) が支配。残り 10 fills の Sum は -5.67 bps。  
→ **外れ値を除いた真の期待値はマイナス**。高分散は運任せの構造を示唆。

---

## 4. 簡素化提案

### 4.1 提案 S1: ev_weighted の一時無効化 (YAML変更のみ)

**変更**: `ev_weighted_enabled: false`  
**根拠**: 100% 負スコア → 事実上「全 skip」ルール。safety valve で 5 cycle に 1 度だけ通す構造は ML 判定と呼べない。  
**リスク**: primary model のみで判定 → skip rate 低下 → 元々 skip_precision=96.7% なので品質維持。  
**効果**: 27/70 cycles (38.6%) が解放 → Fill Rate 改善  
**復帰条件**: モデル再訓練で ev_weighted スコアが正値を含む分布になったら再有効化。

### 4.2 提案 S2: velocity_skip 閾値の緩和 (6.0 → 10.0 bps)

**変更**: `sell_velocity_skip_threshold_bps: 10.0`, `buy_velocity_skip_threshold_bps: -10.0`  
**根拠**: 6.0bps は Volatility Guard の velocity_threshold_bps (12.0) より厳しい。VG は offset boost で対処する設計で、velocity_skip は完全遮断。二重防御。  
**リスク**: 6–10bps 帯の velocity 取引を通す → VG offset boost で保護される  
**効果**: 11/70 cycles (15.7%) の一部が解放

### 4.3 提案 S3: B1' ranging_buy_low_vol の vol_ratio 閾値引下げ (0.75 → 0.55)

**変更**: `low_vol_threshold: 0.55`  
**根拠**: vol_ratio 0.55–0.75 の帯域が B1' で全遮断されるが、ev_weighted (S1で無効化する場合) を通した ML 判定の方が精度が高い。  
**リスク**: ranging_buy 再開 → 元々の懸念 (全損失の 69%) が再燃する可能性  
**効果**: vol_ratio > 0.55 の ranging buy が ML 判定に委ねられる。12/70 の一部解放。

### 4.4 提案 S4: stale_order reprice 回数制限 (max 1)

**変更**: reprice の最大回数を 2 → 1 に制限  
**根拠**: Cycle 4936 では 2 回の reprice (4.1bps + 11.0bps ドリフト) が連続。2 回目は市場追随が過剰。  
**リスク**: Fill Rate 微減 (timeout 増)  
**効果**: 逆選択リスクの低減

### 4.5 優先度マトリクス

| 提案 | 効果 | リスク | 実装コスト | 推奨 |
|---|---|---|---|---|
| S1: ev_weighted 無効化 | ◎ 大 | △ 低 | YAML 1行 | **最優先** |
| S2: velocity 閾値緩和 | ○ 中 | △ 低 | YAML 2行 | 推奨 |
| S3: B1' 閾値引下げ | ○ 中 | ○ 中 | YAML 1行 | 要検討 |
| S4: reprice 回数制限 | △ 小 | △ 低 | コード+YAML | 要検討 |

---

## 5. コード複雑性レビュー (外部 AI レビュー向け)

### 5.1 ファイルサイズ

| ファイル | 行数 | 責務 |
|---|---|---|
| fill_loop_orchestrator.py | 1308 | メインループ + Layer 0/1 ガード |
| fill_config.py | 1163 | 設定定義 (dataclass) |
| skip_gate_evaluator.py | 967 | Layer 4: ML + velocity + ev_weighted |
| fill_cycle_executor.py | 828 | Layer 2/3: spread + 注文執行 |
| maker_price.py | 762 | 価格計算 + VG + inv_skew |

**Total core**: ~5,028 行。巨大ではないが、ガードロジックが全ファイルに分散。

### 5.2 アーキテクチャ上の懸念

1. **ガードの分散**: 同種の「取引するかしないか」判定が 3 ファイル・5 レイヤーに散在
2. **設定膨張**: fill_config.py の 1,163 行の大半がガードパラメータ。50+ の閾値。
3. **安全弁の乱立**: ev_weighted safety valve (190#), trending_sell safety valve (158#), balance_forced override, inv_bypass — それぞれ ad hoc に追加
4. **履歴コメント依存**: `# 169# B1'`, `# 155# §9`, `# 190# A` 等のセッション番号注釈が 100+ 箇所。文脈理解にセッション内容の知識が必要。

### 5.3 良い点

1. **型安全**: Protocol ベースの依存性注入 (`_SkipGateLike`, `_SkipDecisionLike`)
2. **テスト**: 2467 件の test suite (190# 時点)
3. **モデル hot-reload**: 126# のファイルハッシュベース自動検出
4. **設定外部化**: fill_test.yaml に全パラメータ集約 (コードハードコード少)
5. **ログ設計**: 構造化ログ with セッション番号→原因追跡が容易

---

## 6. 中期的な構造改善案 (実装は今回行わない)

### 6.1 ガード統合パイプライン

現在の分散ガードを単一の `TradeDecisionPipeline` に集約:

```python
# 概念設計
class TradeDecisionPipeline:
    """全ガードを直列パイプラインで評価。"""
    guards: list[Guard]  # 優先順位付きリスト
    
    async def should_trade(self, ctx: TradeContext) -> Decision:
        for guard in self.guards:
            result = guard.evaluate(ctx)
            if result.block:
                return Decision(trade=False, reason=guard.name)
        return Decision(trade=True)
```

**メリット**: ガード追加・削除が局所的、テスト容易、ログ統一。  
**コスト**: 大規模リファクタリング (3000+ 行影響)。ph3 以降の課題。

### 6.2 モデル品質ゲート

ev_weighted のような ML ガードに「自己評価」機構を設け、スコア分布が全件同一方向の場合は自動無効化:

```python
if all(score < 0 for score in recent_scores[-20:]):
    logger.warning("ev_weighted auto-disabled: 20 consecutive negative scores")
    return None  # ev_weighted bypass
```

---

## 7. 即座の対応 (191# 実施内容)

**本文書は分析レポートのみ。コード変更は行わない。**  
YAML パラメータ変更は次セッション (192#) でユーザー確認後に実施予定。

---

## 付録A: 全 PnL 時系列

| Cycle | Time (JST) | Side | PnL (bps) | 備考 |
|---|---|---|---|---|
| 4868 | 13:06 | buy | +0.13 | 190# 修正後初 fill |
| 4872 | 13:18 | sell | -3.18 | — |
| 4874 | 13:22 | buy | +2.42 | — |
| 4876 | 13:32 | buy | -1.85 | — |
| 4878 | 13:41 | buy | -15.50 | ❌ stale_order reprice 8.5bps 追随 |
| 4884 | 13:54 | sell | +0.45 | — |
| 4888 | 14:01 | buy | -0.03 | — |
| 4891 | 14:11 | sell | +0.28 | — |
| 4897 | 14:29 | sell | -0.51 | — |
| 4910 | 15:20 | buy | +32.06 | ✅ safety valve PASS → 大幅有利 |
| 4921 | 15:34 | sell | +3.12 | — |
| 4930 | 15:47 | buy | -6.29 | stale_order reprice 7.6bps 追随 |

## 付録B: 現行 YAML ガードパラメータ (主要)

```yaml
# ev_weighted
ev_weighted_enabled: true           # → S1: false 提案
ev_w30: 0.4
ev_w120: 0.6
ev_max_consecutive_skip: 5
ev_one_sided_threshold_shift: -1.0
pnl_threshold: -0.5

# velocity
sell_velocity_skip_threshold_bps: 6.0   # → S2: 10.0 提案
buy_velocity_skip_threshold_bps: -6.0   # → S2: -10.0 提案

# B1'
low_vol_threshold: 0.75                 # → S3: 0.55 提案
skip_ranging_buy_low_vol: true

# spread
min_spread_jpy: 1000
narrow_spread_pause_bps: 3.0
sell_guard max_spread: 5000
```
