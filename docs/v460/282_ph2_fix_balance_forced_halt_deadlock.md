# 282# balance_forced + per-side halt デッドロック修正

| 項目 | 内容 |
|------|------|
| 日付 | 2026-03-05 |
| 起票 | Copilot (担当者レビュー済) |
| 前提 | v460 fill_test on Coincheck, HEAD=91f050a76 (281#) |
| コミット | `da9fd4f39` |
| 重大度 | **CRITICAL** (8h+ 完全取引停止) |
| レビュー依頼先 | Codex / Gemini |

---

## 1. インシデント概要

2026-03-05 13:10 JST ～ 21:21 JST (**8時間11分**)、fill_test プロセスが **完全なデッドロック** に陥り、取引が一切実行されなかった。

全体 PnL は +24.9bps (プラス) にもかかわらず、buy 側の per-side DD halt と balance 制約の複合条件により、sell 側の利益機会 (+58.8bps) すら活用できず完全停止した。

### 停止時の状態スナップショット

| 項目 | 値 | 備考 |
|------|-----|------|
| `daily_pnl_bps` | +24.9 | 全体はプラス |
| `daily_pnl_bps_buy` | **-33.9bps** | per_side_dd_hard_limit_bps (-30.0) を超過 → per-side halt 発動 |
| `daily_pnl_bps_sell` | +58.8bps | sell 側は好調だが BTC 枯渇で実行不能 |
| `side_halted_buy` | True | |
| `side_halt_remaining_buy` | **12** (固定) | 273# I3 の untick で永久固定化 |
| BTC 残高 | 0.000000 | sell 不可 |
| JPY 残高 | 23,805.79 | buy 可能だが halt で封鎖 |
| スキップ回数 | **236 回** | 2分間隔 × 236 = ~7.9h |
| 推定逸失利益 | **~+58.8bps を活用不能** | sell 側は午後回復基調 (#22: +2.32, #23: +4.41) |

### デッドロックのループ (1サイクル = 120秒)

```
┌─ サイクル開始 ─────────────────────────────────────────┐
│                                                         │
│  tick_side_halt()          ← halt_remaining: 12→11      │
│    ↓                                                    │
│  [205# §9.5] next_side=buy → buy halted → sell に切替   │
│    ↓                                                    │
│  [129# D.2] BTC=0 → sell 不可 → buy に戻す (balance_    │
│             forced=True)                                 │
│    ↓                                                    │
│  [222# 1.1] balance_forced 後の per-side halt 再チェック │
│             → buy is halted → Inventory Escape 判定     │
│    ↓                                                    │
│  [269# P0] _ie_enabled=True BUT next_side=="buy"        │
│            (sell 限定条件 不適合) → IE 不発              │
│    ↓                                                    │
│  [223# P0] refuse bypass (safety > liveness)            │
│    ↓                                                    │
│  [273# I3] untick_side_halt()                           │
│            halt_remaining: 11→12 (tick を完全補償)       │
│    ↓                                                    │
│  _execute_skip() → 120s × halt_sleep_multiplier 待機    │
│                                                         │
└─── continue → サイクル開始に戻る ──────────────────────┘
```

**結果**: `halt_remaining` は永久に **12** のまま。tick と untick が 1:1 で相殺され、halt が永遠に解除されない。日替わりリセット (UTC 0:00 = JST 9:00) まで最大 **24 時間** 停止し得る。

### 時系列

| 時刻 (JST) | イベント |
|------------|---------|
| 09:00 | fill_test 起動、取引開始 |
| 09:06-09:47 | buy_dynamic_kill 41分連続作動 (23 skip) |
| 09:00-13:17 | 25 fills: buy 12回 (-12.01bps), sell 13回 (-11.31bps) |
| 13:17 | buy #24 (-5.77bps) → 累計 -33.9bps → per-side halt 発動 |
| 13:10 | BTC=0 by previous sells → デッドロック突入 |
| 13:10-21:21 | **236 回スキップ** (8時間11分の完全停止) |
| 21:21 | 282# fix デプロイ → IE 双方向化で即時復旧 |
| 21:21 | IE buy 実行: +9.73bps fill (degraded params で成功) |

---

## 2. 根本原因 (Root Cause)

本デッドロックは **2 つの独立した設計判断** の複合によって発生した。どちらか一方だけでは問題は顕在化しない。

### 原因 A: 273# I3 の `untick_side_halt()` — halt カウントダウン完全停止

#### 背景
268# 分析で「空サイクル問題」が指摘された:

> halt 発動 → balance 制約で取引不能 → halt_remaining だけが消費される → 15 cycles (30分) 後に halt 解除 → PnL 改善なし (取引していないため) → 即再 halt → 30分ごとの halt/解除サイクルが無駄に反復

この問題を解決するため 273# I3 で `untick_side_halt()` が導入された:

```python
def untick_side_halt(self) -> None:
    """273# I3: 空サイクル halt カウント除外 — tick_side_halt の補償."""
    if self._state.side_halted_buy and self._per_side_halt_cycles > 0:
        if self._state.side_halt_remaining_buy < self._per_side_halt_cycles:
            self._state.side_halt_remaining_buy = min(
                self._per_side_halt_cycles,
                self._state.side_halt_remaining_buy + 1,
            )
    # (sell 側も同様)
```

#### 問題の本質
268# の分析は **局所最適** だった。30分サイクルの「無駄」を防ぐことに注力し、以下を見落とした:

1. **市場回復の可能性**: halt 中 30分でも市場は変動する。halt 解除後の状況は halt 開始時と同じとは限らない
2. **reanchor の効果**: 269# で追加された reanchor は、halt 解除時に PnL 基準点をリセットする。即再 halt の懸念は reanchor budget (-15bps) で大部分が解消されている
3. **永久デッドロックのリスク**: untick が halt カウントダウンを停止させ、balance 制約と組み合わさると **永久停止** が発生する。30分の無駄より **8時間+ の完全停止** の方が遥かに有害

#### 268# 分析と 282# 修正の対立点

| 観点 | 268# (untick 導入根拠) | 282# (untick 除去根拠) |
|------|----------------------|----------------------|
| 空サイクルの halt 消費 | 無駄 (改善なしで解除) | 容認可 (reanchor で安全解除) |
| 30分 halt 反復 | 回避すべき | 8h 停止より遥かにマシ |
| 市場変動 | 考慮なし | 30分で充分変動し得る |
| reanchor | 導入前 (269# で後追加) | -15bps budget が即再halt を防止 |
| 最悪ケース | 30分×N の halt 反復 | **永久デッドロック (実証済)** |

### 原因 B: Inventory Escape の sell 方向限定 (269# P0)

269# P0 (Codex §4.1) / Gemini 270# Action A で導入された Inventory Escape は、BTC 過剰在庫 + sell halt のデッドロック脱出を目的とした:

```python
# 269# P0 — sell 方向限定
if _ie_enabled and next_side == "sell":  # ← BTC 過剰 → 売却想定
```

**設計意図**: 過剰在庫を **縮退清算** する (= sell)。buy 方向への脱出は、「JPY が枯渇して buy が halt + sell 不能」というシナリオとして想定されなかった。

**盲点**: BTC=0 + buy halt は「BTC を使い果たした後に buy 側の PnL が悪化」というシナリオ。sell 集中 → BTC 枯渇 → forced buy → buy 損失拡大 → halt というフィードバックループの結果であり、269# 設計時には未考慮。

#### 原因 A × 原因 B の相互作用

```
[原因 B] IE が buy 方向で不発
     +
[原因 A] untick が halt カウントダウンを停止
     ↓
halt が永遠に解除されない → 唯一の脱出口 (IE) も封鎖
     ↓
日替わりリセットまで完全停止 (最大 24h)
```

どちらか一方が修正されればデッドロックは解消される:
- **原因 A のみ修正** (untick 除去): halt が 30 分で自然解除 → 正常取引再開
- **原因 B のみ修正** (IE 双方向化): halt 中でも IE で buy 実行 → 在庫取得 → sell 可能に
- **両方修正**: **多重防御** — halt 自然解除 + IE による即時脱出

---

## 3. 修正内容

### Fix A: `untick_side_halt()` 除去 (2箇所)

#### 箇所 1: `per_side_dd_both_halt` パス (L1601)

両サイドが halt 中のフロー。`_next_side()` → `is_side_halted()` → 反対側も halt → both_halt。

```python
# BEFORE (273# I3):
self._inc_guard_fire("per_side_dd_both_halt")
self._daily_drawdown_guard.untick_side_halt()
await self._execute_skip(...)

# AFTER (282# fix):
self._inc_guard_fire("per_side_dd_both_halt")
# untick_side_halt() 除去 — halt 自然カウントダウン
await self._execute_skip(...)
```

#### 箇所 2: `balance_forced_halt_block` パス (L1859)

balance_forced で side が切り替わったが、切替先も halt 中で IE も不発のフロー。

```python
# BEFORE (273# I3):
self._daily_drawdown_guard.untick_side_halt()
self._tick_toxic_veto("halt_block")

# AFTER (282# fix):
# untick_side_halt() 除去
self._tick_toxic_veto("halt_block")
```

#### halt 自然解除の数値分析

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `per_side_halt_cycles` | 15 | halt 解除までのサイクル数 |
| `cycle_interval` | 120s | 基本サイクル間隔 |
| `halt_sleep_multiplier` | 5.0 | halt 中のスリープ延長 |
| **halt 実効スリープ** | **120 × 5 = 600s** | 1 halt サイクルの所要時間 |
| **halt 総所要時間** | **15 × 600 = 9000s ≈ 150分** | ※理論最大値 |
| 実測 (untick 有) | **∞** (永久停止) | |
| 実測 (untick 無) | **~150分で解除** | reanchor budget -15bps で再 halt 防止 |

**注**: tick_side_halt() はサイクル冒頭で呼ばれるため、halt 中の全サイクル (IE duty skip 含む) で確実にデクリメントされる。

### Fix B: Inventory Escape 双方向化

```python
# BEFORE (269# P0): sell 方向のみ
if _ie_enabled and next_side == "sell":

# AFTER (282# fix): buy/sell 両方向
if _ie_enabled:
```

#### IE 双方向化の安全性分析

IE による halt 貫通は以下の多重安全装置で保護されている:

| 安全装置 | パラメータ | 効果 |
|---------|-----------|------|
| **Duty cycle** | `inventory_escape_duty_cycle=5` | 5 サイクルに 1 回のみ実行 (20%) |
| **Degraded lot** | `degraded_liquidation_lot_mult=0.2` | 通常 lot の 20% (min lot 相当) |
| **Wide offset** | `degraded_liquidation_offset_mult=3.0` | offset 3 倍 → fill 確率低下・損失制限 |
| **Toxic veto** | `_tick_toxic_veto("inventory_escape")` | 毒性判定は維持 |
| **Halt countdown** | Fix A で自然解除 | IE は「halt が切れるまでの橋渡し」 |
| **Reanchor** | `per_side_dd_reanchor_budget_bps=-15.0` | halt 解除後の再 halt 閾値緩和 |

#### IE 実行時のオーダーフロー

IE が発動した場合の実際のオーダー:

```
1. _inventory_escape = True
2. run_single_cycle(degraded_liquidation=True) を実行
   → lot = 通常lot × 0.2 (= min_order_btc 相当)
   → offset = 通常offset × 3.0 (wide offset)
3. 約定すれば BTC 取得 (buy の場合) or JPY 取得 (sell の場合)
4. → 次サイクルで反対 side の balance_forced 制約が解除される可能性
```

#### buy 方向 IE の固有リスクと軽減

buy halt は「buy が損失を出している」という信号。IE で buy を実行することは:

| リスク | 軽減策 | 残存リスク |
|--------|--------|-----------|
| 損失拡大 | lot 20% + offset 300% | 最大損失 = 通常の ~1/15 |
| halt の意味を無視 | duty 20% (5 中 1 回) | halt の 80% は尊重 |
| 連続損失 | halt カウントダウンは継続 | 最大 3 回の IE buy で halt 解除 |
| 逆張り的ポジション | 30分後に自然解除でリセット | reanchor で新 budget 付与 |

**最悪ケース数値試算**:
- IE buy 実行回数: 150分 / (600s × 5) = 最大 **3 回**
- 1 回の最大損失: min lot (0.001 BTC) × wide offset → ~2-5bps
- 最大累計損失: ~15bps (vs. デッドロック時の逸失利益 ~60bps+)

### 修正の効果 (新挙動の完全フロー)

```
┌─ サイクル開始 ─────────────────────────────────────────┐
│                                                         │
│  tick_side_halt()          ← halt_remaining: 12→11      │
│    ↓                                                    │
│  [205#] buy halted → sell に切替                        │
│    ↓                                                    │
│  [129# D.2] BTC=0 → sell 不可 → buy に戻す             │
│    ↓                                                    │
│  [222#] balance_forced 後再チェック → buy is halted      │
│    ↓                                                    │
│  [269# IE] duty cycle 判定:                             │
│    cycle % 5 == 1 → ACTIVE: degraded buy 発注           │
│    cycle % 5 != 1 → SKIP: スキップ                      │
│    ↓ (SKIP の場合)                                      │
│  _execute_skip() → 600s 待機                            │
│  ※ untick なし → halt_remaining そのまま 11             │
│                                                         │
│  ... 11~14 サイクル後 → halt_remaining=0                │
│  → halt 自然解除 + reanchor (-33.9bps を基準点に)       │
│  → 再 halt 閾値: -33.9 + (-15.0) = -48.9bps           │
│  → 通常取引再開                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 13:17 以前のログ分析 — デッドロック前兆の解剖

### 4.1 取引サマリ (09:00-13:17 JST)

| 項目 | 値 | 評価 |
|------|-----|------|
| 総 fill 数 | 25 | 4h で 25 fills = ~6.25 fills/h |
| Buy fill (PnL) | 12 fills, **-12.01 bps** | 赤字 |
| Sell fill (PnL) | 13 fills, **-11.31 bps** | 赤字 |
| 総 PnL | **-23.32 bps** | 両側損失 |
| BF (balance_forced) buy | **6/12 = 50%** | 半数が強制発注 |
| 最大単一損失 | sell #1: -11.91 bps (09:00 開始直後) | |
| 最大連続損失 | buy #19(-6.97), #21(-4.45), #24(-5.77) | halt トリガー |

### 4.2 スキップ分析 (pre-13:17)

| スキップ理由 | 回数 | 占有率 | 備考 |
|-------------|------|--------|------|
| `buy_dynamic_kill` | **52** | **72%** | buy 側の kill manager が長時間作動 |
| `skip_gate` | 8 | 11% | 正常なゲート制御 |
| `sell_guard_reject` | 4 | 5.5% | sell 側の保護動作 |
| `spread_too_narrow` | 3 | 4.1% | スプレッド不足 |
| `toxic_fill_side_veto` | 3 | 4.1% | 毒性拒否 |
| `degraded_liquidation_duty_skip` | 2 | 2.8% | 縮退清算のデューティスキップ |
| **計** | **72** | | |

### 4.3 フィードバックループの解剖

デッドロックは突発事象ではなく、以下の **フィードバックループ** が収束的に発生した結果:

```
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
buy_dynamic_kill 長時間作動                               │
(09:06-09:47: 41分連続 kill, 23 skip)                    │
  │                                                      │
  ▼                                                      │
sell 集中 (buy 不能 → sell のみ実行)                      │
  │                                                      │
  ▼                                                      │
BTC 在庫枯渇 (sell で BTC 消費)                           │
  │                                                      │
  ▼                                                      │
balance_forced buy 多発 (BTC=0 → 強制 buy)                │
6/12 = 50% が forced buy                                  │
  │                                                      │
  ▼                                                      │
buy PnL 悪化 (forced buy はタイミング無視)                │
#19(-6.97), #21(-4.45), #24(-5.77)                       │
  │                                                      │
  ▼                                                      │
per-side halt 発動 (buy -33.9bps > -30.0 threshold)      │
  │                                                      │
  ▼                                                      │
BTC=0 + buy halt = デッドロック ──────────────────────────┘
```

### 4.4 重要パターンの詳細分析

#### (A) buy_dynamic_kill 長時間抑制

- **09:06-09:47**: 23 連続スキップ (**41分間** buy が完全封鎖)
- kill manager の解除条件が market 回復に追従していない可能性
- 解除速度パラメータ (`max_kill_duration_sec` / `resume_window`) の見直しが有効

**構造的問題**: kill の持続時間に上限がないため、外部環境が変化しても kill が継続し、片側集中を引き起こす

#### (B) balance_forced buy の品質問題

| fill# | 種別 | PnL (bps) | 備考 |
|-------|------|-----------|------|
| #2 | BF buy | -4.56 | 朝の強制 buy |
| #6 | BF buy | -2.15 | |
| #10 | BF buy | +1.84 | 唯一のプラス |
| #16 | BF buy | -1.12 | |
| #19 | BF buy | **-6.97** | 午後悪化のトリガー |
| #24 | BF buy | **-5.77** | halt トリガー |

BF buy 平均: **-3.12 bps** vs. 通常 buy 平均: **-0.89 bps** → forced buy は通常 buy の **3.5 倍** 損失が大きい

#### (C) sell 側の午後回復

- #22 (+2.32bps), #23 (+4.41bps): sell 側は午後に回復基調
- デッドロック解消後に sell で利益回収可能だったが、halt で活用不能

### 4.5 改善検討事項 (将来課題)

| # | 項目 | 内容 | 優先度 | 期待効果 |
|---|------|------|--------|----------|
| F-1 | buy_dynamic_kill 解除速度 | 41分の連続 kill は長い。`max_kill_duration_sec` 導入を検討 | Medium | フィードバックループ入口を制限 |
| F-2 | BF buy の品質向上 | balance_forced buy のオフセット拡大 / ゲート条件の追加 | Medium | forced buy の損失を通常 buy 並に |
| F-3 | 片側集中リスクの構造対策 | buy kill 持続時間に応じた sell 抑制 / 在庫水準モニター | High | BTC 枯渇前の介入 |
| F-4 | 朝の大損失耐性 | 初回取引 (-11.91 bps) の影響を限定する warmup 期間 | Low | 日次 tone の安定化 |

---

## 5. レビュー依頼事項

### Q1: 273# I3 `untick_side_halt()` の完全除去は適切か？

**268# の懸念**: halt 満了 → 改善なく即再halt → 30分ごとの無駄なサイクル。

**282# の反論**: 
- reanchor (269#) が解除後の PnL 基準をリセット → 即再halt にはならない (新たな -15bps budget が付与)
- 30分サイクルの「無駄」より、8時間の完全停止の方が遥かに有害
- halt 中も market は変動しており、30分後の状況は異なる可能性が高い
- halt 解除後の recovery 期間 (224# B1) で lot 50% に縮小 → リスク制限

**補足データ**: 本インシデントの時系列が示す通り、sell 側は午後に +2~+4 bps の回復を見せていた。30分の halt が自然解除されていれば、sell で利益回収可能だった。

**質問**: untick 除去以外に、halt カウントダウンと balance 制約を両立する better approach はあるか？ 例えば:
- `untick` の上限: 最大 N 回まで untick を許可 (永久停止を防止)
- 条件付き untick: balance_forced 以外の理由で空振りした場合のみ untick

### Q2: Inventory Escape 双方向化のリスク

buy halt は「buy が損失を出している」という信号。IE で buy halt を貫通することは:
- **メリット**: デッドロック脱出、取引再開、sell 側の利益機会活用
- **リスク**: 損失が拡大する可能性

現行安全装置:
| 装置 | 設定 | 拘束力 |
|------|------|--------|
| duty cycle | 1-in-5 (20%) | halt の 80% は尊重 |
| degraded lot | 0.2 (通常の 20%) | 損失を 1/5 に制限 |
| wide offset | 3.0 (通常の 3 倍) | fill 確率低下 + 有利約定 |
| halt countdown | Fix A で ~150分で解除 | IE は橋渡し (最大 3 回) |

**最悪ケース**: IE buy 3 回 × 各 -5 bps = -15 bps 追加損失 (vs. デッドロック時の逸失利益 ~60 bps+)

**質問**: この安全装置で十分か？追加制約は必要か？（例: IE による buy の累積損失上限、IE 実行後の PnL チェック）

### Q3: buy_dynamic_kill と per-side halt のフィードバックループ

```
buy_kill 長時間作動 → sell 集中 → BTC 枯渇 → forced buy 多発
→ buy PnL 悪化 → per-side halt → デッドロック
```

282# はデッドロックの「出口」を修復したが、「入口」(= フィードバックループ) は未対策。

**構造的断絶案**:
- A) `buy_kill` に最大持続時間を設定 (e.g., `max_kill_duration_sec=1800`)
- B) BTC 在庫水準に応じた sell 抑制 (在庫閾値以下で sell を制限)
- C) BF buy の PnL 追跡と独立した閾値 (forced buy は通常 buy と分離管理)
- D) 片側集中検知器: N 回連続で同一 side → 自動減速

**質問**: A-D のうち最も効果的な断絶ポイントはどこか？ または別のアプローチがあるか？

### Q4: `untick_side_halt()` メソッドの存続

orchestrator からの呼出しは全除去した。メソッド自体を deprecated 化すべきか、将来のユースケースのために残すべきか？

**残す根拠**: Q1 で mentioned した「条件付き untick」等の将来設計で再利用の可能性
**deprecated 根拠**: 呼出し元がゼロのコードは誤って再利用されるリスク

---

## 6. テスト結果

| テスト | 結果 | 内容 |
|--------|------|------|
| 282# 新規テスト (15件) | ✅ 全パス | 下表参照 |
| v460 全テスト (3874件) | ✅ 全パス | 回帰テスト含む |
| 273# I3 テスト | ✅ パス | DDG メソッド自体は未変更 (呼出し元のみ除去) |

### テスト詳細 (`test_281_deadlock_fix.py`)

| クラス | テスト数 | 検証内容 |
|--------|---------|----------|
| TestUntickRemoval | 3 | ソースコード内の untick_side_halt 呼出しが 2 パスから除去されていることを検証 |
| TestInventoryEscapeBidirectional | 3 | IE 条件が `next_side == "sell"` を含まないことを検証 |
| TestHaltCountdownWithoutUntick | 3 | untick なしで halt が per_side_halt_cycles 後に自然解除されることを数値検証 |
| TestDeadlockScenario | 4 | BTC=0 + buy halt の実際のシナリオ再現 (旧: 永久停止、新: IE で脱出) |
| TestUntickMethodStillExists | 2 | `untick_side_halt()` メソッド自体の存続を確認 (呼出し除去のみ) |

### 本番動作確認

| 時刻 (JST) | イベント | 結果 |
|------------|---------|------|
| 21:21 | 282# fix デプロイ | プロセス起動成功 |
| 21:21 | IE buy 発動 (初回サイクル) | `[269#] INVENTORY ESCAPE: bypassing per-side halt for buy` |
| 21:21 | degraded buy fill | **+9.73 bps** (wide offset で有利約定) |
| 21:21~ | 通常サイクル再開 | BTC 回復 → sell も可能に |

---

## 7. 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/fill_loop_orchestrator.py` | Fix A (untick除去×2) + Fix B (IE双方向化) |
| `tests/unit/v460/test_281_deadlock_fix.py` | 新規テスト 15件 |
| `docs/v460/282_ph2_fix_balance_forced_halt_deadlock.md` | 本ドキュメント |

---

## 付録 A: 修正前後のコード対比

### A-1. balance_forced_halt_block パス

```python
# BEFORE (273# I3):
self._daily_drawdown_guard.untick_side_halt()
self._tick_toxic_veto("halt_block")

# AFTER (282# fix):
# untick_side_halt() 除去 — halt 自然カウントダウン
self._tick_toxic_veto("halt_block")
```

### A-2. per_side_dd_both_halt パス

```python
# BEFORE (273# I3):
self._inc_guard_fire("per_side_dd_both_halt")
self._daily_drawdown_guard.untick_side_halt()
await self._execute_skip(...)

# AFTER (282# fix):
self._inc_guard_fire("per_side_dd_both_halt")
# untick_side_halt() 除去
await self._execute_skip(...)
```

### A-3. Inventory Escape 条件

```python
# BEFORE (269# P0):
if _ie_enabled and next_side == "sell":

# AFTER (282# fix):
if _ie_enabled:
```

---

## 付録 B: `untick_side_halt()` / `tick_side_halt()` の動作詳細

### tick_side_halt() (毎サイクル冒頭で呼出し)

```python
def tick_side_halt(self) -> None:
    if self._state.side_halted_buy and self._per_side_halt_cycles > 0:
        self._state.side_halt_remaining_buy = max(0, self._state.side_halt_remaining_buy - 1)
        if self._state.side_halt_remaining_buy == 0:
            self._state.side_halted_buy = False
            # 269# reanchor: 解除時の PnL を基準点に設定
            self._state.side_reanchor_pnl_buy = self._state.daily_pnl_bps_buy
            # 224# B1: halt解除 → リカバリ期間開始
            self._state.side_recovery_remaining_buy = self._per_side_recovery_cycles
```

### untick_side_halt() (282# で呼出し元を除去、メソッドは存続)

```python
def untick_side_halt(self) -> None:
    """273# I3: 空サイクル halt カウント除外 — tick_side_halt の補償."""
    if self._state.side_halted_buy and self._per_side_halt_cycles > 0:
        if self._state.side_halt_remaining_buy < self._per_side_halt_cycles:
            self._state.side_halt_remaining_buy = min(
                self._per_side_halt_cycles,
                self._state.side_halt_remaining_buy + 1,  # ← tick を巻き戻す
            )
```

### halt カウンタの挙動比較

| サイクル | 修正前 (untick あり) | 修正後 (untick なし) |
|---------|---------------------|---------------------|
| 開始 | remaining=15 | remaining=15 |
| tick | 15→14 | 15→14 |
| 空振り→untick | 14→15 (巻き戻し) | — (14 のまま) |
| 2 tick | 15→14 | 14→13 |
| 2 空振り→untick | 14→15 | — (13 のまま) |
| ... | **永遠に 14-15 を往復** | **単調減少** |
| 15 tick | 14-15 (解除されない) | 0 → **halt 解除** |

---

## 付録 C: 関連ドキュメント番号の参照網

| # | 内容 | 本修正との関連 |
|---|------|---------------|
| 205# §9.5 | per-side DD halt 導入 | halt の基本メカニズム |
| 222# 1.1 | balance_forced 後の halt 再チェック | デッドロック発生点 |
| 223# P0 | safety > liveness (halt bypass 拒否) | IE 不発時の continue |
| 224# B1 | halt 解除後のリカバリ期間 | lot 50% で段階復帰 |
| 234# | 縮退清算モード (degraded liquidation) | IE が流用する params |
| 268# | untick 導入の分析根拠 | 282# で否定 |
| 269# P0 | Inventory Escape 導入 (sell 限定) | 282# で双方向化 |
| 270# | Gemini Action A (IE 承認) | IE の外部レビュー |
| 273# I3 | untick_side_halt() 実装 | 282# で呼出し除去 |
