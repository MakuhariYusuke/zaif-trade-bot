# 347# 最低取引単位制約 (1 mBTC) の影響分析

**日付**: 2026-03-09
**HEAD**: `aac283f91`
**種別**: 調査分析
**ステータス**: 完了 (実施内容は Phase 2 以降)

---

## §1 背景

現在のボットは **全残高 ≈ 2 mBTC** で運用されており、Coincheck 板取引の最小注文数量
`min_order_btc: 0.001` (1 mBTC) と実質同額のロットで取引している。

- 実績 lot 分布: **99.3%** が 0.001 BTC (2854 fills 中 2835)
- 0.001 BTC 以外: dust sweep (端数売却) 等のわずかな例外のみ

この制約下では、設計・実装済みの **6 つのリスク管理・ロット制御機構が全て dead code 化**
しており、スケーリング戦略が根本的に制限されている。

---

## §2 Coincheck の注文精度

コード調査の結果:

- `adapter.py` L301: `order_data["amount"] = str(quantity)` — **丸め処理なし**
- `balance_checker.py` L255: `self._current_lot = round(btc_free, 8)` — **8桁精度** (satoshi 単位)
- Coincheck API は BTC 数量を文字列で受け取り、内部で小数点精度を処理

**結論**: Coincheck は **satoshi (0.00000001 BTC) 精度**での注文が可能。
`min_order_btc: 0.001` は API の最小注文量であり、刻み幅 (precision) の制約ではない。

→ `lot_step: 0.001` を `0.0001` 以下に変更すれば、fine-grained lot control が可能になる。

---

## §3 制約下で死蔵している機構一覧

### §3.1 confidence_lot (AS 確率連動ロット縮小) — 完全 no-op

YAML 明記: `152# order_quantity == min_order_btc が 99.7% → 完全 no-op のため有効化見送り`

| 項目 | 状態 |
|------|------|
| `enabled` | `false` |
| 設計 | AS 確率が高い局面で lot を `factor × regime_lot` に縮小 |
| 制約 | `factor × 0.001` < 0.001 → `min_lot` にクランプ → **動作不能** |

### §3.2 動的ロットサイジング (lot_sizer) — 離散的すぎて制御不能

| 設定 | 値 | 問題 |
|------|-----|------|
| `lot_step` | 0.001 | 0.001 → 0.002 で **即 2 倍リスク暴露** |
| `max_lot` | 0.005 | 最大 5x スケール (全残高超過) |
| `enabled` | `false` | 粗すぎる刻みでは有効化しても意味がない |

### §3.3 regime_lot_multiplier — 実質固定

`regime_lot_multipliers` は YAML 未設定。仮に設定しても:

- `0.001 × 0.5 = 0.0005` → `min_lot` にクランプ → **縮小方向の制御が無効**
- `0.001 × 2.0 = 0.002` → 効果はあるが 2 段階しかない (1x / 2x)

### §3.4 balance_shrink (残高不足ロット縮小) — 縮小余地なし

`balance_shrink_divisor: 2` → `0.001 / 2 = 0.0005` → `min_lot` 以下 → **shrink 不可能**

代替として `balance_forced_switch` が発動: **全 filled の 19.1%** (545/2854)

### §3.5 Kelly Criterion — 理論値と実運用の乖離

- `kelly_equity_btc: 0.002` → half-Kelly 推奨 lot を算出
- ただし推奨値が 0.0005〜0.0015 の範囲に入っても lot = 0.001 (floor)
- **Kelly による精密リスク管理がフロア制約で無効化**

### §3.6 DailyDrawdown per_side_recovery — 段階的復帰が即復帰

- `per_side_recovery_lot_scale: 0.5` → `0.001 × 0.5 = 0.0005` → `min_lot` クランプ
- **recovery = 通常 lot** → halt 後の段階的リスク低減が機能しない

---

## §4 直近パフォーマンス (スケーリング根拠)

### §4.1 期間別 PnL 推移

| 期間 | Mean PnL (bps) | Median (bps) | Fills/h | 日次 JPY (0.001 BTC) |
|------|----------------|--------------|---------|---------------------|
| 全期間 (559.9h) | -0.2992 | -0.2115 | 5.10 | -38.85 |
| 2026-03 以降 (185h) | -0.2768 | -0.2186 | 3.83 | — |
| 2026-03-02 以降 (161h) | -0.1258 | — | 4.23 | -13.59 |
| **2026-03-07 以降 (41.5h)** | **+0.0223** | **+0.2050** | **6.24** | **+3.56** |

### §4.2 直近 41.5h のスケーリング試算

直近の +0.0223 bps/trade が安定すると仮定した場合 (BTC ≈ 10,650,000 JPY):

| lot (BTC) | 日次 PnL | 月次 PnL | 備考 |
|-----------|----------|----------|------|
| 0.001 | +3.56 JPY | +107 JPY | 現状 |
| 0.002 | +7.11 JPY | +213 JPY | 残高 2x 必要 |
| 0.005 | +17.78 JPY | +533 JPY | 現 max_lot |
| 0.010 | +35.56 JPY | +1,067 JPY | max_lot 引上げ必要 |
| 0.050 | +177.8 JPY | +5,334 JPY | 中期目標候補 |
| 0.100 | +355.6 JPY | +10,668 JPY | market impact 要検討 |

### §4.3 注意事項

- **+0.0223 bps は 41.5h の極くわずかなプラス** — 統計的有意性は未確認
- 信頼区間が 0 をまたぐ可能性は十分にある
- 全期間/3月平均は依然マイナス → **正の期待値の安定的確認が増資の必須前提条件**

---

## §5 付随して発見された課題

### §5.1 balance_forced_switch 19.1% — 在庫管理の歪み

全 filled の 19.1% (545/2854) が残高不足による side 強制切替。

- **影響**: 本来 buy すべき局面で sell を強制 (またはその逆)
- **根本原因**: 残高 ≈ 1 mBTC (片側全額使用で反対 side の原資が 0)
- **緩和策**: 残高増加で自然解消。lot < 全残高の 50% が安全ライン

### §5.2 soft_loss_cap ロット半減 — 最小ロットでは無意味

`safety.soft_loss_cap_ratio: 0.02` → 残高の 2% でロット半減。
しかし `0.001 / 2 = 0.0005` → min_lot クランプで **半減が発動しない**。

### §5.3 confidence_lot / lot_sizing enabled: false — 明示的な「必要残高」未文書化

152# で「order_quantity > min_order_btc (例: 0.003 BTC) で再検討」と注記あり。
→ lot control 復活のために **最低限必要な残高水準を明示化すべき**。

### §5.4 lot_step 0.001 による離散的スケーリング問題

API は satoshi 精度を受け付けるにもかかわらず、lot_step を 0.001 に設定。
0.001 → 0.002 への増量 (100% 増) は、連続的リスク管理の原則に反する。

### §5.5 maker_price.py ハードコード `_MIN_ORDER_BTC: Final[float] = 0.001`

[maker_price.py](../scripts/v460/lib/maker_price.py) L84 で min_order が定数ハードコード。
config.min_order_btc と二重管理になっている。

### §5.6 fill 全量平均 PnL がマイナス

全期間平均が -0.30 bps、3 月でも -0.28 bps。直近のプラス転換が安定するまでは
ロット増量・残高増強は期待値の確認を待つべき。

---

## §6 段階的スケーリングロードマップ

### Phase 0: 現行 (残高 ≈ 2 mBTC)
- **制約**: lot = min_lot → 6 機構 dead code
- **目標**: 168h fill_test で正の期待値を統計的に確認 (p-value < 0.10)
- **KPI**: mean PnL > 0 bps, Sharpe > 0, 勝率 > 50%

### Phase 1: lot 精度改善 (残高変更なし)
- `lot_step: 0.001 → 0.0001`
- `min_order_btc: 0.001 → 0.0005` (Coincheck 許容なら)
- `_MIN_ORDER_BTC` ハードコードを config 参照に統一
- **効果**: balance_shrink, DD recovery が機能開始

### Phase 2: 初回増資 (残高 → ≈ 5 mBTC)
- `order_quantity: 0.001 → 0.002`
- `max_lot: 0.005 → 0.010`
- `confidence_lot.enabled: true`
- `kelly.equity_btc: 0.005`
- **効果**: confidence_lot, Kelly, regime_lot が一斉に有効化
- **前提**: Phase 0 の正期待値確認済み

### Phase 3: 本格スケール (残高 → ≈ 50 mBTC)
- lot = 0.005〜0.010 を baseline に
- market impact 監視 (fill_rate 低下、spread widening 検出)
- 在庫偏重リスクの増大に対する inv_decay_tau パラメータ再調整

### Phase 4: 自律的スケーリング (残高 → ≈ 200+ mBTC)
- lot_sizer.enabled = true (Kelly 天井付き)
- profit reinvestment による kelly_equity_btc 自動更新
- lot = 0.01〜0.05 レンジでの運用

---

## §7 即時実行可能な改善 (ボット再起動時)

| ID | 施策 | 安全性 | 影響 |
|----|------|--------|------|
| L-1 | `lot_step: 0.001 → 0.0001` | ✅ 安全 | lot_sizer 有効化時の粒度改善 |
| L-2 | `maker_price._MIN_ORDER_BTC` → config 参照に統一 | ✅ 安全 | 二重管理解消 |
| L-3 | (Coincheck 側確認後) `min_order_btc: 0.001 → 0.0005` | ⚠ 要検証 | shrink/recovery 有効化 |
| L-4 | conf_lot/lot_sizer の有効化必要残高をコメント明記 | ✅ 安全 | 運用ドキュメント改善 |

---

## §8 関連ドキュメント

| # | 関係 |
|---|------|
| 152 | confidence_lot 検証結果 (no-op 判定の根拠) |
| 264 | Kelly Criterion 実装 |
| 274 | Kelly Criterion ロット天井 |
| 131 | レジーム連動ロット制御 (D1) |
| 128 | Dust sweep 設計 |
| 162 | Inventory skewing |
| 316 | S-7 テール損失分析 (p10 改善) |
| 346 | S-7 分析スクリプト実装 |
