# 458# ph2 実装: マクロ連動 Sell 防御 — 455#/456#/457# 統合実装

> **状態**: 実装完了・テスト済  
> **前提**: [454#](454_ph2_plan_uptrend_sell_loss_countermeasures.md) 設計案 → [455#](455_ph2_rev_454_uptrend_sell_loss_countermeasures_review.md) Codex レビュー → [456#](456_ph2_plan_uptrend_sell_centric_paradigm.md) Sell-Centric パラダイム → [457#](457_ph2_rev_456_sell_centric_paradigm_review.md) 456# レビュー  
> **SHA**: `73c36b86f` (実装), `d0769f283` (SR修正), `459#` (low_vol ソフト化)

---

## §1 レビュー間コンセンサス抽出

### 1.1 三文書の立ち位置

| 文書 | 主張の核 | 推奨施策 |
|------|----------|----------|
| **455#** (Codex) | R3 表現を修正: micro sell 保護は既存 (`skip_sell_trending`, `trending_sell_offset_boost_factor`)。欠如しているのは **macro→sell** の接続 (S2)。信号ホライゾンのミスマッチ (S1) が根本。 | F-lite → B(保守的) → H → A(小) → G/J |
| **456#** (Sell-Centric) | 上昇トレンド = _seller's market_。sell をスキップするのではなく、**プレミアム機会**として活用すべき。 | A: Liquidity Mirage (far ask) / B: Dynamic Inventory Offloading / C: Cross-Venue |
| **457#** (456# Review) | A-lite のみ抽出 (×1.2-1.8)。B は過重、C は時期尚早。 | A-lite → micro-timeout 短縮 → threshold チューニング |

### 1.2 合意ポイント

1. **Hard skip 禁止** — sell を止めるのではなく、offset で有利に約定させる
2. **F-lite (macro→sell offset boost)** — 全文書が最優先で一致
3. **B (slope_threshold 引下げ)** — 455# が保守的引下げを推奨、他も反対なし
4. **H (micro-timeout macro 連動)** — 455# 推奨、456#/457# も sell 回転高速化に賛同
5. **Hysteresis** — 455# §4.1 で明示的に言及。flapping 防止は前提条件

### 1.3 不一致・棄却ポイント

| 提案 | 理由 |
|------|------|
| 456# A (×3-5 multiplier) | 457# 指摘: sell ceiling=0.50 で clamp される。1.25 以上 hard skip → ×3-5 は数値的に自殺行為 |
| 456# B (Layered sells) | 457# 指摘: Coincheck API 制約でレイヤー管理が過重 |
| 456# C (Cross-Venue) | 457# 指摘: bitFlyer 基盤 (439#-449#) はまだ disabled。時期尚早 |
| 454# E (hard sell skip) | 455# 指摘: AS コスト低下効果は確実だが fill 機会損失が大きい |

---

## §2 盲点の拾い上げ

### 2.1 Hysteresis 欠如 (455# §4.1)

**発見**: `MacroRegimeDetector` は raw OLS slope を毎 30 秒バケット更新で即座に trend 判定していた。volatile 市場では 1 バケットの外れ値で UP→NEUTRAL→UP→NEUTRAL と高頻度フリップが発生する。

**影響**: F-lite の offset boost が ON/OFF を繰り返し、注文価格が不安定化。最悪ケースでは boost 適用中の注文キャンセル → 再注文 → boost 解除 の振動。

**対策**: `_apply_hysteresis()` 状態機械を追加 (§3.1)

### 2.2 Buy 側の対称性 (454# では未検討)

**発見**: 454# は sell 側損失に焦点を当てていたが、下降トレンドでの buy 側にも同一の問題が存在する。macro=DOWN 時の buy offset boost は設計対称性として必須。

**対策**: `macro_buy_boost_weak_down` / `macro_buy_boost_strong_down` も同時実装

### 2.3 456# のプレミアム概念の数値的限界

**発見**: 456# は「sell offset を ×1.5-5.0 にしてプレミアム sell」を提案するが、既存パイプラインでは:
- `sell_offset_ceiling_bps = 0.50` (320# で設定)
- `hard_skip` は `offset_ratio > 1.25` で発動
- post-ceiling multiplier は 421# で clamp 済

→ 実効的な boost 範囲は **×1.0 - ×1.6 程度** が上限。それ以上は ceiling で丸められるか hard_skip で棄却される。

**対策**: A-lite (×1.3 / ×1.6) は ceiling 内に収まる安全な範囲

### 2.4 micro-timeout の逆効果リスク

**発見**: H (micro-timeout 短縮) は sell 回転を高速化するが、AS 市場では「待たない sell = 不利な sell」になりうる。STRONG_UP で timeout を 6 秒にすると、価格が急伸中に早期キャンセル → 再発注を繰り返す可能性がある。

**対策**: timeout は `None` をデフォルトとし、YAML で opt-in 設定。初期値は保守的 (WEAK_UP=12s, STRONG_UP=6s だが live 検証で調整)

---

## §3 実装詳細

### 3.1 MacroRegimeDetector Hysteresis

**ファイル**: `scripts/v460/lib/macro_regime.py`

```
状態機械:
  _confirmed_trend: 確定済みトレンド (外部返却値)
  _pending_trend: 候補トレンド
  _pending_count: 候補の連続回数
  _hold_remaining: 確定後の最低保持回数

遷移ルール:
  1. hold_remaining > 0 → 確定値を維持、hold_remaining--
  2. raw == _pending_trend → _pending_count++
     - _pending_count >= hysteresis_count → 確定遷移
  3. raw != _pending_trend → _pending_trend = raw, _pending_count = 1
```

**Config 追加**:
- `hysteresis_count: int = 3` — 遷移に必要な連続一致回数
- `hold_count: int = 2` — 確定後の最低保持バケット数

### 3.2 F-lite: Macro→Sell Offset Boost

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py`

**挿入位置**: VG sell supplement の後、alert_mode の前 (offset pipeline §7b)

```
条件: side == "sell" AND _last_macro_trend in {WEAK_UP, STRONG_UP}
  → offset *= macro_sell_boost_weak_up (1.3) or macro_sell_boost_strong_up (1.6)

条件: side == "buy" AND _last_macro_trend in {WEAK_DOWN, STRONG_DOWN}
  → offset *= macro_buy_boost_weak_down (1.3) or macro_buy_boost_strong_down (1.6)
```

既存の `_apply_offset_multiplier()` を再利用し、一貫した乗算・ログ出力を確保。

### 3.3 B: Slope Threshold 引下げ

**ファイル**: `configs/v460/fill_test.yaml`

| パラメータ | 変更前 | 変更後 | 根拠 |
|-----------|--------|--------|------|
| `slope_threshold` | 1.0 | 0.5 | 455# 推奨: macro_trend=None が 70% → 検出感度向上 |
| `strong_threshold` | 3.0 | 2.0 | 比例的引下げ |

### 3.4 H: Micro-timeout Macro 連動

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py`

**挿入位置**: micro-timeout wait ループ内 (`_mt_wait` 決定後)

```
条件: side == "sell" AND _last_macro_trend == STRONG_UP
  → _mt_wait = min(_mt_wait, macro_sell_timeout_strong_up)  # 6.0s

条件: side == "sell" AND _last_macro_trend == WEAK_UP
  → _mt_wait = min(_mt_wait, macro_sell_timeout_weak_up)  # 12.0s
```

### 3.5 観測: FillRecord macro_boost_applied

**ファイル**: `ztb/metrics/fill_quality.py`, `scripts/v460/lib/fill_record_builder.py`

- `FillRecord.macro_boost_applied: bool | None = None` フィールド追加
- `_build_fill_strategy_fields()` に `macro_boost_applied` パラメータ追加
- `_build_fill_record()` に `macro_boost_applied` パラメータ追加

### 3.6 Config 追加フィールド

**ファイル**: `scripts/v460/lib/fill_config.py`, `scripts/v460/lib/fill_config_parser.py`

| フィールド | 型 | デフォルト | 説明 |
|-----------|---|-----------|------|
| `macro_sell_boost_weak_up` | float | 1.3 | WEAK_UP 時の sell offset 乗数 |
| `macro_sell_boost_strong_up` | float | 1.6 | STRONG_UP 時の sell offset 乗数 |
| `macro_buy_boost_weak_down` | float | 1.3 | WEAK_DOWN 時の buy offset 乗数 |
| `macro_buy_boost_strong_down` | float | 1.6 | STRONG_DOWN 時の buy offset 乗数 |
| `macro_sell_timeout_weak_up` | float \| None | None | WEAK_UP 時の sell timeout 上書き (秒) |
| `macro_sell_timeout_strong_up` | float \| None | None | STRONG_UP 時の sell timeout 上書き (秒) |

---

## §4 テスト結果

**ファイル**: `tests/unit/v460/test_macro_regime.py`

| テストクラス | テスト数 | 結果 |
|-------------|---------|------|
| §1 Basic Classification | 4 | ✅ PASS |
| §2 Hysteresis | 3 | ✅ PASS |
| §3 compose_regimes | 3 | ✅ PASS |
| §4 Config Defaults | 2 | ✅ PASS |
| §5 FillRecord macro_boost_applied | 1 | ✅ PASS |
| §6 Hot-reload wiring (SR-1) | 1 | ✅ PASS |
| §7 Memory leak prevention (SR-2) | 1 | ✅ PASS |
| **合計** | **15** | **✅ ALL PASS** |

---

## §5 変更ファイル一覧

| ファイル | 変更種別 |
|---------|----------|
| `scripts/v460/lib/macro_regime.py` | MODIFIED — hysteresis 追加 + `_current_bucket_prices` 200件キャップ (SR-2) |
| `scripts/v460/lib/fill_cycle_executor.py` | MODIFIED — F-lite boost + H timeout + macro state 永続化 |
| `scripts/v460/lib/fill_config.py` | MODIFIED — 6 フィールド追加 |
| `scripts/v460/lib/fill_config_parser.py` | MODIFIED — YAML マッピング 6 件追加 |
| `scripts/v460/lib/fill_record_builder.py` | MODIFIED — macro_boost_applied パラメータ追加 |
| `ztb/metrics/fill_quality.py` | MODIFIED — FillRecord フィールド追加 |
| `configs/v460/fill_test.yaml` | MODIFIED — threshold 変更 + boost/timeout 設定追加 |
| `scripts/v460/lib/config_hot_reload.py` | MODIFIED — 6 フィールド hot-reload 配線追加 (SR-1) |
| `tests/unit/v460/test_macro_regime.py` | NEW — 15 テスト |

---

## §6 期待効果と検証計画

### 6.1 期待効果

| 施策 | 期待 |
|------|------|
| slope_threshold 0.5 | macro_trend=None の割合を 70% → 40-50% に削減。UP/DOWN 検出増加 |
| Hysteresis (count=3, hold=2) | macro_trend フリップを 50% 以上削減。安定した boost 適用 |
| F-lite sell boost (×1.3/×1.6) | UP 時の sell offset 増加 → sell AS 率低下 (38% → 目標 30% 以下) |
| H sell timeout (12s/6s) | UP 時の sell reprice 高速化 → sell 滞留時間短縮 |

### 6.2 検証方法

1. **same-SHA 24h 再実走** — 458# コミット後の SHA 固定で fill_test 実行
2. **macro_trend 分布確認** — ログから macro_trend の出現頻度を集計、None 割合を検証
3. **FillRecord macro_boost_applied 集計** — boost 適用回数と適用時の PnL 比較
4. **sell AS 率** — 458# SHA のみでフィルタした sell AS 率が 38% → 30% 以下を目標
5. **sell pnl120** — -0.11 bps → 0.0 bps 以上を目標

---

## §7 セルフレビュー (SR)

| # | 問題 | 重大度 | 対策 |
|---|------|--------|------|
| **SR-1** | 6 新フィールドが `_HOT_RELOADABLE_FIELDS` 未登録 — YAML 変更しても live 反映されない | HIGH | `config_hot_reload.py` に 6 フィールド追加 (`d0769f283`) |
| **SR-2** | `_current_bucket_prices` にバウンドなし — バケット未確定時に無制限成長の理論的リスク | MED | `macro_regime.py` に 200 件キャップ追加 (`d0769f283`) |

**問題なし確認済**:
- `_buckets` は `max_buckets=60` で既にキャップ済
- ヒステリシス状態はスカラー 4 変数のみ（蓄積なし）
- `_last_macro_trend` は単一 `str | None`（蓄積なし）
- `from ... import MacroTrend` はループ内だが Python がモジュールキャッシュするため O(1)

---

## §8 459# ranging_buy_low_vol ソフト化

### 8.1 背景

458# デプロイ後 (3/17 04:37 SHA `d0769f283`)、市場ボラティリティが急落し `ranging_low_vol_skip` ゲートが 08:39 以降 **7h+ 連続ブロック**。BTC 残高ゼロのため buy 不可 → sell も不可のデッドロック状態に陥った。

458# の macro boost は一度も発火しなかった (macro_trend が WEAK_UP/STRONG_UP に到達せず)。

### 8.2 データ根拠

3/16 の低ボラ fill 実績:

| 区分 | n | pnl30 | AS率 |
|------|---|-------|------|
| low vol (ratio<0.75) | 12 | **+3.06 bps** | 33% |
| high vol (ratio>=0.75) | 71 | +0.49 bps | 28% |

低ボラ fills は `low_vol_offset_boost` (1.4x) の保護下で高ボラより良好な PnL。hard skip による機会損失の方が大きいと判断。

### 8.3 変更内容

| 設定 | 変更前 | 変更後 | 理由 |
|------|--------|--------|------|
| `ranging_buy_low_vol_as_offset` | `false` | **`true`** | hard skip → offset boost に委譲 |
| `low_vol_offset_boost` | 1.4 | **1.5** | ソフト化に伴う保護強化 |

hot-reload 対応済み (再起動不要)。

---

## §9 残課題

| ID | 内容 | 優先度 |
|----|------|--------|
| R-1 | live 検証後の boost 乗数チューニング (ceiling 到達頻度で判断) | P1 |
| R-2 | STRONG_UP timeout 6s が AS 市場で逆効果でないか live 検証 | P1 |
| R-3 | macro-micro 連携の深化 (micro regime が macro を override するケース) | P2 |
| R-4 | hysteresis パラメータの最適化 (count=3, hold=2 は理論値、live で調整) | P2 |
