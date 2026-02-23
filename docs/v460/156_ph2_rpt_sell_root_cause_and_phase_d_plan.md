# 156# Sell 根本原因分析 + 168h 試験総括 + Phase C/D 並行計画

> 155# §9 残レビュー対応、sell 弱体の構造的根本原因の貪欲的調査、
> 168 時間試験の全体フロー再確認、Phase D 並行開始計画を統合する。

---

## §1 概要

| 項目 | 値 |
|------|-----|
| 前提文書 | 000# (命名規則), 134# (ロードマップ), 144# (§7-§10 レビュー), 155# (§9-§11 分析) |
| 対象期間 | 2026-02-13 〜 2026-02-23 (Phase C dry-run 10日間) |
| 分析対象 | sell 弱体の 7 重ゲート問題、168h 試験フロー、Phase C/D 並行可否 |
| 作成日 | 2026-02-23 |

---

## §2 168h 試験全体フロー再確認

### 2.1 タイムライン

| 日付 (UTC) | イベント | 文書 | 備考 |
|------------|---------|------|------|
| 02/13 | Phase C dry-run 開始 | 147# | PID 初期 |
| 02/13-02/17 | 安定稼働期 | - | ~600 records/day |
| 02/18 | 最初の restart | 148# | auto-restart 実装 |
| 02/19 | 10h ログ分析 | 154# | P0-08 deadlock 発見 |
| 02/20 | deadlock 対策デプロイ | 149#-150# | lock timeout + auto-restart |
| 02/21-02/22 | balance_forced 集中発生 (131+183件) | 155# §9.2 | 実質停止時間 |
| 02/22 | 155# ヒンドサイト分析 | 155# | 2,407 records |
| 02/22-02/23 | §9 レビュー対応 + §11 残課題 | 155# | trending sell抑制等 |
| 02/23 04:36 | PID 108148 deadlock (最終) | - | 現在もハング中 |

**累計**: 28 restarts, 2,407 records, ~240h elapsed (168h 超)

### 2.2 蓄積データの品質評価

| 指標 | 値 | G1.2-full 閾値 | 判定 |
|------|-----|---------------|------|
| レコード数 | 2,407 | ≥500 (attempted) | ✅ 十分 |
| カレンダーカバレッジ | 10 暦日 | ≥7 暦日 | ✅ 超過 |
| 有効分析レコード (price>0) | 1,948 | - | 81% (19% が price=0) |
| restart 回数 | 28 | - | 多い (安定性に課題) |
| balance_forced_skip | 314 (13.0%) | - | 過大 → 実質停止時間 |
| orderbook_error | 156 (6.5%) | - | 改善済 (§11 fallback) |

### 2.3 168h Gate 判定に向けた現状

Phase C の目的は G1.1-exec (168h Qualification Gate) の判定データ蓄積。  
168h は暦日で超えているが、**deadlock + restart で実稼働時間は不完全**。  
→ 正式な Gate 判定は Phase C restart 後のクリーンデータで行う。  
→ ただし、2,407 records での **暫定的分析は十分に価値がある**。

---

## §3 Sell 根本原因分析 — 7 重ゲートと負の螺旋

### 3.1 問題定量化

| 指標 | Buy | Sell | 差分 |
|------|-----|------|------|
| avg PnL 30s (bps) | -0.081 | **-0.516** | -0.435 |
| avg PnL 120s (bps) | +0.136 | **-0.470** | -0.606 |
| 逆行率 (逆 side が良い率) | 52.6% | **58.5%** | +5.9pt |
| 総損失寄与 | 62% | **138%** | +76pt |
| trending PnL | +0.594 | **-0.687** | -1.281 |
| VG ON PnL | - | **-0.728** | vs VG OFF -0.412 |

**結論**: sell は buy の 5-6 倍悪い。損失の 138% を sell が生み出している。

### 3.2 Sell を殺す 7 重ゲート (構造図)

sell 注文が発注に到達するまでのフィルタチェーン:

```
注文サイクル開始
  │
  ├─ Gate 1: Time Filter (skip_utc_hours_sell=[4,8,14])
  │   → 3h/24h = 12.5% ブロック
  │
  ├─ Gate 2: skip_sell_unknown_regime=true
  │   → unknown regime で sell 全殺し (buy は offset boost 止まり)
  │
  ├─ Gate 3: skip_sell_trending=true
  │   → trending regime で sell 全殺し (buy にはなし)
  │
  ├─ Gate 4: sell_dynamic_kill (rolling 50-fill < -0.5bps)
  │   → 発動で 20 cycle = 40分間 sell 凍結 (buy にはなし)
  │
  ├─ Gate 5: SkipGate ML (target_skip_rate_sell=0.20)
  │   → buy (0.15) より 33% 高い skip 率目標
  │   → sell は pnl120 モデル (buy は pnl30) → 時間軸不利
  │
  ├─ Gate 6: sell_guard (max_spread=4000JPY, offset_floor=0.10)
  │   → 高 spread で sell 全面拒否 (buy にはなし)
  │
  └─ Gate 7: MakerPrice (offset sell=0.18 vs buy=0.05)
      → sell は bid/ask から 3.6 倍遠い位置に配置
      → fill_rate 構造的低下
      → VG/trending boost でさらに offset 拡大 (0.18×1.5=0.27)
      → offset_floor 二重適用で下限が固定

最終: sell がフィルタを全通過しても、約定位置が不利 → PnL 悪化
```

**Buy にはゲート 2, 3, 4, 6 に相当するものがない。**

### 3.3 根本原因 5 層分解

#### 層 1: 市場構造的逆選択 (Adverse Selection)

- BTC/JPY 市場参加者の大半は **買い手 (=maker sell の相手方)**
- sell (ask に並ぶ) は「情報を持った買い手」にピッキングされやすい
- これは maker market making の既知の構造問題で、**sell offset を上げて対応するのは正しい**
- ただし offset 0.18 は **過剰防御** の可能性

**証拠**: sell fast fill avg -1.66 bps (090#) — 即座に成る sell は逆選択が激しい

#### 層 2: 上昇トレンドバイアス

- 分析期間 (02/13-02/23) は BTC/JPY が概ね上昇トレンド
- trending regime で sell = 必然的に逆行
- **しかし**: ranging でも sell avg -0.412 bps → **トレンドだけでは説明できない**

#### 層 3: 防御策積層による負の螺旋 (本文書の核心)

```
sell PnL 悪い
  ↓
防御策を追加 (skip_sell_trending, sell_dynamic_kill, offset 引き上げ)
  ↓
sell fill 数が激減 (7重ゲートの通過率低下)
  ↓
ML モデルの sell 訓練データ不足
  ↓
SkipGate sell モデル精度悪化 (「良い sell 機会」も skip)
  ↓
残った sell は偏ったサンプル → sell PnL さらに悪化
  ↓  
さらに防御策強化 ...
```

**この螺旋が sell 弱体の最大の構造的問題**。防御策の個別は合理的だが、積層効果が機会損失の増大を招いている。

#### 層 4: パラメータ非対称の過剰

sell 固有パラメータ数: **18+ 個** (§3.4 に全列挙)  
buy 固有パラメータ数: **5 個** (offset, skip_hours, fast_fill_threshold, stale_check, skip_buy_unknown_regime)

sell は buy の **3.6 倍** のパラメータで制御されている。これは:
- パラメータ間の相互作用が予測困難
- 個別最適化が全体最適から乖離
- デバッグ・原因特定が困難

#### 層 5: OB 特徴量欠損による ML 精度劣化

144# §9 #3 (CRITICAL): SkipGate の OB 取得で `.price/.quantity` 前提だが、実 adapter は tuple list。  
→ try/except で握り潰し → OB 特徴量が実質無効化  
→ ML モデルが板情報なしで判定 → **sell 判定精度が構造的に低い**

この問題は buy/sell 共通だが、sell は ML 依存度が高い (pnl120 モデル) ため影響が大きい。

### 3.4 Sell 固有パラメータ全量 (18 個)

| # | パラメータ | 現在値 | Buy 対応 | 非対称度 |
|---|-----------|--------|---------|---------|
| 1 | `side_offset.sell` | 0.18 | 0.05 (共通) | **3.6x** |
| 2 | `order_timeout_sec_sell` | 75s | 90s | 0.83x |
| 3 | `skip_utc_hours_sell` | [4,8,14] | [8,16,18] | 異なる時間帯 |
| 4 | `fast_fill_threshold_sec_sell` | 15s | 10s | 1.5x |
| 5 | `fast_fill_offset_boost_sell` | 2.5 | 2.0 (共通) | 1.25x |
| 6 | `stale_check_after_sec_sell` | 30s | 15s | **2.0x** |
| 7 | `stale_drift_bps_sell` | 5.0bps | 4.0bps | 1.25x |
| 8 | `stale_max_reprice_sell` | 1 | 1 | 同等 |
| 9 | `narrow_spread_boost_sell` | 2.0 | 1.5 | 1.33x |
| 10 | `skip_gate_target_skip_rate_sell` | 0.20 | 0.15 | 1.33x |
| 11 | `skip_gate_as_threshold_sell` | 0.50 | 0.545 (共通) | 0.92x (厳格) |
| 12 | `skip_gate_model_path_sell` | pnl120_sell | pnl30_buy | 異なるモデル |
| 13 | `sell_guard.max_spread_jpy` | 4000 | なし | **sell のみ** |
| 14 | `sell_guard.offset_floor` | 0.10 | なし | **sell のみ** |
| 15 | `skip_sell_unknown_regime` | true | false相当 | **sell のみ全殺し** |
| 16 | `skip_sell_trending` | true | なし | **sell のみ全殺し** |
| 17 | `sell_dynamic_kill.enabled` | true | なし | **sell のみ** |
| 18 | `sell_dynamic_kill.threshold_bps` | -0.5 | なし | **sell のみ** |

### 3.5 定量的影響推定

| ゲート | 推定 sell ブロック率 | sell PnL への寄与 |
|--------|-------------------|------------------|
| Time Filter | ~12.5% | 中 (悪い時間帯を正しく回避) |
| unknown regime skip | ~15-20% | **高** (良い sell も殺す) |
| trending skip | ~10-15% | **最高** (trending sell = -0.687 だが、trending down sell は +?) |
| sell_dynamic_kill | ~5-10% (発動時 40分凍結) | 中 (発動頻度は不明) |
| SkipGate ML | 20% | **高** (OB 欠損で精度劣化) |
| sell_guard | ~5% | 低 (極端な spread のみ) |
| offset 3.6x | fill_rate -15-20pt | **最高** (約定自体を阻害) |

**累積推定**: sell の実発注到達率は buy の **40-50%** 程度。  
これは「sell を防御している」のではなく「sell の機会を半減させている」。

---

## §4 Sell 改善: 脱・負の螺旋 戦略

### 4.1 原則: 「防御削減 + 精度向上」の同時実行

防御を単に削減するだけでは PnL 悪化。ML 精度を同時に上げることで、  
「良い sell を通し、悪い sell を止める」精度を高める。

### 4.2 Phase D で即実行すべき sell 改善

| 優先 | 施策 | 期待効果 | 工数 | 根拠 |
|------|------|---------|------|------|
| **P0** | OB 特徴量正規化 (144# §9 #3) | SkipGate ML 精度回復 → sell 判定改善 | 0.3日 | OB tuple/object 不整合で全特徴量欠損 |
| **P0** | SkipGate sell 再訓練 (2,407 records) | sell モデル精度向上 | 0.5日 | 旧データで訓練済みモデルが実環境に適合していない |
| **P1** | `side_offset.sell` 段階的縮小 (0.18→0.14) | fill_rate +10-15pt | 0.1日 | 0.18 は過剰防御。0.14 でも offset_floor 0.10 が下限を保証 |
| **P1** | `skip_sell_trending` を方向別に分解 | trending_down sell を復活 | 0.3日 | trending_down での sell は自然な方向。一律全殺しは過剰 |
| **P2** | `sell_dynamic_kill` の cooldown 短縮 (20→10 cycle) | 凍結時間半減 | 0.1日 | 40分凍結は過剰。20分で十分 |
| **P2** | `stale_check_after_sec_sell` 30s→20s | sell reprice 高速化 | 0.1日 | buy (15s) との差を縮める |
| **P3** | sell pnl120→pnl30 モデル統一評価 | 時間軸公平化 | 0.5日 | 120s horizon は短期リバージョンを見逃す |

### 4.3 A/B テスト設計

Phase D で以下を順次検証:

1. **A/B-1: OB 正規化 ON/OFF** → SkipGate の AS_probability 分布変化を測定
2. **A/B-2: sell offset 0.18 vs 0.14** → fill_rate と PnL30 の比較 (各 24h)
3. **A/B-3: skip_sell_trending → skip_sell_trending_up のみ** → trending_down sell を開放
4. **A/B-4: balance_forced 救済モード** → §9.5 #1 (低リスク執行)

---

## §5 Phase C/D 並行計画

### 5.1 並行の正当性

| 論点 | 判断 |
|------|------|
| Phase C データ量 | 2,407 records、G1.2 F8 (≥500) を超過 → retrainに十分 |
| Phase C 稼働状態 | PID 108148 deadlock → restart 必要、restart 待ちの間に Phase D 作業可能 |
| Phase D 準備状態 | 136# retrain 基盤は ready、SkipGate 訓練パイプラインも稼働可能 |
| リスク | Phase C パラメータと Phase D 変更が干渉する可能性 → A/B テスト的に段階実施で回避 |

### 5.2 Phase C 残作業

| # | タスク | 優先 |
|---|--------|------|
| C-1 | PID 108148 restart + 安定稼働確認 | P0 |
| C-2 | 155# パラメータ変更 (trending sell 抑制等) の効果測定 | P0 |
| C-3 | G1.2-full Gate 暫定判定 (2,407 records ベース) | P1 |
| C-4 | balance_forced 集中発生の根因調査 (02/21-22: 314件) | P1 |

### 5.3 Phase D 作業計画

| # | タスク | 優先 | 前提 |
|---|--------|------|------|
| D-1 | OB tuple/object 正規化 utility (144# §9 #3) | **P0** | なし |
| D-2 | SkipGate sell 再訓練 (2,407 records) | **P0** | D-1 完了 |
| D-3 | sell offset 段階的縮小 A/B | P1 | Phase C restart 後 |
| D-4 | skip_sell_trending 方向別分解 | P1 | D-2 効果測定後 |
| D-5 | balance_forced 救済モード (155# §9.5 #1) | P2 | D-3/D-4 結果次第 |
| D-6 | 144# §8 lot 分離 (CRITICAL #1/#2/#3) | P1 | 別タスク |

### 5.4 並行実行ガントチャート

```
Week 1 (02/24-03/02):
  Phase C: [C-1 restart] [C-2 効果測定]────────────
  Phase D: [D-1 OB正規化] [D-2 SkipGate再訓練]───

Week 2 (03/02-03/09):
  Phase C: ──────[C-3 Gate暫定判定]────[C-4 bf調査]
  Phase D: [D-3 offset A/B] [D-4 trending分解]────

Week 3 (03/09-03/16):
  Phase C: ──[Gate最終判定]──→ Phase C 完了 or 延長
  Phase D: [D-5 bf救済]──[D-6 lot分離]────────────
```

---

## §6 155# §9 残レビュー対応

### 6.1 対応済み項目 (前セッション)

| 項目 | 対応 | コミット |
|------|------|---------|
| §9.4 #1: price=0 補間 | ✅ hindsight_filter.py 修正 | `ac272326f` |
| §9.4 #2: balance_forced_consecutive | ✅ FillRecord 拡張 | `65ee87f98` |
| §9.5 #3: orderbook_error fallback | ✅ _prev_mid_price | `65ee87f98` |
| §9.3 P0 #2: trending sell 抑制 | ✅ skip_sell_trending=true | `ac272326f` |
| §9.2 #3: wait band 分析 | ✅ _analyze_wait_bands() | `ac272326f` |
| §9.2 #4: regime×side 分析 | ✅ _analyze_regime_side() | `ac272326f` |

### 6.2 今セッション対応

| 項目 | 対応 | 備考 |
|------|------|------|
| §9.4 #3: reprice ログ連携 | **Phase D D-1 に委譲** | OB 正規化と合わせて実装が効率的 |
| §9.5 #1: balance_forced A/B | **Phase D D-5 に委譲** | restart 後のクリーンデータで検証 |
| §9.3 P0 #1: balance_forced 救済 | **Phase D D-5 に委譲** | 同上 |

### 6.3 155# §11.4 残課題の disposition

| 項目 | 分類先 | 理由 |
|------|--------|------|
| reprice ログ連携 | Phase D D-1 | OB 正規化と同一作業スコープ |
| balance_forced A/B | Phase D D-5 | Phase C restart 後に clean データ必要 |
| 118# §8.5 Oracle テスト | ph3-pre | 大規模作業、ph2 スコープ外 |
| 118# §6.3 運用失敗モードテスト | ph3-pre | 障害注入テスト設計が必要 |

---

## §7 144# §8-§10 未対応レビュー項目の現況

144# の Codex レビューには重大な未修正項目が残っている。

### 7.1 最優先 (収益直結)

| 項目 | 重大度 | 概要 | 対応計画 |
|------|--------|------|---------|
| §8 #1 | CRITICAL | preflight と lot 適用順の不一致 | Phase D D-6 |
| §8 #2 | HIGH | _current_lot 乗算的増加 | Phase D D-6 (lot 分離で同時解決) |
| §8 #3 | HIGH | 片側更新による preflight 過大化 | Phase D D-6 |
| §9 #3 | HIGH | SkipGate OB tuple/object 不整合 | **Phase D D-1 (最優先)** |

### 7.2 中優先

| 項目 | 重大度 | 概要 | 対応計画 |
|------|--------|------|---------|
| §9 #1 | CRITICAL | reprice 時数量不整合 | Phase D D-6 (lot 分離で解決) |
| §9 #2 | HIGH | timeout 判定の FillRecord 基準ズレ | Phase D (minor fix) |
| §9 #4-#7 | MEDIUM-LOW | SkipGate lot 整合、FillRecord 重複等 | Phase D D-6 + リファクタ |

### 7.3 144# §9 D1-D4 リファクタ案の評価

| リファクタ | 判定 | 理由 |
|-----------|------|------|
| D1: RecordFactory | Phase D 実施推奨 | FillRecord 生成重複は保守コスト増大中 |
| D2: OrderIntent | Phase D 実施推奨 | lot/timeout のズレを構造的に防止 |
| D3: CancelReason Enum | ✅ 一部完了 (145#) | cancel_reasons.py に定数集約済み |
| D4: Top-of-book 正規化 | **Phase D D-1 で実施** | sell 改善に直結 |

---

## §8 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-02-23 | 初版: sell 根本原因 7 重ゲート分析 + 168h 総括 + Phase C/D 並行計画 |

---

## §9 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `docs/v460/144_ph2_impl_regime_reprice_timeout.md` | §7 ロードマップ更新 (145#-155# 追加、Phase C/D 状況) |
| `docs/v460/155_ph2_rpt_hindsight_filter_analysis.md` | 命名規則準拠リネーム (旧: 155_hindsight_filter_analysis.md) |
| `docs/v460/132_ph2_rpt_fill_test_log_analysis.md` | 命名規則準拠リネーム (旧: 132_fill_test_log_analysis.md) |
| `docs/v460/index.md` | 132# リンク更新 + 155# エントリ追加 |
| `docs/v460/133_ph2_rev_132_profitability_max_plan.md` | 132# リネームに伴うリンク更新 |
| `archived/docs_v460/146_multi_exchange_registry.md` | 146# 重複ファイルのアーカイブ |
| `docs/v460/156_ph2_rpt_sell_root_cause_and_phase_d_plan.md` | 本ドキュメント (新規) |
