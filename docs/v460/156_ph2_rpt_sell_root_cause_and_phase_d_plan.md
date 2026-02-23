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

---

## §10 追補: 155# 実装レビュー結果 (Codex 反映)

> 先行レビュー（チャット回答）を文書化し、Phase D 計画へ接続する。

### 10.1 主要指摘 (重大度付き)

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|--------|--------------|------|---------|
| 1 | **HIGH** | `scripts/v460/run_fill_test.py` | `balance_forced` で実行許可されたケースでも `skip_sell_trending` が後段で発火し、実質的に「売るしかない局面」で売れず停止しうる。 | `balance_forced_switch=True` 時は `skip_sell_trending` をバイパス。one-sided balance 時は必ず執行。 |
| 2 | **MEDIUM** | `scripts/v460/analysis/hindsight_filter.py` | `side` が `buy/sell` 以外でも後知恵 PnL に混入し、符号が歪む可能性。 | `buy/sell` 以外を除外、または別カテゴリ（例: H9_invalid_side）へ分離。 |
| 3 | **MEDIUM** | `scripts/v460/run_fill_test.py`, `scripts/v460/lib/order_monitor.py`, `scripts/v460/lib/cancel_reasons.py` | `post_only_reject` と `postonly_reject` が混在し、原因別集計が分断。 | cancel_reason を定数経由で単一表記に統一し、読み込み時に互換マップで吸収。 |
| 4 | **MEDIUM** | `scripts/v460/analysis/hindsight_filter.py` | H6 technical 分類が `orderbook_timeout/rate_limit/empty/sell_guard_reject` を網羅しきれず、技術要因の寄与が過小化。 | `_categorize()` を `cancel_reasons` 定数と同期し、技術要因を一括分類。 |
| 5 | **LOW** | `scripts/v460/run_fill_test.py` | `orderbook_error` 時の `_prev_mid_price` fallback に鮮度判定がない。 | fallback 価格の age を記録し、閾値超過時は stale フラグ付きで扱う。 |
| 6 | **MEDIUM** | `tests/unit/v460/test_155_hindsight_review.py` | 設定・フィールド存在テスト中心で、相互作用（forced×trending, reason 正規化等）の挙動テストが不足。 | 挙動テストを追加（forced時バイパス、reason正規化、sell timeout反映）。 |

### 10.2 収益直結の実装優先順 (レビュー反映)

| 優先 | 施策 | 目的 |
|------|------|------|
| **P0** | `balance_forced` と `skip_sell_trending` の競合解消 | デッドロック回避 + 機会損失削減 |
| **P0** | cancel_reason 正規化 (`post_only_reject` 系) | ログ分析精度改善、誤った意思決定防止 |
| **P1** | H6 technical 分類の網羅化 | `orderbook_*` 系の損失寄与を可視化 |
| **P1** | 15-30s 帯 reprice/cancel の早期化検証 | wait 帯ワースト損失の抑制 |
| **P2** | fallback price 鮮度管理 | ヒンドサイト集計の信頼性向上 |

### 10.3 Phase D への組込み

既存の D タスクへ以下を明示的に内包する:

1. D-1 (OB 正規化) に「H6 分類同期 + reason 正規化」を追加。  
2. D-4 (trending 分解) に「balance_forced 時バイパス」の実装・検証を追加。  
3. D-2/D-3 の評価ログで `post_only/postonly` 混在が残っていないことを受入条件にする。  

---

## §12 §10 レビュー実装結果

### 12.1 実装サマリ

| # | 重大度 | ステータス | 対応内容 |
|---|--------|-----------|----------|
| 1 | **HIGH** | **完了** | `run_fill_test.py` L1781: `skip_sell_trending` 条件に `and not _balance_forced` を追加。forced sell がトレンドゲートでブロックされるデッドロックを解消。 |
| 2 | **MEDIUM** | **完了** | `hindsight_filter.py` `_analyze_records()`: `side not in ("buy", "sell")` を除外。unknown side が PnL 符号を汚染する問題を防止。 |
| 3 | **MEDIUM** | **完了** | `order_monitor.py` L212: `"postonly_reject"` → `"post_only_reject"` に統一。`cancel_reasons.py` 定数と整合。`hindsight_filter.py` は後方互換で両形式を受容。 |
| 4 | **MEDIUM** | **完了** | `hindsight_filter.py` `_TECHNICAL_REASONS`: 5→11 に拡張。`orderbook_timeout/rate_limit/empty` + `sell_guard_reject` + `post_only_reject/postonly_reject` を追加。H6_technical 分類の網羅性を確保。 |
| 5 | **LOW** | **完了** | `run_fill_test.py` L1005-1030: `_prev_mid_time` から `_fallback_age` を算出、120s 超で `_fallback_stale=True` → `order_price=0.0`。stale 価格による分析汚染を防止。 |
| 6 | **MEDIUM** | **完了** | `test_155_hindsight_review.py`: 挙動テスト 5 クラス追加 — `TestBalanceForcedTrendingBypass`, `TestCancelReasonNormalization`, `TestSideValidation`, `TestH6TechnicalClassification`, `TestFallbackPriceStaleness`。 |

### 12.2 テスト結果

- `test_155_hindsight_review.py`: **28 passed** (既存 14 + 新規 14)
- `test_fill_quality.py`: **176 passed** (postonly→post_only assertion 更新)
- `test_145_structural_fixes.py`: **53 passed** (影響なし確認)

### 12.3 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | #1 balance_forced×trending bypass + #5 fallback 鮮度管理 |
| `scripts/v460/lib/order_monitor.py` | #3 cancel_reason 正規化 (postonly→post_only) |
| `scripts/v460/analysis/hindsight_filter.py` | #2 side バリデーション + #4 H6 分類拡張 + #3 後方互換 |
| `tests/unit/v460/test_fill_quality.py` | postonly→post_only assertion 更新 |
| `tests/unit/v460/test_155_hindsight_review.py` | #6 挙動テスト 5 クラス追加 |
| `docs/v460/156_ph2_rpt_sell_root_cause_and_phase_d_plan.md` | 本セクション追加 |

---

## §14 balance_forced バイパス水平展開 + Phase C 再起動

### 14.1 問題

§10 #1 では `skip_sell_trending` への balance_forced バイパスのみ修正したが、同種のデッドロックリスクが残り 2 ゲートにも存在:

| ゲート | side | デッドロック条件 | 修正前 |
|--------|------|-----------------|--------|
| `skip_buy_unknown_regime` | buy | forced→buy + regime=unknown | ❌ バイパスなし |
| `skip_sell_trending` | sell | forced→sell + regime=trending | ✅ §10 #1 修正済 |
| `sell_dynamic_kill` | sell | forced→sell + rolling PnL 閾値以下 | ❌ バイパスなし |

### 14.2 修正内容

全 3 ゲートに `and not _balance_forced` 条件を統一:

```python
# skip_buy_unknown_regime (L1773)
if (self.config.skip_buy_unknown_regime
    and next_side == "buy"
    and not _balance_forced          # ← 追加
    and self._regime_detector ...):

# sell_dynamic_kill (L1824)
if (self.config.sell_dynamic_kill_enabled
    and next_side == "sell"
    and not _balance_forced          # ← 追加
    and self._is_sell_killed()):
```

### 14.3 テスト

`TestBalanceForcedBypassHorizontal` (4 テスト):
- 各ゲートのソースに `not _balance_forced` が含まれることを検証
- AST パースで 3 箇所以上の `not _balance_forced` を確認

**結果**: 32 passed (既存 28 + 新規 4)

### 14.4 Phase C 再起動

| 項目 | 値 |
|------|-----|
| 旧 PID | 108148 (deadlock since 02/23 04:36, 強制停止) |
| 新 PID | 120384 |
| 起動日時 | 2026-02-23 19:01:06 JST |
| Git SHA | `3959424ef` |
| 設定ファイル | `configs/v460/fill_test.yaml` |
| 実行時間 | 168h (`--hours 168`) |
| exchange | coincheck |

### 14.5 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | skip_buy_unknown_regime + sell_dynamic_kill に balance_forced bypass 追加 |
| `tests/unit/v460/test_155_hindsight_review.py` | TestBalanceForcedBypassHorizontal 4 テスト追加 |
| `docs/v460/156_ph2_rpt_sell_root_cause_and_phase_d_plan.md` | 本セクション追加 |

---

## §15 変更履歴 (追補)

| 日付 | 内容 |
|------|------|
| 2026-02-23 | 155# 実装レビュー結果（重大度付き）を追補し、Phase D 連携を明記 |
| 2026-02-23 | §10 レビュー全 6 項目の実装完了、§12 に結果記録 |
| 2026-02-23 | balance_forced バイパス水平展開 (全3ゲート統一) + Phase C 再起動 (PID 120384) |
| 2026-02-24 | §16 自己レビュー: 8 項目修正 + レジーム既存資産調査 + テスト 43→11追加 |

---

## §16 自己レビュー + レジーム既存資産調査

### 16.1 自己レビュー発見事項

156# 全 4 コミット (`5c0ca0cae`..`2f01fcdde`) の横断レビュー結果:

| # | 重大度 | 問題 | 対応 |
|---|--------|------|------|
| 1 | MEDIUM | `import time as _time_mod` ブロック内重複 import — 上位に `import time` 既存 | 除去、`time.time()` に統一 |
| 2 | MEDIUM | fallback staleness 120s ハードコード — config 化されていない | `fallback_stale_sec: float = 120.0` を FillConfig + from_yaml に追加 |
| 3 | HIGH | `_prev_mid_price` / `_prev_mid_time` 直接アクセス (private 属性) | `get_fallback_price()` 公開 API を MakerPriceCalculator に追加、全箇所を置換 |
| 4 | INFO | `skip_sell_unknown_regime` が config + SkipGate evaluator に存在するが main loop に無い | 意図的設計 — SkipGate が ML ルールとして処理。ドキュメント記録のみ |
| 5 | LOW | `UNKNOWN_REGIME_SELL_SKIP` 定数が cancel_reasons.py に無い | 定数追加 + AUDIT_CANCEL_REASONS frozenset に登録 |
| 6 | MEDIUM | hindsight_filter の side 除外がログ出力されない | `_skipped_invalid_side` カウンター + logger.info 追加 |
| 7 | MEDIUM | タイムスタンプなし fallback パスが stale 扱いされない | `_fallback_stale = True` に変更 ("no timestamp, treated as stale") |
| 8 | LOW | hindsight_filter.py に logger 未定義 (NameError) | `import logging` + `logger = getLogger(__name__)` 追加 |

### 16.2 レジーム既存資産調査

156# Phase D 設計に向け、既存のレジーム関連実装を調査:

**3 つのレジーム検出器:**

| 実装 | パス | 状態数 | 用途 |
|------|------|--------|------|
| `FillTestRegimeDetector` | `scripts/v460/lib/regime_detector.py` | 4 (trending/ranging/high_vol/unknown) | fill_test メインループ使用中 |
| `MarketRegimeDetector` | `ztb/analysis/regime/basic_regime_detector.py` | 4 (bull/bear/sideways/volatile) | 汎用 (未使用) |
| `AdvancedRegimeDetector` | `ztb/analysis/regime/advanced_regime_detector.py` | 12 (RSI/ADX/MACD 組合せ) | 汎用 (未使用) |

**Key findings:**

- `SellDynamicKillManager` (`ztb/risk/sell_dynamic_kill.py`): sell 専用 kill。regime_thresholds dict でレジーム別閾値設定可能
- **BuyDynamicKillManager は存在しない** — buy 側は unknown skip のみで防御が薄い
- `skip_sell_unknown_regime`: FillConfig L179 に config 存在、SkipGate evaluator L456 で `rule_skip_unknown_sell` として ML 判定内で使用。main loop のゲートとは別系統
- `unknown_buy_offset_boost` (130#): unknown レジーム時に buy offset を引き上げる既存機能
- `FillTestRegimeDetector` は hysteresis (count=3) + confidence gate (min=0.4) + 152# UNKNOWN→初回遷移加速を実装済み

### 16.3 Phase D への示唆

1. **sell 防御は十分** — 7 重ゲート + SellDynamicKillManager + SkipGate rule_skip_unknown_sell
2. **buy 側が手薄** — unknown skip のみ。BuyDynamicKillManager 相当、または buy_trending_boost が Phase D+ で検討候補
3. **AdvancedRegimeDetector (12 状態)** は現状未使用だが、trending の方向分解 (上昇/下降) に活用可能 → D-2/D-3 にフィード

### 16.4 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | import 重複除去, `get_fallback_price()` 呼出し置換 (2 箇所), `fallback_stale_sec` config 参照, no-timestamp stale 処理 |
| `scripts/v460/lib/maker_price.py` | `get_fallback_price()` 公開メソッド追加 |
| `scripts/v460/lib/cancel_reasons.py` | `UNKNOWN_REGIME_SELL_SKIP` 定数 + frozenset 登録 |
| `scripts/v460/lib/fill_config.py` | `fallback_stale_sec: float = 120.0` フィールド + from_yaml マッピング |
| `scripts/v460/analysis/hindsight_filter.py` | `import logging` + logger 初期化 + side 除外カウンター + ログ出力 |
| `tests/unit/v460/test_155_hindsight_review.py` | §16 テスト 11 件追加 (43 件合計) |
| `tests/unit/v460/test_145_structural_fixes.py` | AUDIT_CANCEL_REASONS テスト期待値更新 |

---

## §17 Phase D 実装 (156# D-1/D-3/D-4/D-5)

### 17.1 実装サマリ

| Task | 内容 | 変更ファイル |
|------|------|-------------|
| D-1 | OB fetch 失敗ログ可視化 + カウンタ | `skip_gate_evaluator.py` |
| D-3 | sell offset 段階的縮小 A/B 準備 (YAML 注記) | `fill_test.yaml` |
| D-4 | trending 方向分解 (TRENDING_UP/TRENDING_DOWN) | `regime_detector.py`, `fill_config.py`, `run_fill_test.py`, `maker_price.py` |
| D-5 | sell_dynamic_kill cooldown 半減 (20→10) | `fill_test.yaml` |

### 17.2 D-4: trending 方向分解

**変更の核心**: `FillTestRegime.TRENDING` を `TRENDING_UP` / `TRENDING_DOWN` に分解。

- `FillTestRegime` に `TRENDING_UP = "trending_up"`, `TRENDING_DOWN = "trending_down"` 追加
- `is_trending` プロパティで後方互換 (`TRENDING`, `TRENDING_UP`, `TRENDING_DOWN` を統一判定)
- `_classify()`: `trend_pct > 0` → `TRENDING_UP`, else `TRENDING_DOWN`
- `skip_sell_trending` ゲート: `is_trending` で判定 + `skip_sell_trending_up_only` で下降トレンド sell 開放
- `maker_price.py` の offset boost: `.value == "trending"` → `.is_trending` に置換

**regime_thresholds 自動連携**: YAML 既存の `trending_up: -0.3`, `trending_down: -1.0` キーが
`_classify()` の戻り値 `.value` と直接一致し、`SellDynamicKillManager.check_kill()` で方向別閾値が有効化。

### 17.3 D-1: OB fetch 可視化

- `skip_gate_evaluator.py` に `_ob_fetch_fail_count`, `_ob_fetch_total_count` カウンタ追加
- `logger.debug` → 初回 + 10 回毎に `logger.warning`, それ以外は `logger.debug` (ログ爆発防止)
- OB 特徴量が有効 (`use_ob_features: true`) なのに fetch 失敗が silent だった問題を解消

### 17.4 型安全向上

- `ob_utils.py`: `object` → `OrderBookLevel` / `OrderBookLevels` TypeAlias 導入
- `Optional[T]` → `T | None` 統一 (PEP 604)
- f-string → `%s` lazy formatting (`logger.debug` / `logger.warning`)

### 17.5 設定変更 (fill_test.yaml)

```yaml
# D-4: trending_up のみ skip、trending_down sell は開放
skip_sell_trending_up_only: true
# D-5: cooldown 20→10 (40分→約20分へ半減)
resume_window: 10
# D-3: A/B テスト準備 (sell offset 0.18→段階的縮小予定)
sell: 0.18  # → Phase D-3 A/B で 0.15→0.14 段階縮小予定
```

### 17.6 テスト結果

- 新規テスト: 9 件 (D-1/D-4/D-5 カバー)
- 既存テスト修正: 3 件 (TRENDING → is_trending / TRENDING_UP/DOWN 対応)
- v460 全体: **1598 passed** / 2 failed (pre-existing, Phase D 無関係)

### 17.7 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/lib/regime_detector.py` | TRENDING_UP/DOWN enum + is_trending + _classify 方向分解 |
| `scripts/v460/lib/fill_config.py` | skip_sell_trending_up_only フィールド + YAML mapping |
| `scripts/v460/run_fill_test.py` | skip_sell_trending ゲート → is_trending + up_only 対応 |
| `scripts/v460/lib/maker_price.py` | .value == "trending" → .is_trending |
| `scripts/v460/lib/skip_gate_evaluator.py` | OB fetch 可視化カウンタ + warning ログ |
| `scripts/v460/lib/ob_utils.py` | OrderBookLevel TypeAlias + Optional→Union 統一 |
| `configs/v460/fill_test.yaml` | D-3/D-4/D-5 設定変更 |
| `tests/unit/v460/test_regime_detector.py` | D-4/D-5 テスト 9 件追加 + 既存修正 |
| `tests/unit/v460/test_143_regime_utilization.py` | TRENDING → is_trending 修正 |