# 067# Codex レビュー用情報パッケージ — AS-LR SkipGate + 次の一手

**目的**: 外部 AI エージェント (Codex) による AS-LR SkipGate 戦略レビュー + 次フェーズの方向性判断  
**日時**: 2026-02-15 JST  
**プロジェクト**: v460 "Microstructure Edge" BTC/JPY maker-only 自動売買  
**取引所**: Coincheck (日本国内、現物取引、maker 手数料 0%)  
**ステータス**: fill test **停止中** — 口座残高デッドロック (JPY 9,006 / BTC 0.00093)。月曜入金で再開予定。

---

## §0 結論先出し: Codex に判断を求めたい論点

### Q1. AS-LR SkipGate の妥当性

Skip20% +0.245 bps の改善は「166 samples, 12 features」の小サンプル LR で達成。
v459 では E2α の +41.95% が 2/4 seed で消えた前例がある。

- **この改善は統計的に信頼できるか？ walk-forward 6fold の信頼性は十分か？**
- **166 samples, k=12 (n/p=13.8) は過学習リスクとしてどう評価するか？**

### Q2. G1-info FAIL 後の戦略

公式 G1-info (000# §3.2) は **FAIL** (Cliff's Delta < 0.33)。
しかし AS-LR SkipGate は「方向予測ではなく悪い約定の除外」という異なるアプローチ。

- **G1 FAIL の状態で fill test を継続する判断は妥当か？**
- **AS-LR SkipGate は G1 の代替として十分か、それとも別の Gate が必要か？**

### Q3. 次に粘るべき方向

入金まで時間がある。モデル改善で粘れる余地はあるか？

- **OB 特徴量 (depth_imbalance_ob) への依存度 65% (FI) は脆弱性か？**
- **284 samples (OB 不要) vs 166 samples (OB 必須) のトレードオフでどちらを重視すべきか？**
- **v459 の 10+ 実験全 FAIL の教訓から、いつ「見切り」をつけるべきか？**

### Q4. 過去バージョンから活かすべき資産

v457–v459 で蓄積された統計検証手法・教訓のうち、v460 で未活用のものはあるか？

---

## §1 システム概要

### 戦略ロジック
1. Coincheck orderbook から best_bid/best_ask を取得
2. スプレッド × `spread_offset_ratio` だけ内側に maker limit 注文 (`post_only`)
3. 5 秒ポーリング → 約定 or 5 分タイムアウト
4. **SkipGate**: 注文前に AS-LR で P(adverse_selection) を推定。高リスク注文をスキップ
5. 約定後 30 秒の mid 変化で adverse selection を判定
6. buy/sell 交互、120 秒サイクル

### アーキテクチャ

```
configs/v460/fill_test.yaml (148行, 全設定一元管理)
    │
    ├── scripts/v460/run_fill_test.py (2086行, メインランナー)
    │   ├── FillTestConfig (dataclass)
    │   ├── FillTestRunner
    │   │   ├── SkipGate (skip_gate.py, 481行)
    │   │   │   ├── Primary: A(k=12) — OB 特徴量あり
    │   │   │   └── Fallback: Curated(k=5) — OB なし
    │   │   ├── ParamAdapter (方策A: offset 自動適応)
    │   │   ├── LotSizer (方策B: ロット自動適応)
    │   │   └── RegimeDetector (market regime)
    │   └── CoincheckAdapter (REST API, 749行)
    │
    ├── scripts/v460/ml/
    │   ├── walk_forward_as.py (271行, expanding window WF)
    │   └── as_classifier.py (280行, AS 二値分類)
    │
    └── models/v460/
        ├── skip_gate_as.pkl (Primary)
        └── skip_gate_as_fallback.pkl (Fallback)
```

---

## §2 Fill Test 実績 (491 records, 3 日間)

### 全体統計

| 指標 | 全データ (491) | クリーン (342) | ターゲット |
|------|---------------|---------------|-----------|
| fill_rate | 76.0% (373/491) | 83.0% (284/342) | ≥ 60% ✅ |
| AS_ratio | 39.1% (146/373) | 34.2% (97/284) | ≤ 50% ✅ |
| avg_pnl | -0.620 bps | -0.459 bps | ≥ 0 ❌ |
| median_pnl | -0.121 bps | — | — |
| cum_pnl | -231.2 bps | — | — |

### データ品質問題

| git_sha | 件数 | 品質 |
|---------|------|------|
| a9320c9a5 | 136 | ✅ 正常 (037# YAML config) |
| 51c02be69 | 106 | ✅ 正常 |
| ca1bcaed1 | 70 | ✅ 正常 |
| (空) + 34 | 149 | ❌ ゾンビプロセス由来 (043# 参照) |

**149 件 (30%) はゾンビプロセス由来**のデータ汚染。time_filter なし、regime なし、旧コード。  
クリーンデータ (342 件) では fill_rate 83.0%, AS 34.2% と改善が見られる。

### サイド別分析

| サイド | n | AS 率 | avg PnL |
|--------|---|-------|---------|
| buy | 192 | 39.6% | -0.301 bps |
| sell | 181 | 38.7% | -0.958 bps |

**sell 側が 3 倍悪い** (PnL -0.958 vs -0.301)。sell 後の価格上昇バイアスの可能性。

### 期間

- 2026-02-13 (Day 1): 211 records
- 2026-02-14 (Day 2): 220 records
- 2026-02-15 (Day 3): 60 records (早期停止 — 残高デッドロック)

---

## §3 AS-LR SkipGate モデル詳細

### Primary Model: A(k=12)

| 項目 | 値 |
|------|-----|
| **パイプライン** | SimpleImputer → SelectKBest(k=12) → StandardScaler → LogisticRegression(C=0.01, l2, balanced) |
| **学習データ** | 166 samples (filled + AS label + spread available) |
| **入力** | 39 features (10 base + 8 micro + 14 v2 + 3 interaction + 4 side-aligned) |
| **選択** | 12 features (SelectKBest chi2) |
| **n/p 比** | 166 / 12 = 13.8 |
| **AS rate** | 56.6% |
| **Walk-Forward** | 6 fold, expanding window, embargo=2 |
| **ROC-AUC** | 0.449 (mean) |
| **Skip20% 改善** | **+0.245 bps** (baseline: -1.063 bps) |
| **Jaccard 安定性** | **0.529** (12 features 中 6.3 が平均的に安定) |

#### Selected Features (重要度順)

| Feature | |LR coef| | OB依存 | 説明 |
|---------|-----------|--------|------|
| return_60s | 0.086 | No | 直近60秒リターン |
| depth_imbalance_ob | 0.065 | **Yes** | 板の買売深度比 |
| tfi_acceleration | 0.064 | No | TFI加速度 |
| velocity_300s | 0.060 | No | 約定速度5分 |
| return_300s | 0.060 | No | 直近300秒リターン |
| side_aligned_return_30s | 0.053 | No | サイド整合リターン30秒 |
| vpin_300s | 0.050 | No | VPIN 5分 |
| tfi_300s | 0.049 | No | Trade Flow Imbalance 5分 |
| realized_vol_300s | 0.035 | No | 実現ボラティリティ5分 |
| return_30s | 0.033 | No | 直近30秒リターン |
| trade_flow_imbalance_60s | 0.028 | No | TFI 1分 |
| buy_ratio | 0.028 | No | 買比率 |

### Fallback Model: Curated(k=5)

| 項目 | 値 |
|------|-----|
| **学習データ** | 284 samples (filled + AS label, spread 不要) |
| **入力** | 12 curated features (058# 実績ベース、全て OB 不要) |
| **選択** | 5 features (SelectKBest chi2) |
| **n/p 比** | 284 / 5 = 56.8 |
| **Skip20% 改善** | +0.019 bps (ほぼ中立) |

#### Selected Features

| Feature | |LR coef| | 説明 |
|---------|-----------|------|
| edge_bps | 0.103 | スプレッド内エッジ |
| vpin_30s | 0.097 | VPIN 30秒 |
| vpin_300s | 0.056 | VPIN 5分 |
| hour_cos | 0.053 | 時刻コサイン |
| log_queue_wait | 0.046 | 待機時間対数 |

### Two-Tier 動作

```
注文前 → SkipGate.evaluate(features)
  ├── OB特徴量 (depth_imbalance_ob, spread_bps_ob) が NaN でない
  │   └── Primary (A(k=12), 12 features) で判定
  └── OB特徴量が全て NaN
      └── Fallback (Curated(k=5), 5 features) で判定 → 実質中立
```

Coincheck WebSocket でリアルタイム OB 取得中 → 通常は **Primary が 99% 使用**。

---

## §4 Walk-Forward 検証結果

### 6-Model 比較 (066# Trade-Only 検証)

| Model | ROC-AUC | Skip20% (bps) | Samples | Features | Jaccard |
|---|---|---|---|---|---|
| **A: Enriched (OB+spread=True)** | 0.449 (k=12採用) | **+0.245** | 166 | 12 select/39 input | **0.529** |
| B: Trade-only k=8 | 0.530 | +0.007 | 284 | 8 select/27 input | 0.067 |
| C: Base-only | 0.509 | -0.354 | 284 | 8/8 | 1.000 |
| D: Full+NaN impute | 0.522 | -0.136 | 284 | 8 select/39 input | 0.050 |
| E: Trade-only k=5 | 0.519 | +0.001 | 284 | 5 select/27 input | 0.077 |
| F: Trade-only k=3 | **0.536** | -0.027 | 284 | 3 select/27 input | 0.111 |

**核心的発見**: Trade-only モデルは ROC-AUC が高い (0.530–0.536) が Skip20% はほぼゼロ。  
OB 特徴量 (`depth_imbalance_ob`) は ROC-AUC を下げるが、**極端な AS 事象のピンポイント検出**に不可欠。

### HP Sweep (k=3,5,8,12)

| k | Skip20% (bps) | Jaccard | 評価 |
|---|---|---|---|
| 3 | -0.059 | 0.167 | 過少 |
| 5 | +0.162 | 0.200 | 中程度 |
| 8 | +0.230 | 0.417 | 旧設定 |
| **12** | **+0.245** | **0.529** | **採用** |

---

## §5 Gate 状態サマリ

| Gate | 状態 | 根拠 |
|------|------|------|
| **G0-data** | ✅ PASS | データハッシュ一致、NaN < 1% |
| **G1-info** | ❌ **FAIL** | 公式基準で全 9 ターゲット FAIL (Cliff's Delta max=0.208 < 0.33)。IC は h5/h15 で > 0.02 |
| **G1.1-exec** | 🔄 進行中 | fill_rate 83.0% ≥ 60% ✅, AS 34.2% ≤ 50% ✅, avg_pnl -0.459 bps ❌ |
| **G2-train** | 🔒 未着手 | G1 FAIL により RL (SAC) への進行は 000# により制限 |
| **G3-pnl** | 🔒 未着手 | — |
| **G4-live** | 🔒 未着手 | — |

### G1 FAIL の緩和策 (現行)

000# §3.2 の G1 FAIL 時ルール: 「特徴量再設計へ戻る。RL には進まない。」

現在のアプローチは:
- **G1 (方向予測) は FAIL** → SAC 学習は保留
- **AS-LR SkipGate は G1 の範囲外** → 「方向を予測する」のではなく「悪い約定を除外する」
- fill test を実運用データで SkipGate の効果を実測中

**これが正しい判断かを Codex に確認したい** (Q2)。

---

## §6 過去バージョンからの教訓と活用状況

### v459 (119 文書, No-Go)

| 教訓 | v460 での対応 |
|------|-------------|
| K2: OHLCV 特徴量は情報不足 | ✅ マイクロストラクチャ 12 特徴量に置換 |
| E1: Taker 0.1% では不成立 | ✅ maker-only post_only 戦略 |
| E2α: 単一 seed の成功を信じるな | ⚠️ WF 6fold だが単一データセット |
| E2: ペナルティ過多→HOLD偏重 | ✅ RL 不使用、Skip/Trade 2値判定 |
| 105#: 手数料支配の分析 | ✅ maker 0% 前提で費用問題を解消 |
| 113#: H1 時間帯限定エッジ | ✅ time_filter で高AS時間スキップ |
| 113#: H2 報酬が行動頻度を学習 | ✅ 統計ベース AS-LR で回避 |
| 101#: window分割でbaseline | ⚠️ 未完全適用 — 検証余地あり |

### v458 (19 文書)

| 資産 | 適用 |
|------|------|
| Walk-Forward 評価基盤 | ✅ ztb/evaluation/ 基盤を流用 |
| Go/No-Go「全WF窓でPF>1.05」 | ⚠️ AS-LR には未適用 |
| 05#: 新機能より既存資産統合 | ✅ 方針として踏襲 |
| 16#: AI レビュー 10 視点 | ✅ 本パッケージで活用 |

### v457 (32 文書)

| 資産 | 適用 |
|------|------|
| v451 Golden Config: hold_penalty=0 | — RL 不使用のため直接関係なし |
| v457.3: TTL固定の有効性 | ✅ order_timeout_sec=300 固定 |
| Config散乱問題 | ✅ YAML一元管理で解消 |

### 未活用の有力資産

| 資産 | 出典 | 適用案 |
|------|------|--------|
| **Mann-Whitney + Holm + Cliff's Delta** | v459 0# §5.6 | AS-LR の Skip/No-skip グループ間の PnL 差検定に利用可 |
| **4seed × 4split 標準** | v458 05# | AS-LR の fold 数増加 or bootstrap 適用 |
| **Action bin 別 PnL** | v459 116# §2.2 | AS 確率 bin 別の PnL 分布分析 |
| **反実仮想ログ** | v459 116# §8 | fill test でパラメータ感度分析 |
| **Deterministic baseline** | v459 101# | threshold 無関係の random skip baseline |

---

## §7 設定ファイル全文

[configs/v460/fill_test.yaml](../../configs/v460/fill_test.yaml) (128行)

```yaml
# 主要パラメータ抜粋
symbol: btc_jpy
order_quantity: 0.001
cycle_interval_sec: 120.0
spread_offset_ratio: 0.05
as_deadzone_bps: 2.5
time_filter.skip_utc_hours: [1,2,8,9,12,13,14,16,17,18,19,21]  # 12/24h スキップ

skip_gate:
  enabled: true
  mode: as
  model_path: models/v460/skip_gate_as.pkl
  fallback_path: models/v460/skip_gate_as_fallback.pkl
  as_threshold: 0.65
  max_skip_rate: 0.3

safety:
  loss_cap_auto: true
  loss_cap_ratio: 0.05
```

---

## §8 リスクマトリクス

| 重要度 | リスク | 現状 | 緩和策 |
|--------|--------|------|--------|
| ⭐⭐⭐ | **AS-LR が過学習** | 166 samples, 6fold WF | k=12 で n/p=13.8。bootstrap/更なる WF が必要か |
| ⭐⭐⭐ | **G1 FAIL → 全体戦略の瓦解** | AS-LR は G1 判定外 | 000# の Gate 体系に AS-LR をどう位置付けるか |
| ⭐⭐⭐ | **OB 依存のリスク** | depth_imbalance_ob が FI 2位 | WebSocket 断絶時は Fallback (ほぼ中立) に退化 |
| ⭐⭐ | **sell 側の構造的劣位** | sell PnL -0.958 vs buy -0.301 | 非対称 offset (sell: 0.10) で緩和中 |
| ⭐⭐ | **sample 増加に時間がかかる** | 120s/cycle → 720 cycles/日 → 540 filled/日 | 入金後 7 日で ~3,780 filled。n/p 比は改善 |
| ⭐⭐ | **time_filter 50% スキップ** | 12h/24h スキップ → 機会損失 | AS 率に基づく設定で正当化済 (052#) |
| ⭐ | **市場レジーム変化** | BTC 相場転換時のモデル陳腐化 | 定期再訓練 (方策 B) + レジーム検知 |

---

## §9 Codex への質問リスト

1. **統計的妥当性**: 166 samples × 6fold WF × Skip20% +0.245 bps — この改善が偶然でない信頼度は？ bootstrap p-value を計算すべきか？
2. **Gate 体系の整合性**: G1 FAIL 状態での AS-LR SkipGate は 000# の Gate 順守を逸脱しているか？ G1 の「特徴量再設計」と両立するためにどう位置付けるべきか？
3. **OB 特徴量の脆弱性**: depth_imbalance_ob なしでは Skip20% がゼロ。これは戦略として脆弱か、あるいは OB が利用可能な限り問題ないか？
4. **sell 側の改善余地**: PnL -0.958 bps (buy の 3 倍悪い)。side 別 SkipGate threshold、side 別モデル、sell 一時停止のいずれが適切か？
5. **次の改善方向**: (a) 特徴量エンジニアリング (064# の vwap_deviation 等)、(b) モデル変更 (LR→XGBoost/RF)、(c) threshold 最適化、(d) サンプル増加待ち — 優先順位は？
6. **v459 教訓の活用漏れ**: 上記 §6 の「未活用資産」から、今すぐ適用すべきものはあるか？
7. **撤退基準**: v459 は Phase E で 10+ 実験全 FAIL の後に No-Go。v460 の AS-LR がこのまま PnL マイナスを続けた場合、何サイクルで見切るべきか？ 000# §3.9 の継続中止ルールで十分か？
8. **time_filter 50% スキップ**: 24h 中 12h を棄却。攻撃的すぎないか？ AS 率ベースの動的 filter への移行は有効か？

---

## §10 主要ファイルの場所

| ファイル | パス | 行数 |
|----------|------|------|
| fill test ランナー | scripts/v460/run_fill_test.py | 2086 |
| Coincheck adapter | ztb/trading/live/exchanges/coincheck/adapter.py | 749 |
| SkipGate | scripts/v460/ml/skip_gate.py | 481 |
| Walk-Forward AS | scripts/v460/ml/walk_forward_as.py | 271 |
| AS 分類器 | scripts/v460/ml/as_classifier.py | 280 |
| fill quality metrics | ztb/metrics/fill_quality.py | — |
| YAML 設定 | configs/v460/fill_test.yaml | 128 |
| Primary model | models/v460/skip_gate_as.pkl | — |
| Fallback model | models/v460/skip_gate_as_fallback.pkl | — |
| fill records (Day 1) | results/v460/fill_test/fill_records_20260213.jsonl | 211 |
| fill records (Day 2) | results/v460/fill_test/fill_records_20260214.jsonl | 220 |
| fill records (Day 3) | results/v460/fill_test/fill_records_20260215.jsonl | 60 |
| 000# プロジェクト定義 | docs/v460/000_ph0_plan_project_proposal.md | 209 |
| 043# 前回 Codex レビュー | docs/v460/043_ph2_codex_review_package.md | 275 |
| 064# G1-info 再検証 | docs/v460/064_ph1_g1_info_verify.md | — |
| 065# AS-LR 実装 rev | docs/v460/065_ph1_rev_064.md | — |
| 065# AS-LR 実装 impl | docs/v460/065_ph1_impl_as_lr_prep.md | — |
| 066# Phase 2 Two-Tier | docs/v460/066_ph2_rpt_trade_only_two_tier.md | 111 |
| テストスイート | tests/unit/v460/ | 644 passed |

---

## §11 Git コミット履歴 (最新)

```
f96fcf8f9 066# docs: 065#整理 + 000# archive方針改定 + Codex review package
ee6a049fe refactor: type signal performance analysis and extend history helper
980fdaaa0 refactor: complete ASG components Any cleanup and helper rollout
d919a86f9 refactor: extend history helper and type sac/cache/plugin components
39b0707ff 065# ph2: Two-Tier SkipGate (A(k=12) primary + curated(k=5) fallback) + SimpleImputer fix
73cd7d6f7 refactor: type validation and pattern statistics payloads
7a970958b 065# AS-LR SkipGate prep: G1 FAIL confirmed, skip_gate_as.pkl trained, fill_test.yaml enabled
1bd4c74e6 refactor(gann): add shared base and remove Any debt
1b6c25e8d 064# ph1: G1-info再検証 PASS (簡易基準), vwap_deviation/toxicity/ask_slope有力
1082c5dad 063# ph3: SAC重複実装整理
0c4c0f896 062# ph2: AS SkipGate -> fill_test live integration
```

---

## §12 043# からの変化 (前回 Codex レビューとの差分)

| 項目 | 043# 時点 (02-14) | 現在 (02-15) |
|------|-------------------|-------------|
| fill records | 355 件 | 491 件 (+136) |
| クリーン records | ~206 件 | 342 件 |
| AS_ratio (clean) | ~42.7% | 34.2% (改善) |
| SkipGate | 未実装 | Two-Tier AS-LR 実装済 |
| G1-info | 未正式評価 | 公式 FAIL 確定 |
| 043# Bug 7 (排他) | 未修正 | PID ロックファイル実装済 |
| データ品質 | 149件汚染 | 汚染は固定化 (追加なし) |
| テスト | 396 passed | 644 passed (+248) |
| 口座残高 | ~19K JPY | デッドロック (JPY 9K, BTC 0.00093) |

---

## §13 口座残高 (現在)

| 通貨 | free | reserved | 備考 |
|------|------|----------|------|
| JPY | ~9,006 | 0 | sell 不足 (BTC < 0.001) で buy → 残高減少 |
| BTC | 0.00093 | 0 | 最小ロット (0.001) 未満 → sell 不可 |

**推定口座総額**: ~18,700 JPY  
**状態**: fill test 停止中。月曜入金 (10,000+ JPY) で再開予定。
