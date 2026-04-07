# 710# 再起動前アクション整理 & 方針比較 & Codex タスク & Deep Analysis

## 概要

708#-709# の全変更を統合レビューし、再起動前の最終確認・パラメータ修正・残課題の方針比較を行った。
新規 Codex タスク CX4-CX6 を設計。後半でself-task deep analysis (4 phase) を実施し、追加YAML変更を決定。

---

## 1. 708#-709# デプロイ変更一覧

| # | 変更 | 影響 | 安全性 |
|---|------|------|--------|
| 1 | `trend_5s_sell_guard: disabled` | +136.8 bps counterfactual (32 sells 解放) | 高 |
| 2 | `buy base_offset: 0.05→0.08` | buy 攻撃性低下、ranging 損失軽減 | 高 |
| 3 | `sell/trending_up hard_skip_mult: 5.0` | 31→3 hard_skip, 28 sells 救出 (+3.77bps/fill) | 高 |
| 4 | `AS trailing gate soft: 0.30→0.20` | 1.3x boost 到達率 7%→~25% | 高 |
| 5 | SAG redesign opt-in (false) | インフラのみ | 無影響 |
| 6 | entry_gate stale check 順序修正 | buy suppress が auto_disable 前に評価 | 高 |
| 7 | **710# SAG param fix**: ref 4.0→2.0, cap 2.0→1.5 | コードデフォルト修正 | 無影響(disabled) |

---

## 2. 発見: CX3 SAG redesign パラメータバグ

CX3 のデフォルト `inverse_penalty_reference_bps=4.0` は、median spread ≈ 2.05 に対して全帯域でペナルティをほぼ 2 倍にする。

```
penalty = ev_penalty_bps × (reference / spread)
= 0.5 × (4.0 / 2.05) ≈ 0.976 bps  (現行 flat: 0.500)
```

**修正**: reference_bps=2.0, cap_bps=1.5 (710#)

修正後:
| spread 帯 | n | redesign avg | flat avg | 差 |
|-----------|---|-------------|---------|-----|
| <1.5 bps | 106 | 0.881 | 0.500 | +76% (狭スプレッド=高AS→重ペナ ✓) |
| 1.5-2.0 | 125 | 0.577 | 0.500 | +15% |
| 2.0-2.5 | 142 | 0.446 | 0.500 | -11% |
| 2.5-3.0 | 87 | 0.370 | 0.500 | -26% (広スプレッド=低AS→軽ペナ ✓) |
| >3.0 | 39 | 0.308 | 0.500 | -38% |
| **全体** | **499** | **0.547** | **0.500** | **+9.4%** |

方向性は正しい: 狭スプレッド（AS リスク大）に重く、広スプレッド（AS リスク小）に軽い。全体微増は許容範囲。

---

## 3. 残課題の方針比較

### A. buy/ranging 損失 (n=215, avg=-0.349 bps) — **最大損失源**

| アプローチ | 概要 | 期待効果 | リスク | 推奨 |
|-----------|------|---------|--------|------|
| **A1: base_offset 追加引上げ** | 0.08→0.10-0.12 | fill 単価改善 | fill rate 低下 | △ 0.08 の効果を見てから |
| **A2: skip_gate 復活** | bypass_mode=false + thresh=0.6 | 低品質 fill 排除 | fill rate 低下、model 精度不明 | ◎ 段階的に |
| **A3: OBI 絶対値ベース guard** | \|OBI\|>0.25 で offset boost | U字型の両端を防御 | 新機能実装必要 | ○ Codex 向け |
| **A4: ranging 頻度チェック** | ranging 連続回数で escalation | 負のストリーク遮断 | complexity 増 | △ |

**推奨**: A2 を段階的に実施（データ蓄積後）、A3 を Codex で準備

### B. skip_gate bypass_mode 停止 — **高優先度**

| アプローチ | 概要 | 期待効果 | リスク |
|-----------|------|---------|--------|
| **B1: 即座に bypass=false** | ブロッキング有効化 | 低品質 fill 排除 | model 精度未検証で過剰 skip |
| **B2: threshold 引上げ + bypass=false** | 0.1→0.6 + blocking 再開 | 高確信のみブロック | 安全だが効果も限定的 |
| **B3: gradual rollout** | side 別に有効化 (sell 先行) | sell は avg=+0.048 で余裕あり | sell のみでは buy 改善なし |

**推奨**: B2 (thresh=0.6 → bypass=false) を次期データ期間で実施

### C. entry_gate 復活

| アプローチ | 概要 | 期待効果 | リスク |
|-----------|------|---------|--------|
| **C1: CalibrationMap 再学習** | 直近データで batch retrain | EV 分布が現実的に | 再学習しても -1.9 に集中する可能性 |
| **C2: ev_threshold 引下げ** | -0.5→-2.0 で全通し | dead layer の形式化 | entry_gate の意味消失 |
| **C3: hybrid EV** | CalibrationMap EV + raw spread/AS features のブレンド | 多信号活用 | 設計が複雑 |

**推奨**: C1 を Codex で実施、結果次第で C3

### D. SAG redesign AB 実験

| アプローチ | 概要 | 期待効果 | リスク |
|-----------|------|---------|--------|
| **D1: redesign_enabled=true (hot-reload)** | 即時有効化 | 狭スプレッド防御強化 | 全体 +9.4% にペナ増 |
| **D2: AB split test** | fill cycle 交互に flat/inverse | 因果推論可能 | 実装コスト |
| **D3: 708# 変更効果判定後** | 24-48h 待ち | baseline 安定 | 時間喪失 |

**推奨**: D3（まず 708# 効果を評価してから D1）

### E. sell+trending_down (n=22, avg=-0.396)

| アプローチ | 概要 | 期待効果 | リスク |
|-----------|------|---------|--------|
| **E1: sell_trending_down_offset 調整** | offset 増加 | 損失軽減 | n 小さく過学習 |
| **E2: 静観** | 708# の trend_5s 無効化で改善待ち | 間接効果 | 不確実 |

**推奨**: E2（n=22 で median は +0.175。大局的には noise レベル）

---

## 4. 新規 Codex タスク

### CX4: skip_gate bypass_mode 段階的停止フレームワーク

**目的**: skip_gate を observe→active に安全に切り替える

**要件**:
1. `bypass_mode` を `false` にする前に、threshold の最適値を走査
2. CX1 の `skip_gate_quality_analysis.py` 結果を利用
3. threshold=0.6 でのブロック率・PnL 影響のドライラン CLI
4. side 別有効化オプション（sell 先行の pathway）
5. fill rate impact の推定器

### CX5: CalibrationMap 再学習 + entry_gate 復活準備

**目的**: entry_gate の CalibrationMap を直近データで再学習し、EV 分布を現実化

---

## 5. Self-Task Deep Analysis (710# 後半)

### 5.1 分析概要

4 phase の deep analysis を Apr 4-6 データ (499 fills) で実施。

### 5.2 主要発見

#### 5.2.1 skip_gate adaptive threshold simulation

| 側 | target_skip_rate | adaptive_thresh | 改善(bps) | 判定 |
|----|-----------------|-----------------|-----------|------|
| buy | 0.15 | -0.7805 | **+27.5** | 有効 |
| sell | 0.20 | -0.7900 | **-15.6** | **有害** |

**sell 側の skip_gate モデルは反転** — 低スコア fill が高収益 (blocked avg=+0.323, rest avg=-0.020)。
CX4 の side-aware bypass が不可欠。

#### 5.2.2 velocity_offset_mult フィールド問題

`velocity_offset_mult` フィールドが fill_records に None で記録されており、velocity 防御が無効に見えたが、
実際は **`executor_offset_stages.velocity` に正しく記録** されており機能していた（例: vel=-6.10 → mult=1.762）。

velocity threshold -4.0 での発動効果:
- 発動時 12 fills: avg PnL = **-0.114** (ほぼ break-even)
- 非発動時 243 fills: avg PnL = **-0.371**
→ velocity defense は実証済みの有効性。

#### 5.2.3 buy/ranging worst/best 分析

| 指標 | worst 20% (n=43) | best 20% (n=43) | 差 |
|------|------------------|-----------------|-----|
| OBI med | -0.019 | 0.030 | 微差 |
| velocity avg | 0.16 | 0.06 | 微差 |
| spread avg | 2.09 | 2.04 | 微差 |
| offset avg | 0.1457 | 0.1382 | 微差 |

**worst/best fills は全特徴量で分離不能** → 逆選択が確率的（stochastic AS）。
閾値チューニングでの改善は構造的に限界あり。

#### 5.2.4 OBI U-shape 確認 (`orderbook_imbalance` フィールド)

前回フィールド名を `obi` と誤り "none" と報告。正しくは `orderbook_imbalance` で **全 499 fills にデータあり**。

| OBI zone | buy/ranging n | avg PnL | 判定 |
|----------|--------------|---------|------|
| sell_heavy (<-0.1) | 75 | -0.578 | 損失 |
| **neutral (-0.1~0.1)** | **52** | **+0.180** | **唯一の黒字** |
| buy_heavy (>0.1) | 88 | -0.466 | 損失 |

U-shape 完全確認。CX6 の absolute/quadratic OBI モードで対応予定。

#### 5.2.5 buy ceiling ヒット率

buy ceiling (0.35) ヒット率はわずか **7.5%** (19/255)。ceiling 引上げの効果は限定的。
ただし ceiling ヒット + velocity boost の 4 fills は avg=**+2.150** と好成績。

### 5.3 追加 YAML 変更

| 変更 | 値 | 根拠 |
|------|-----|------|
| `buy_velocity_skip_threshold_bps` | -4.0→**-3.0** | -3~-4 帯 7 fills avg=-1.39。velocity defense (-4.0 発動時 avg=-0.114) の有効性実証済み |

#### 見送り事項

| 案 | 判断 | 理由 |
|----|------|------|
| buy offset ceiling 0.35→0.40 | 見送り | ceiling ヒット率 7.5%、影響軽微 |
| buy base offset 0.08→0.10 | 見送り | 708# 変更後、初期データで再変更は早計 |
| skip_gate bypass=false (buy) | CX4 待ち | sell 有害確認、side-aware 実装が前提 |
| OBI absolute/quadratic filter | CX6 待ち | code 変更必要 |
| min_spread_floor 0.38→1.0 | 見送り | fill rate 影響が大きく、追加分析必要 |

### 5.4 CX4-CX6 への追加インプット

- **CX4**: sell bypass は**絶対に維持**。sell SG モデルは反転（低スコア=高収益）
- **CX5**: entry_gate CalibrationMap の再学習は OBI 修正後に実施が効果的
- **CX6**: `orderbook_imbalance` フィールド名確認済み。全499件にデータあり。U-shape の安全帯は OBI ∈ [-0.1, 0.1]

**要件**:
1. `scripts/v460/ml/calibration_batch.py` を最新 fill_records で実行
2. 再学習後の EV 分布を 708# 分析と比較
3. `buy_suppress_ev_threshold` の最適値を推定
4. entry_gate 有効化時の counterfactual 影響レポート

### CX6: OBI 絶対値ベース非対称ファクター

**目的**: 707# の OBI 線形モデルを U 字型対応に改善

**要件**:
1. 現行: `offset += OBI * factor` (buy_heavy 時に offset を下げる → 危険)
2. 新設計: `offset += |OBI| * factor` (両端で等しく警戒)
   または `offset += (OBI² / threshold) * factor`
3. `ranging_obi_asymmetry_factor` と `ranging_obi_threshold` の変更
4. テスト追加: U 字型データでの動作検証
5. OBI 中立帯 (|OBI|<0.10) では factor=0 のデッドバンド維持

---

## 5. 再起動前チェックリスト

- [x] trend_5s disabled
- [x] buy offset 0.08
- [x] sell/trending_up hard_skip 5.0
- [x] AS trailing gate soft 0.20
- [x] CX2 entry_gate 順序修正
- [x] CX3 SAG redesign (opt-in, disabled)
- [x] **710# SAG param fix (ref 2.0, cap 1.5)**
- [x] index.md 更新 (701#-709#)
- [x] テスト全通過 (186/186 focused, 4533/4533 broad)
- [ ] 再起動後 24-48h モニタリング

---

## 6. 再現コマンド

```bash
# SAG redesign 影響検証
.venv/Scripts/python.exe temp/pre_restart_analysis.py

# CX1 skip_gate quality 分析
.venv/Scripts/python.exe -m scripts.v460.analysis.analyze_708_skip_gate_quality \
  --results-dir results/v460/fill_test \
  --date-from 2026-04-01 --date-to 2026-04-06 --json
```
