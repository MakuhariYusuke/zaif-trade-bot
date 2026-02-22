# 108# Phase D1 特徴量スクリーニング実験結果

**Date**: 2026-02-10
**対象**: 107# Phase D 改訂版 §4.2 D1: 既存特徴量セット比較
**実施**: seed=42 粗選別 (50K timesteps × 3 実験)
**大義**: 短期間での高収益性 — 特徴量の次元を変えて予測優位が改善するか判定

---

## §0 結論（先に要点）

1. **3 実験すべて Gate2 FAIL** — 特徴量数の増加 (8 → 25 → 73) は 50K では有意な改善をもたらさなかった
2. **支配的ボトルネックは特徴量ではなく手数料** — 全実験で Gross PnL (|-1.6K〜-1.9K|) に対し Fees (13K+) が 7〜8 倍
3. **medium (25 特徴) に深刻な SELL バイアス** (47.4% Sell vs 7.9% Buy) → action_mean=-0.263 で偏り学習
4. **full_registry (73 特徴) は NaN/Inf 汚染** あり (57 件の non-finite 警告) → 特徴量品質に問題
5. **D1 粗選別の結論**: 特徴量は現段階の主因ではない (non-bottleneck)。D2 コスト感度分析に進む

---

## §1 実験条件

### §1.1 共通パラメータ（107# §4.2 準拠）

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| timesteps | 50,000 | 106#/107# 50K 固定制約 |
| seed | 42 | D1 Step 1 粗選別 |
| ent_coef | 0.01 | C3 最良条件 |
| threshold | 0.70 | C3 最良条件 |
| gamma | 0.99 | C3 固定 |
| reward_scale | 100.0 | C3 固定 |
| transaction_cost | 0.001 (0.1% 片道) | Zaif taker 手数料 |
| eval_dd_thresholds | [1.0, 0.30] | 107# §2.2 multi_dd_eval |
| eval 方式 | normalized (VecNormalize 適用) + raw obs 比較 | Phase C 標準 |

### §1.2 特徴量セット定義

| 実験 | 特徴量セット | 特徴数 | 構成 | Parquet |
|------|------------|---------|------|---------|
| d1_v451opt | v451_optimized | 8 | Close/SMA5/EMA12/RSI14/BB_upper/BB_lower/MACD/Signal | `data/btc_jpy_1m_v451_optimized_features.parquet` |
| d1_medium | medium | 25 | RSI(7) + Scalping(12) + ATR(3) + Time(3) | `data/btc_jpy_1m_medium_features.parquet` |
| d1_full_registry | full_registry | 73 | Ichimoku(42) + RSI(7) + Scalping(12) + Time(9) + ATR(3) | `data/btc_jpy_1m_full_registry_features.parquet` |

### §1.3 Parquet データ

- 全セット共通: **1,216,930 行** (BTC/JPY 1 分足 OHLCV)
- OHLCV 5 カラムは特徴量とは別に Parquet に含まれる
- v451_optimized: 14 cols (5 OHLCV + 1 Volume + 8 features)
- medium: 31 cols (5 OHLCV + 1 Volume + 25 features)
- full_registry: 77 cols (5 OHLCV + 1 Volume + 73 features) — ただし一部特徴量に NaN/Inf あり

---

## §2 実験結果

### §2.1 主要メトリクス比較

| メトリクス | d1_v451opt (8特徴) | d1_medium (25特徴) | d1_full_registry (73特徴) | 備考 |
|-----------|-------------------|-------------------|--------------------------|------|
| **Net ROI** | -15.01% | -15.32% | -15.00% | 全て ≈-15% に収束 |
| **Gross PnL** | -1,638 JPY | -1,912 JPY | -1,908 JPY | Gross 損失は特徴量で差異小 |
| **Total Fees** | 13,376 JPY | 13,409 JPY | 13,096 JPY | 全体の損失を支配 |
| **Fee/|Gross| Ratio** | 8.2× | 7.0× | 6.9× | 手数料が Gross の 7-8 倍 |
| Total Trades (学習) | 866 | 846 | 812 | — |
| **Gate2 PASS** | **FAIL** | **FAIL** | **FAIL** | 全て不合格 |

### §2.2 Gate2 評価メトリクス (eval, normalized)

| メトリクス | d1_v451opt | d1_medium | d1_full_registry | Gate2 基準 (0#) |
|-----------|-----------|-----------|-----------------|----------------|
| **Sharpe** | -2.84 | -2.45 | **-4.22** | > 1.0 |
| **MaxDD** | -0.085% | -0.083% | **-0.121%** | < -15% |
| **PF** | 0.982 | 0.984 | 0.973 | > 1.20 |
| **StepWR** | 48.9% | 49.2% | 49.1% | > 35% (PASS) |
| **TradeWR** | 29.7% | 31.1% | **36.6%** | > 35% (borderline) |
| MTM ROI (eval) | -6.66% | -6.31% | **-9.85%** | > 5% |
| Eval Trades | 263 | 277 | **865** | — |
| Avg Gross/Trade | -16.03 | -8.08 | 11.03 | — |
| Avg Fee/Trade | 32.77 | 33.29 | 33.59 | — |
| Avg Net/Trade | -48.80 | -41.37 | -22.56 | — |
| Binom p-value | 2.1e-6 | 4.0e-6 | 1.7e-8 | — |

### §2.3 Multi DD 評価 (dd100 / dd030)

| 実験 | DD閾値 | Eval Net ROI | PF | TradeWR | MaxDD | Gate2 |
|------|--------|-------------|-----|---------|-------|-------|
| d1_v451opt | 1.0 | -5.31% | 0.986 | 31.2% | -7.2% | FAIL |
| d1_v451opt | 0.3 | -6.49% | 0.982 | 30.7% | -8.5% | FAIL |
| d1_medium | 1.0 | -6.47% | 0.983 | 29.8% | -8.4% | FAIL |
| d1_medium | 0.3 | -6.56% | 0.983 | 28.7% | -8.5% | FAIL |
| d1_full_registry | 1.0 | -5.81% | 0.984 | 37.0% | **-11.1%** | FAIL |
| d1_full_registry | 0.3 | -5.83% | 0.984 | 36.8% | **-11.2%** | FAIL |

**観察**: dd100/dd030 間の差異は微小 → eval_dd_threshold 問題は D1 レベルでは顕在化せず

### §2.4 学習中行動分布

| 実験 | HOLD% | BUY% | SELL% | Action Mean | Action Std | 特記 |
|------|-------|------|-------|-------------|-----------|------|
| d1_v451opt | **59.3%** | 21.4% | 19.4% | -0.000 | 0.298 | HOLD 優勢 (健全) |
| d1_medium | 44.7% | **7.9%** | **47.4%** | **-0.263** | 0.342 | ⚠️ SELL バイアス |
| d1_full_registry | 34.9% | 23.7% | **41.4%** | -0.095 | 0.335 | やや SELL 寄り |

### §2.5 計算資源

| 実験 | 所要時間 | Memory Peak | Non-finite 警告数 | Emergency Stop |
|------|---------|------------|-------------------|----------------|
| d1_v451opt | 21.4 分 (1283s) | ~919 MB | 0 | step 8127 (15% DD) |
| d1_medium | 22.0 分 (1322s) | ~1229 MB | 2 | step 21601 (15.3% DD) |
| d1_full_registry | 23.5 分 (1408s) | ~2100 MB | **57** | なし (DD 未到達) |

---

## §3 考察

### §3.1 特徴量数と性能の関係（D1 結論）

**核心的発見: 特徴量を 8 → 25 → 73 に増やしても Net ROI は -15% 前後で収束する。**

原因の分析:
1. **手数料支配**: 全実験で Total Fees ≈ 13K JPY に対し Gross PnL は |-1.6K 〜 -1.9K| JPY。Net Loss の 87-89% は手数料によるもの。特徴量の quality を変えても、取引数が 800-870 回で安定している以上、手数料構造が変わらない限り Net ROI は改善しない。
2. **50K steps での限界**: SAC が 73 特徴量を有効に学習するには 50K steps は不十分な可能性。full_registry は eval_trades=865 (他の 3 倍) → 過剰取引傾向。学習が収束する前に評価されている可能性が高い。
3. **NaN 汚染の影響**: full_registry は 57 件の non-finite 警告 → 一部特徴量がゼロ埋めされ、入力情報が劣化。Ichimoku 系の長期ウィンドウ特徴量は初期区間で NaN を生じやすい。

### §3.2 SELL バイアス問題 (d1_medium)

d1_medium は action_mean=-0.263 で明確な SELL バイアスを示した:
- BUY 7.9% vs SELL 47.4% → SELL が BUY の **6 倍**
- これは「特定の特徴量の分布が action を偏向させた」可能性が高い
- medium 特有の Scalping 系 12 特徴（VWAP 乖離、約定インバランスなど）が原因か
- **対策**: 特徴量の正規化・クリッピングが必要。FeatureSetManager の正規化パイプラインの改善が D2 以降の課題

### §3.3 full_registry の矛盾 — TradeWR は最高だが Sharpe は最悪

| 指標 | full_registry が最良 | full_registry が最悪 |
|------|--------------------|--------------------|
| TradeWR | **36.6%** (vs 29.7/31.1%) | — |
| Avg Gross/Trade | **+11.03** (唯一正) | — |
| Sharpe | — | **-4.22** (vs -2.84/-2.45) |
| MaxDD | — | **-0.12%** (vs -0.08%) |
| Eval Net ROI | — | **-9.85%** (vs -6.66/-6.31%) |

解釈:
- 1 取引あたりの粗利は最良 (Avg Gross=+11.03) だが、取引回数が 865 回と突出して多い
- 取引頻度 × 手数料 が Sharpe を悪化させている
- 膨大な特徴量 → 低い HOLD 率 (34.9%) → 過剰取引 → 手数料破産パターン
- **示唆**: 特徴量数の増加は「取引の質」を部分的に改善するが、「取引頻度」の制御が伴わないと無意味

### §3.4 二項検定結果の解釈

全実験の binom_p_value は極めて小さい (2e-6 ~ 2e-8):
- TradeWR が 50% よりも有意に **低い** ことを示す（29-37% vs 50%）
- つまり「ランダムより有意に悪い」= **学習した戦略がランダム以下**
- これは手数料構造上ほぼ必然: 往復 0.2% のコストがあるため、PnL>0 の閾値が高くなる
- 真に意味のある検定は「手数料控除前の方向予測精度」vs 50% であるべき → D2 以降で検討

### §3.5 Emergency Stop の影響

| 実験 | Stop Step | 残り Steps | 影響 |
|------|-----------|-----------|------|
| d1_v451opt | 8,127 | 41,873 | 学習の 83.7% を DD 後に実施 → 回復学習のバイアス |
| d1_medium | 21,601 | 28,399 | 学習の 56.8% を DD 後に実施 |
| d1_full_registry | なし | — | DD 閾値 (15%) 未到達 |

Emergency stop は学習を中断しないが、ポートフォリオをリセットする。この影響の定量化は未実施。

---

## §4 D1 判定と D2 への進行判断

### §4.1 D1 粗選別結果

107# §4.2 Gate 基準: 「trade_win_rate, PF, avg_gross_per_trade の相対順位で上位を選定」

| 指標 | 1 位 | 2 位 | 3 位 |
|------|------|------|------|
| PF | d1_medium (0.984) | d1_v451opt (0.982) | d1_full (0.973) |
| TradeWR | d1_full (36.6%) | d1_medium (31.1%) | d1_v451opt (29.7%) |
| Avg Gross/Trade | d1_full (+11.03) | d1_medium (-8.08) | d1_v451opt (-16.03) |
| Avg Net/Trade | d1_full (-22.56) | d1_medium (-41.37) | d1_v451opt (-48.80) |
| Sharpe | d1_medium (-2.45) | d1_v451opt (-2.84) | d1_full (-4.22) |

**総合判定**: 明確な勝者なし。3 実験とも Gate2 遠く未達。

107# §4.2 の分岐ルール適用:
> 「全セット同等 → 現行 8 特徴で D2 へ（特徴量は非ボトルネックと判定）」

→ **d1_v451opt (8 特徴) をベースに D2 コスト感度分析に進行**

理由:
1. 8 特徴量でも25/73 と大差ない → 追加特徴量の限界費用 > 限界利得
2. v451opt は HOLD 59% で最も HOLD 重視 → 手数料抑制に有利
3. NaN 問題なし、メモリ最小 (919 MB)、最速 (21.4 分)
4. Action mean ≈ 0 で最も偏りが少ない → 安定した基盤

### §4.2 D2 で検証すべきこと

D1 の最大発見は「**手数料支配**」であり、特徴量ではない。D2 では:

1. **コスト感度分析** (107# §4.3 D2-a): transaction_cost = {0.0005, 0.001, 0.0015}
   - 「もし maker 手数料 (0%) で取引できたら PF>1.0 になるか」を判定
   - 現行 0.1% (taker) が唯一のボトルネックか、それとも予測力自体が不足かを切り分け

2. **非対称報酬** (107# §4.3 D2-b): loss × 1.2 ペナルティ
   - 損失取引を重く罰することで TradeWR を引き上げられるか

3. **HOLD 率制御**: threshold の再検討
   - d1_full の HOLD 34.9% に対して v451opt は 59.3%。HOLD 率が高いほど取引頻度が下がり手数料が減る
   - threshold 引上げ (0.70 → 0.80?) による手数料削減効果の検証

---

## §5 ログ・成果物の所在

### §5.1 実験ログ

| ファイル | サイズ | 内容 |
|---------|--------|------|
| `results/phase_c/d1_v451opt_log.txt` | 135 KB | 学習ログ + 結果 JSON |
| `results/phase_c/d1_medium_log.txt` | 137 KB | 学習ログ + 結果 JSON |
| `results/phase_c/d1_full_registry_log.txt` | 163 KB | 学習ログ + 結果 JSON (non-finite 警告含む) |

各ログの末尾に JSON 形式の完全な結果データ (trade_pnls 配列含む) が出力されている。

### §5.2 入力 Parquet

| ファイル | 行数 | カラム数 | 特徴量数 |
|---------|------|---------|----------|
| `data/btc_jpy_1m_v451_optimized_features.parquet` | 1,216,930 | 14 | 8 |
| `data/btc_jpy_1m_medium_features.parquet` | 1,216,930 | 31 | 25 |
| `data/btc_jpy_1m_full_registry_features.parquet` | 1,216,930 | 77 | 73 |

### §5.3 実験定義

| ファイル | 関数/変数 | 内容 |
|---------|----------|------|
| `scripts/v459/run_phase_c.py` | `get_experiment_configs()` | d1_v451opt, d1_medium, d1_full_registry 定義 |
| `scripts/v459/run_phase_c.py` | `BATCHES["d1"]` | D1 バッチ定義 |
| `tests/unit/scripts/test_run_phase_c_d0.py` | `TestD1Experiments` | D1 実験定義の単体テスト (5 テスト) |

### §5.4 コマンド再現

```powershell
# D1-1: v451_optimized (8 特徴)
.\.venv\Scripts\python.exe -u scripts/v459/run_phase_c.py --single-run --experiment d1_v451opt --seed 42

# D1-2: medium (25 特徴)
.\.venv\Scripts\python.exe -u scripts/v459/run_phase_c.py --single-run --experiment d1_medium --seed 42

# D1-3: full_registry (73 特徴)
.\.venv\Scripts\python.exe -u scripts/v459/run_phase_c.py --single-run --experiment d1_full_registry --seed 42
```

---

## §6 実装変更ログ（D0 → D1 間）

107# D0 で実装済みの項目に加え、D1 実験のために以下の追加変更を実施:

| 変更 | ファイル | 内容 |
|------|---------|------|
| 特徴量セット名の修正 | `run_phase_c.py` | curated→medium, minimal→full_registry に変更 (FeatureRegistry ベース) |
| FeatureRegistry/Manager 不整合の回避 | `precompute_*.py` | `--feature-names` 引数追加で Registry から直接特徴量名を指定 |
| Time 特徴量バグ修正 | `ztb/features/` | `dayofweek`/`hour_of_day` が配列に対して scalar 関数を呼んでいた問題を修正 |
| Schema 不一致修正 | `precompute_*.py` | DOW/HourOfDay の int8 vs halffloat 問題を `promote_options="permissive"` で解決 |
| curated 重複修正 | `feature_set_manager.py` | EMACross_Signal_H1/H4 重複エントリを削除 |
| SACTrainer 修正 | `sac_trainer.py` | valid_sets に "curated", "v451", "v454" を追加 |

---

## §7 D2 実験計画（次ステップ）

### §7.1 既に定義済みの D2 実験

`scripts/v459/run_phase_c.py` に以下が定義済み:

| 実験 | transaction_cost | 報酬変更 | 目的 |
|------|-----------------|---------|------|
| d2_cost05 | 0.0005 (maker) | なし | 手数料半額での PF 変化 |
| d2_cost10 | 0.001 (taker) | なし | baseline (D1 と同一) |
| d2_cost15 | 0.0015 (悪条件) | なし | 頑健性確認 |
| d2_asymm12 | 0.001 | loss×1.2 | 非対称報酬 |

### §7.2 D2 で追加検討すべき実験

D1 結果から追加で以下を検討:

1. **threshold 引上げ実験** (0.70 → 0.80/0.85): HOLD 率増加による手数料削減効果
2. **方向予測精度の分離評価**: 手数料控除前の勝率を別途計算するメトリクスの追加
3. **NaN-safe 特徴量セット**: full_registry から NaN を生じない特徴量のみを選定した中間セット

### §7.3 D2 Gate 基準 (107# §4.3 準拠)

D2 通過条件: **PF >= 1.05 かつ Net ROI >= -2%**

現在の最良 PF=0.984 → PF の 0.066 改善が必要。コスト半減 (0.0005) で到達可能か。

---

## §8 リスクと課題

### §8.1 CRITICAL: 全実験がランダム以下

binom_p < 0.001 (全実験) は TradeWR < 50% が統計的に有意であること、すなわち **学習されたポリシーがランダムより劣る** ことを示す。

ただし注意点:
- この TradeWR は **手数料込みの Realized PnL > 0** を基準としている
- 往復 0.2% の手数料があるため、方向予測が正しくても PnL<0 になる取引が多い
- **真の問題は「方向予測精度」か「手数料」か** → D2 cost05 (手数料半額) で切り分け可能

### §8.2 HIGH: 50K steps で多特徴量モデルが学習未収束

full_registry (73 特徴) は observation_space が 73 次元。SAC の actor/critic NN が 50K steps で有効なパターンを学習するには次元が高すぎる可能性。しかし 50K 制約下では step 拡大は禁止 (107# §4.0)。

### §8.3 MEDIUM: Emergency Stop タイミングのバラつき

d1_v451opt は step 8127 で停止、d1_medium は step 21601、d1_full は未停止。emergency stop はポートフォリオリセットを伴うため、stop タイミングの違いが学習経路に影響を与える。

---

## §9 参照ドキュメント

| Doc# | 参照した要点 |
|------|------------|
| 0# | Gate 2 基準 (ROI>5%, PF>1.20, Sharpe>1.0, MaxDD<-15%, WinRate>35%) |
| 104# | Phase C 最良結果 (ROI -4.38%, PF 0.990) |
| 106# | 50K 固定方針、既存特徴量資産活用 |
| 107# | Phase D 改訂版: D0 計測基盤修正 + D1 特徴量比較 + D2 コスト感度 |

---

## §10 Git 状態

- 最新コミット: `4c22a0931` "107# D0完了+D1/D2準備"
- 未コミット変更: Parquet 生成スクリプトの --feature-names 対応、Time 特徴量バグ修正、Schema 修正
- 推奨: D1 結果ログと 108# ドキュメントを含めてコミット
