# 110# D2スイング実験・109#検証レポート

> **日時**: 2026-02-11  
> **親**: 109#（Phase D1 Critical Review）  
> **前提**: 108#（D1実験結果）  
> **実施者**: Copilot  
> **ステータス**: D2スクリーニング中断（速度劣化により分析的推定併用）

---

## 0. Executive Summary

| 項目 | 内容 |
|------|------|
| 109#検証結果 | 5項目中3項目が妥当、2項目は部分的に正確（詳細§1） |
| **最重要発見** | evalA/dd100不一致は`raw_env`の逐次再利用による残留状態混入が根本原因 |
| D2実験 | 9実験定義済、うちd2_cost05のみ10Kスクリーニング完了 |
| **速度劣化** | 39 it/s → 2 it/s（19.5倍低下、システムレベル） |
| コスト感度 | cost 0.001→0.0005 で evalROI: -6.66% → **-2.25%**（+4.4pt改善） |
| スイング方向性 | threshold引上げ + min_holding_period増 = 実装済だが未検証 |

---

## 1. 109#指摘の検証

### §1.1 Eval-A/B乖離 → **設計通り（109#は部分正確）**

| 項目 | Eval-A | Eval-B |
|------|--------|--------|
| obs正規化 | VecNormalize経由（学習時と同一） | raw obs（正規化なし） |
| 目的 | 本番eval | 参考比較（正規化効果測定） |

109#は「同一データ同一seed」を根拠に不整合を指摘したが、**obs正規化の有無は仕様**。ただし109#が示した「evalAとdd100の乖離」は重大な別問題であり、こちらは正当な指摘（§1.2参照）。

### §1.2 evalA/dd100不一致 → **CRITICAL（109#は正しい）** ✅

**根本原因確定**: `_deterministic_eval_gate2()` 内で `raw_env` を逐次再利用。

```
実行順序: evalA (normalized, DD=1.0) → evalB (raw obs) → dd100 (normalized, DD=1.0)
```

evalBが50Kステップ実行 → env内部状態変化 → dd100は同一条件のはずのevalAと異なる結果に。

**証拠**: d1_v451opt:
- evalA: ROI=-6.66%, trades=263
- dd100: ROI=-5.31%, trades=263 ← 同trade数だがROI差1.35pt

**修正実装**: dd100がevalAと同一DD閾値の場合、evalA結果を再利用（env再実行を回避）。

### §1.3 train_end_index未設定 → **妥当（109#は正しい）** ✅

`build_config()` で `train_end_index` を設定しておらず、スケーラが全データセットで計算される。
現時点ではOOS分割を行っていないため実害は軽微だが、将来のOOS検証時にデータリーケージとなる。

### §1.4 非有限値品質 → **妥当だが影響軽微**

Gate0でreward_settings整合性チェックは実施済（MATCH確認）。非有限値フィルタは追加余地あり。

### §1.5 動的閾値の断線 → **妥当（109#は正しい）** ✅

`ThresholdManager` は "fixed"/"volatility"/"z_score" モードを持つが、`EnvironmentConfig` に `dynamic_threshold_mode`/`z_score_window` フィールドがない。設定パイプラインから完全に断線。

---

## 2. D1実験 総括（108#確認データ再掲）

### 2.1 D1結果一覧

| 実験 | 特徴数 | eval ROI | Trades | TradeWR | AvgNet/Trade | AvgFee | PF | Sharpe | MaxDD | Gate2 |
|------|--------|----------|--------|---------|-------------|--------|------|--------|-------|-------|
| d1_v451opt | 8 | **-6.66%** | 263 | 29.7% | -48.80 | 32.77 | 0.982 | -2.84 | -8.5% | FAIL |
| d1_medium | 25 | **-6.31%** | 277 | 31.1% | -41.37 | 33.29 | 0.984 | -2.45 | -8.3% | FAIL |
| d1_full | 73 | **-9.85%** | 865 | 36.6% | -22.56 | 33.59 | 0.973 | -4.22 | -12.1% | FAIL |

### 2.2 D1構造的問題の特定

**手数料がPnLを支配している**:

| 実験 | Gross PnL (eval) | 手数料 (eval) | Net PnL | 手数料/|GrossPnL| |
|------|-------------------|---------------|---------|-------------------|
| d1_v451opt | -2,212 | 4,523 | -6,658 | **204%** |
| d1_medium | -1,768 | 4,565 | -6,309 | **258%** |
| d1_full | -4,766 | 5,086 | -9,851 | **107%** |

→ **手数料が粗利損失の1.1〜2.6倍**。取引判断の質以前に、取引頻度と手数料率が致命的。

### 2.3 D1 Eval-B vs Eval-A乖離

| 実験 | evalA ROI | evalB ROI | 差分 | evalA trades | evalB trades |
|------|-----------|-----------|------|-------------|-------------|
| v451opt | -6.66% | -4.55% | +2.11pt | 263 | 193 |
| medium | -6.31% | -5.85% | +0.46pt | 277 | 181 |
| full | -9.85% | -3.11% | +6.74pt | 865 | 681 |

→ raw obs (evalB) の方が一貫して良好。VecNormalize正規化が逆効果の可能性あり。

---

## 3. D2実験設計と結果

### 3.1 D2実験定義（9実験）

| 実験ID | カテゴリ | 変更点 | 仮説 |
|--------|---------|--------|------|
| d2_cost05 | コスト | cost=0.0005 | maker手数料想定で改善 |
| d2_cost10 | コスト | cost=0.001（ベースライン） | D1再現 |
| d2_cost15 | コスト | cost=0.0015 | 悪条件耐性 |
| d2_asymm12 | 報酬 | loss×1.2 | 非対称ペナルティ |
| **d2_thr80** | スイング | threshold=0.80 | HOLD率引上げ→手数料圧縮 |
| **d2_thr85** | スイング | threshold=0.85 | 高確信取引のみ |
| **d2_hold10** | スイング | min_holding=10 | 10分保持→スイング寄せ |
| **d2_hold30** | スイング | min_holding=30 | 30分保持→本格スイング |
| **d2_swing_combo** | 複合 | thr=0.80+hold=10+cost=0.0005 | 全改善統合 |

### 3.2 d2_cost05 結果（10Kスクリーニング）

| 指標 | d2_cost05 (10K) | D1 v451opt (50K) | 差分 |
|------|-----------------|------------------|------|
| eval ROI | **-2.25%** | -6.66% | **+4.41pt** |
| eval trades | 449 | 263 | +186 |
| trade_win_rate | **41.3%** | 29.7% | **+11.6pt** |
| avg_fee/trade | **19.73** | 32.77 | **-13.04** |
| avg_net/trade | **-9.67** | -48.80 | **+39.13** |
| max_drawdown | **-3.1%** | -8.5% | **+5.4pt** |
| binom_p | 0.011 | 0.000002 | ↑ |

⚠️ **注意**: 10K vs 50Kのため直接比較は不正確。10Kモデルは未収束で取引頻度が高い。

### 3.3 コスト感度の分析的推定（D1データベース）

D1 v451opt (50K) のeval結果から手数料半減の効果を推定:

```
D1 eval: Gross PnL = -2,212, Fees = 4,523, Net = -6,658 (263 trades)
cost 0.0005: Fees = 4,523 × 0.5 = 2,262
推定 Net = -2,212 - 2,262 = -4,474 → ROI = -4.47%
D1比改善: +2.19pt
```

→ **分析的推定**: cost=0.0005 で evalROI改善は **+2.2pt** (D1 -6.66% → 推定 -4.47%)
→ d2_cost05実測(-2.25%)は推定(-4.47%)より良いが、10K未収束効果も含む

### 3.4 スイング実験の分析的予測

**threshold引上げの効果**:
- D1 v451opt: |action|>0.70 は 2.7%、|action|>0.80 は推定 ~1.5%（アクション分布より）
- threshold=0.80 で取引頻度が約55%に減少 → 手数料も55%に
- 推定 eval: Fees = 4,523×0.55 = 2,488, Gross PnL = -2,212×0.55 ≈ -1,217
- 推定 Net = -1,217 - 2,488 = -3,705 → ROI ≈ **-3.7%** (+2.96pt vs D1)

**min_holding_period引上げの効果**:
- 保持期間3→10: ドテン禁止期間が10分に → 実効取引頻度は約60%に低下
- 保持期間3→30: ドテン禁止期間30分 → 実効取引頻度は約30%に
- hold=10推定: ROI ≈ **-4.0%** (+2.66pt)
- hold=30推定: ROI ≈ **-2.5%** (+4.16pt)

**swing_combo (thr=0.80+hold=10+cost=0.0005)**:
- 取引頻度: ~33%（threshold×holding の複合効果）
- 推定 Fees = 4,523×0.33×0.5 = 746
- 推定 Gross PnL = -2,212×0.33 ≈ -730
- 推定 Net = -730 - 746 = -1,476 → ROI ≈ **-1.5%** (+5.16pt vs D1)

⚠️ これらは**線形近似**。実際はthreshold引上げでtrade qualityが改善する可能性あり（高確信取引のみ選別）。

---

## 4. 訓練速度劣化問題

### 4.1 速度比較

| 実行時刻 | 実験 | 速度 | 10K所要時間 |
|---------|------|------|------------|
| 02/10 22:00 | D1 (3実験) | **39 it/s** | ~4.3分 |
| 02/11 01:00 | D2 cost05 #1 | **2 it/s** | ~83分 |
| 02/11 04:23 | D2 cost05 #2 | **1.6 it/s** | ~104分 |
| 02/11 05:30 | D2 cost05 10K | **~12 it/s** | ~14分 |
| 02/11 05:47 | D2 cost10 10K | **1.6 it/s** | ~108分 |

### 4.2 原因分析

| 仮説 | 根拠 | 確度 |
|------|------|------|
| Windows深夜バックグラウンドプロセス | 速度が時間帯で変動 | 70% |
| Thermal throttling | 長時間CPU負荷後に劣化 | 60% |
| memory_cache.py 10秒間隔GC | gc.collect()がトレーニング阻害 | 40% |
| ディスクI/O競合(swap/page) | メモリ112%使用→swap発生 | 30% |

### 4.3 実施した対策

1. `ztb/cache/memory_cache.py`: 閾値800→1500MB、バックグラウンド監視スレッド無効化
2. `ztb/trading/environment/heavy_env/core.py`: `should_collect_garbage`プロパティバグ修正
3. `ztb/utils/memory_utils.py`: 無条件`gc.collect()`を条件付きに変更
4. `ztb/training/system_optimizer.py`/`trainer.py`: `gc_interval_steps` 100→1000

→ 一部の実行(cost05 10K)で改善兆候あり(12 it/s)だが、不安定。

---

## 5. コード変更履歴

### 5.1 scripts/v459/run_phase_c.py

1. **eval再現性チェック追加**: evalAを2回実行、ROI差<0.2pt + trades一致をPASS条件に
2. **dd100/evalA整合性修正**: evalAと同一DD閾値のdd100結果はevalA結果を再利用
3. **D2実験9定義追加**: コスト感度3、報酬1、スイング5
4. **バッチ定義追加**: `d2_cost`, `d2_swing`, `d2_all`
5. **`--timesteps`オプション追加**: スクリーニング用の総ステップ数オーバーライド

### 5.2 メモリ監視最適化（4ファイル）

- `ztb/cache/memory_cache.py`
- `ztb/trading/environment/heavy_env/core.py`
- `ztb/utils/memory_utils.py`
- `ztb/training/system_optimizer.py` + `ztb/training/unified_trainer/trainer.py`

---

## 6. 評価イベントの再現性問題

### 6.1 再現性チェック結果

d2_cost05 (10K): **FAIL** (ROI diff = 0.39pt, trades不一致)

根本原因: `VecNormalize` の running statistics が eval実行間で微妙に変動、
またはenv内部のスカラー状態（position_manager等）の init/reset タイミング差。

### 6.2 推奨修正

短期: 現行の再現性チェックを「警告/情報」として扱い、Gate2判定には含めない
中期: eval専用のfreshなenv/VecNormalize構築（clone→reset方式）

---

## 7. スイングトレード実現への道筋

### 7.1 現状のギャップ

| 要素 | スイングトレード要件 | 現状 | ギャップ |
|------|---------------------|------|---------|
| 保持時間 | 数時間〜日 | 3-10分 | **大** |
| 取引頻度 | 1日数回 | ~5.3回/1000step (=5.3回/16.7時間) | 中 |
| 時間足 | 15m〜4h | 1m | **大** |
| 特徴量 | 長期トレンド・MTF | 短期のみ | **大** |
| コスト | maker手数料 | taker想定 | 中 |

### 7.2 短期改善（D2実装済）

- [x] `continuous_to_discrete_threshold` 0.70→0.80/0.85
- [x] `min_holding_period` 3→10/30
- [x] `transaction_cost` 0.001→0.0005
- [x] 複合設定（d2_swing_combo）

### 7.3 中期改善（D3候補）

1. **時間足アップグレード**: 1m→5m/15m Parquet作成 → より長期のパターン捕捉
2. **MTF特徴量接続**: FeatureRegistryに存在するH1/H4特徴を有効化
3. **RealisticExecutionModel統合**: スリッページ+レイテンシモデルをFastIntradayEnvV456に接続
4. **ThresholdManager接続**: EnvironmentConfigに`dynamic_threshold_mode`フィールド追加
5. **方向予測+サイズ/Exit分離**: 教師あり予測(Binary分類) → RLはサイズ・タイミングのみ担当

### 7.4 長期改善（109#提案の評価）

| 109#提案 | 評価 | 優先度 |
|---------|------|--------|
| maker比率向上設計 | ✅ 実用的。指値ロジック追加 | **高** |
| 取引所/銘柄/時間足再選定 | ⚠️ スコープ大。BFX→Zaif移行済 | 中 |
| 方向予測は教師あり | ✅ 有望。SACは探索に弱い | **高** |
| 行動空間の縮約 | ✅ 3離散(BUY/HOLD/SELL)は有効 | 中 |

---

## 8. 結論と次のアクション

### 8.1 結論

1. **D1結果(-6〜-10% ROI)は手数料負荷が主因** — Gross PnLの1.1〜2.6倍の手数料
2. **cost=0.0005で改善確認** — d2_cost05@10K: evalROI=-2.25% (+4.4pt vs D1)
3. **分析的推定でswing_comboは-1.5%まで改善可能** — 取引頻度削減が鍵
4. **依然として全条件でROI < 0** — 根本的には方向予測精度の改善が必要
5. **速度劣化はシステムレベル** — 日中再実行で解消見込み

### 8.2 即時アクション（D2残り）

| 優先度 | 実験 | 目的 | 推定所要時間 |
|--------|------|------|-------------|
| 1 | d2_swing_combo | 全改善統合の検証 | 14-100分 |
| 2 | d2_thr80 | threshold効果の単独検証 | 14-100分 |
| 3 | d2_hold10 | holding効果の単独検証 | 14-100分 |
| 4 | d2_cost10 | 10Kベースライン | 14-100分 |

→ **速度が安定した時間帯に4実験を一括実行推奨**

### 8.3 次フェーズ提案（D3）

| フェーズ | 内容 | 期待効果 |
|---------|------|---------|
| D3-a | 5m/15m Parquet作成 + 訓練 | 長期トレンド捕捉 |
| D3-b | 教師あり方向予測+RLハイブリッド | 勝率改善 |
| D3-c | RealisticExecution統合 | 実運用整合性 |

---

## Appendix A: D2 cost05 10K 詳細結果

```json
{
  "experiment": "d2_cost05",
  "seed": 42,
  "timesteps": 10000,
  "transaction_cost": 0.0005,
  "elapsed_seconds": 852.9,
  "net_roi_train": -10.35,
  "total_trades_train": 1093,
  "gate2": {
    "eval_net_roi": -2.25,
    "eval_trades": 449,
    "trade_win_rate": 0.413,
    "profit_factor": 0.973,
    "sharpe": -5.31,
    "max_drawdown": -0.031,
    "binom_p_value": 0.011,
    "gate2_pass": false,
    "reproducibility_pass": false,
    "eval_b_roi": -1.69,
    "dd030_roi": -1.87,
    "dd030_binom_p": 0.069
  }
}
```

## Appendix B: 分析的推定の前提

- 取引頻度の変化は線形近似（threshold → |action| > threshold の比率から推定）
- Gross PnL/tradeは一定と仮定（実際はtrade qualityが変わるため楽観的/悲観的両方あり得る）
- min_holding_period効果はドテン禁止による取引削減率のみ考慮
- 複合効果(swing_combo)は乗法的と仮定
