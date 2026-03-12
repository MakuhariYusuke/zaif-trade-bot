# 361# 351–360 レビュー: ph3 移行妥当性と利益最優先の再整理

**日付**: 2026-03-10  
**対象**: 351#–360#, 現行コード, 関連設定/テスト  
**観点**: 収益性最優先 + ph3 readiness + システム工学 + 市場理論

---

## 総括

351#–360# で進んだこと自体は大きい。

1. **EWMA 系の数学的不整合は 352# で是正された**
2. **buy 側の逆選択防御は 353# で前進した**
3. **ph3 の G2 パイプラインは 358#/359# で一応 end-to-end 化された**
4. **360# で ph2 の現実的な詰まりどころが明文化された**

ただし、**ph3 に進む判断基準そのものがまだ利益問題とズレている**。  
現状の ph3 は「SAC が動く」には近づいたが、「live maker で儲かる方向を学べているか」の保証が弱い。

結論を先に書くと:

- **ph3 の plumbing は概ね前進**
- **しかし ph3 gate はまだ信頼し切れない**
- **最も儲かる可能性が高い次の一手は、ph3 直行ではなく ph2 現行 live 収益源の純化**
- **ph3 は “直接売買エージェント” より “方向/参加度の補助シグナル生成器” として使う方が勝ち筋が太い**

---

## Findings

| # | 重大度 | 対象 | 指摘 | 推奨 |
|---|---|---|---|---|
| 1 | CRITICAL | `scripts/v460/lib/tasks/sac_train.py:117`, `scripts/v460/lib/tasks/sac_train.py:335` | G2 の評価が **完全に in-sample**。訓練に使った同一 `df`・同一 `env` をそのまま checkpoint ROI と最終評価に使っている。これでは G2 PASS でも out-of-sample 収益は証明できない。 | train/val/test の時系列分割を追加し、E1/E3/E4 は validation or holdout で判定する。最低でも train と eval の env を分離する。 |
| 2 | HIGH | `scripts/v460/run_experiment.py:253`, `scripts/v460/run_experiment.py:335`, `scripts/v460/lib/tasks/sac_train.py` | G2 の E2 `ic_seed_std` は実質ダミー。`task_sac_train.py` は `ic_mean` を生成しておらず、`run_experiment.py` は欠損時に 0.0 を入れるため、seed 間標準偏差は常に 0 に寄る。 | E2 は削除するか、RL に意味のある指標へ置換する。例: `roi_seed_std`, `max_drawdown_seed_std`, `trade_count_seed_std`, `turnover_seed_std`。 |
| 3 | HIGH | `configs/v460/experiments/g2_sac_train.yaml:23`, `docs/v460/356_ph3_plan_sac_training.md:227` | ph3 は FeatureRegistry の OHLCV 特徴量だけで進めており、ph2 live の本丸である **maker fill / adverse selection / microstructure** と課題設定がズレている。 | ph3 を「live 直接執行 policy」ではなく「directional/regime prior を出す sidecar」に再定義するか、env に fill-probability / toxicity / spread / quote-age proxy を入れて問題設定を合わせる。 |
| 4 | HIGH | `docs/v460/360_ph2_rpt_fill_test_analysis.md:26`, `docs/v460/360_ph2_rpt_fill_test_analysis.md:55` | 360# は 03-05〜03-09 の **mixed-SHA / pre-post fix 混在集計** で現行 tuning を論じている。`forced_buy_delay` のような削除済み理由も混ざっており、現在の意思決定資料としては汚れている。 | post-348 / post-352 / post-359 の current-SHA に限定して再集計する。`run_id`, `git_sha`, `date_from` 固定で SSOT を作る。 |
| 5 | MEDIUM | `docs/v460/360_ph2_rpt_fill_test_analysis.md:52`, `docs/v460/360_ph2_rpt_fill_test_analysis.md:192` | 360# 内で cancel reason 件数や fill-rate 分解の数値が揺れており、K1 ギャップ分解の信頼性が低い。 | `attempted_fill_rate`, `skip_gate除外率`, `cancel_reason件数` を 1 スクリプトで再生成し、文書へ手貼りしない。 |
| 6 | MEDIUM | `ops/windows/task_scheduler.xml:22`, `docs/v460/360_ph2_rpt_fill_test_analysis.md:275` | 360# の OPS-5 は妥当な方向だが、リポジトリ内の XML は既に `IgnoreNew`。つまり原因は「本番 Task Scheduler が drift している」か「別の scheduler/trigger がある」可能性が高い。 | まず本番スケジューラ実設定を採取する。repo XML 修正より先に、運用環境 drift の有無を確認する。 |
| 7 | MEDIUM | `ztb/risk/sell_dynamic_kill.py:346`, `docs/v460/360_ph2_rpt_fill_test_analysis.md:209` | 353# の EWMA time decay は liveness には効くが、毒性が消えた証拠なしに EWMA を 0 に近づける。これを前提に kill 閾値まで緩めると、防波堤を二重に薄くする。 | TUNE-3 は current-SHA 分析後に限定実験で行う。time decay と threshold 緩和を同時に入れない。 |

---

## 1. ph3 は「進められる」が「そのまま進めるべき」ではない

358#/359# により、ph3 のコード基盤はかなり整った。

- `g2_sac_train.yaml` 作成
- `feature_names` 注入
- multi-seed 実行
- `gross_roi` / convergence 指標
- E2E テスト

この意味で **plumbing は前進** している。

しかし、profit-first で見ると、未解決の本丸は 2 つある。

1. **G2 の評価が信用できるか**
2. **学習している問題が live maker の損益問題と同じか**

この 2 つが弱いまま ph3 に入ると、  
「綺麗に学習して綺麗に gate を通るが、live では稼げない」状態になりやすい。

---

## 2. ph3 最大の盲点: in-sample 評価

`task_sac_train.py` は:

- `scripts/v460/lib/tasks/sac_train.py:99` で parquet 全体を読み
- `scripts/v460/lib/tasks/sac_train.py:118` で単一 `HeavyTradingEnv` を作り
- `scripts/v460/lib/tasks/sac_train.py:127` でその env 上で訓練し
- `scripts/v460/lib/tasks/sac_train.py:132` でその **同じ env** を評価している

さらに convergence も

- `scripts/v460/lib/tasks/sac_train.py:294`

で checkpoint ごとに **同じ env** を `reset()` して計測している。

これは「動作確認」としてはよい。  
しかし「ph3 に進む gate」としては弱い。

理由:

- 時系列 holdout がない
- train/eval leakage がある
- live で欲しいのは汎化性能なのに、見ているのは訓練データ上の再現性能

ここは **CRITICAL**。  
最初に直すべきは、ハイパラではなく **評価設計**。

---

## 3. E2 `ic_seed_std` は今のままだと gate の飾り

`run_experiment.py` は

- `scripts/v460/run_experiment.py:254` で `ic_mean` を読む
- 欠損時は `0.0`

としており、G2 判定は

- `scripts/v460/run_experiment.py:333`

で `ic_mean` の標準偏差を見ている。

しかし `task_sac_train.py` 側には `ic_mean` を生成する処理がない。  
実際 `gross_roi`, `trade_count`, `gross_pnl` はあるが `ic_mean` はない。

つまり E2 は現状、

- 全 seed `ic_mean = 0.0`
- `stdev = 0.0`
- 常に PASS 寄り

になりやすい。

これは gate の 1/4 が実質死んでいる状態。  
359# は E1/E3/E4 を大きく改善したが、**E2 はまだ未完成**。

---

## 4. ph3 が学ぶ問題と、ph2 で負けている問題は一致していない

`g2_sac_train.yaml` は

- `configs/v460/experiments/g2_sac_train.yaml:23`

で OHLCV ベースの 12 FeatureRegistry 特徴量を使う。  
356# でも

- `docs/v460/356_ph3_plan_sac_training.md:229`

として、ph3 は microstructure を切り離す方針を明示している。

これは実装都合としては理解できる。  
ただし、ph2 live の本丸は明らかに

- buy の adverse selection
- spread / timeout
- fill quality
- quote placement

であり、**1 分足 OHLCV だけではここを直接学習できない**。

したがって ph3 の最適な使い方は、現時点では

- live maker の直接 policy

ではなく、

- **方向バイアス**
- **参加/不参加の上位シグナル**
- **regime prior**

を ph2 実行エンジンへ供給する sidecar として使うこと。

ここを誤ると、ph3 に進んでも「学習は綺麗、live は微妙」が続く。

---

## 5. 360# は有益だが、今のままでは current-state tuning の根拠に弱い

360# の価値はある。  
特に以下は良い。

- crash / dual-spawn / K1 未達を一つに並べた
- fill-rate と profitability が別問題だと見え始めている
- ph3 に行けない理由を code blocker と ops blocker に分けた

ただし、その集計窓は

- `docs/v460/360_ph2_rpt_fill_test_analysis.md:26` 03-05〜03-10
- `docs/v460/360_ph2_rpt_fill_test_analysis.md:30` Bot SHA も複数

で、さらに

- `docs/v460/360_ph2_rpt_fill_test_analysis.md:55` `forced_buy_delay`

のような pre-348 理由も含んでいる。

つまり 360# は **「歴史の整理」には良いが、「次の tuning を決める current-state 判断材料」としては汚れが大きい**。

profit-first なら必要なのは:

1. post-348
2. post-352
3. post-359
4. current-SHA

の狭い窓での再分析。

---

## 6. K1 を 60% → 40% に下げる案は、今はまだ早い

360# の

- `docs/v460/360_ph2_rpt_fill_test_analysis.md:291`

の GATE-1 案は理解できる。  
fill rate だけ高くても儲からない、はその通り。

ただし今の段階で単純に K1 を下げると、

- crash 問題
- mixed-SHA 集計
- timeout / spread / current live edge の未分離

を抱えたまま「通したい gate を通す」動きになりやすい。

正しい順序は:

1. current-SHA で K1/K2/PnL/AS を再計測
2. それでも K1 が structurally unreachable か検証
3. その上で **K1 単独ではなく複合 gate** に変える

もし改訂するなら、

- fill rate floor
- PnL 非負
- AS ratio ceiling
- crash-free 72h

の複合がよい。  
単純な 60→40 は雑。

---

## 7. 収益最大化の観点で、今いちばんやるべきこと

### 7.1 ph2 側

最も儲かる可能性が高いのは、ph2 現行 live の磨き込み。

理由:

- 352# では post-fix の局所サンプルで buy/sell ともプラスが見えている
- 353# で buy 側防御も足している
- 360# の 5 日集計は mixed-SHA で current edge を曇らせている

つまり今は「本当にダメ」なのではなく、**今のバージョンの真の実力がまだ見えていない** 可能性が高い。

優先順は以下。

1. current-SHA 限定の fill / cancel / PnL 再集計
2. `buy_ranging` と `timeout` の分離
3. `time decay` と `kill 緩和` を同時に触らず 1 要因ずつ検証
4. `spread_too_narrow` と `timeout` を EV ベースで評価

### 7.2 ph3 側

ph3 は「直接収益化」より「上位シグナル供給」の方が勝ち筋。

具体的には:

1. SAC で directional / regime / participation prior を出す
2. その出力を ph2 の
   - skip_gate threshold shift
   - buy/sell bias
   - volatility / toxicity multiplier
   に渡す
3. quote placement と risk control は既存 ph2 エンジンに任せる

この方が、既存の maker execution 資産を捨てずに済む。

---

## 8. 推奨アクション

### P0: 次に必須

1. ph2 current-SHA 限定で 72h 再集計する  
   `git_sha`, `run_id`, `date_from` を固定。pre-348 を混ぜない。
2. ph3 に train/val/test の時系列分割を入れる  
   同一 env 評価を gate 判定から外す。
3. G2 E2 を無効化ではなく置換する  
   `ic_seed_std` → `roi_seed_std` など RL 向け指標へ。

### P1: 利益に直結

1. `buy_ranging` current-SHA deep dive  
   `wait bucket × VPIN × drift × fill/cancel reason` で切る。
2. `timeout` の根因を current-SHA で分離  
   spread が広過ぎるのか、cancel が早過ぎるのか、板厚が薄いのか。
3. ph3 は sidecar 方針で設計し直す  
   SAC output を live maker の bias/participation に使う。

### P2: その後

1. watchdog の本番設定 drift 確認  
   repo XML と実環境が一致しているか確認。
2. K1 gate 改訂は current-SHA の再計測後  
   先に gate を下げない。
3. microstructure proxy を env に追加  
   ph3 と ph2 の問題設定を近づける。

---

## 最終判断

ph3 に向けたコード基盤は、351#–360# でかなり前進した。  
しかし **「ph3 に進めるか」** と **「ph3 に進むと儲かるか」** は別問題。

現時点で最も危ない盲点は:

- G2 が in-sample であること
- E2 が実質ダミーであること
- ph3 の学習対象が ph2 live の損益問題とズレていること

したがって、profit-first の最適戦略は次の通り。

1. **ph2 current-SHA の本当の edge を先に確定する**
2. **ph3 は direct trader ではなく sidecar alpha として位置付ける**
3. **gate と env を live 問題に合わせてから ph3 に入る**

これが最短で「儲かる」側に寄る。

---

## 検証メモ

- `./.venv/Scripts/python.exe -m pytest tests/unit/v460/test_349_ewma_fixes.py tests/unit/v460/test_356_g2_sac_blockers.py --no-cov`
  - `63 passed`
- `data/btc_jpy_1m_full_registry_features.parquet`
  - exists: `True`
  - rows: `1,216,930`
  - cols: `77`
  - `g2_sac_train.yaml` の 12 特徴量は全て存在確認済み
