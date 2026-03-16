# 423# 422レビュー — 100K Training Forensic Analysis の再点検

**Date**: 2026-03-16  
**Target**: `docs/v460/422_ph4_rpt_100k_forensic_analysis.md`  
**Scope**: 100K SAC 結果の解釈妥当性、設計面、市場理論面、次アクション優先順位

---

## 1. 主要所見

| # | 重要度 | 観点 | 指摘 | 根拠 |
|---|---|---|---|---|
| 1 | HIGH | 実験解釈 | `422#` の **「根因 = val_ratio」** は有力だが、単独原因としては言い切りが強い | `20K=val_ratio 0.02` と `100K=0.20` は確かに交絡しているが、同時に `F6` の checkpoint 選択が `5,000` step 打切りで、100K 側では `243,386` 行 OOS の `2.1%` しか見ていない。`scripts/v460/lib/tasks/sac_train.py:180`, `scripts/v460/lib/tasks/sac_train.py:195`, `scripts/v460/lib/tasks/sac_train.py:530`, `scripts/v460/lib/sac_common.py:296` |
| 2 | HIGH | 設計 | `422#` が特定した **F6 best_model 未使用** は正しいが、`best_model` をそのまま G3 に差し替えるだけでは不十分 | 現状の best checkpoint 自体が `5,000` step OOS proxy で選ばれており、100K では checkpoint OOS ROI が `0.06%–0.15%` と極小。選別ノイズが大きい |
| 3 | HIGH | 方針 | `100K × val_ratio=0.02` は「候補構成」ではなく **診断実験** に留めるべき | そこへ寄せるほど、短い都合の良い tail で通る設定に最適化する危険がある。profit-first でも再現性を落とす |
| 4 | MEDIUM | 市場理論 | 100K の seed 分散爆発は「学習が壊れた」より **レジーム幅が広がって edge の脆弱性が露出した** と見る方が自然 | 20K は `~17日`, 100K は `~169日` の OOS。短い favorable slice で見えていた edge が、長い期間で崩れた可能性が高い |
| 5 | MEDIUM | 打ち手 | `Attention / Residual / PPO / TD3` まで飛ぶのはまだ早い | まず評価ハーネスの公平性を直し、同一プロトコルで `[256,256]`, `[128,128]`, `weight_decay` を比べるべき |
| 6 | MEDIUM | 実運用 | 今の SAC は「直接売買 policy」より、**参加可否 / バイアス / offset 補助の sidecar** として使う方が安全 | ph2 側の fill quality 問題は execution microstructure 依存が強く、SAC 単独で吸収し切れていない |

---

## 2. 422# で妥当だった点

### 2.1 `val_ratio` 交絡の指摘は正しい

これはその通りです。`20K PASS` と `100K FAIL` をそのまま「学習 step 数の差」と読むのは危険で、実際には以下が同時に変わっています。

- OOS 比率: `0.02 -> 0.20`
- OOS 行数: `24,339 -> 243,386`
- OOS 期間: 約 `17日 -> 169日`
- train 行数: `1,192,591 -> 973,544`

したがって、`422#` の **S1: 20K × val_ratio=0.20** は最優先で妥当です。これは残すべきです。

### 2.2 `F6` の 2 つの盲点も正しい

`422#` が追加発見として書いた次の 2 点は、実装照合でもそのまま支持できます。

1. checkpoint OOS 評価が `5,000` step 打切り  
   `scripts/v460/lib/tasks/sac_train.py:530`
2. G3 最終評価が `best_model` ではなく final model を使う  
   `scripts/v460/lib/tasks/sac_train.py:195`

このため、現在の F6 は「早期に良かった checkpoint を保存する」までは出来ていますが、**profit gate に効く形で閉ループになっていません**。

---

## 3. 422# で補正した方がよい点

### 3.1 「20K PASS は偽陽性の可能性が極めて高い」は少し強い

方向性としては理解できますが、現時点で言い切れるのは以下までです。

- `20K PASS` は **楽観的だった可能性が高い**
- ただし、その原因は `val_ratio` だけではなく、**単一 tail holdout** と **F6 の弱い checkpoint 選別** も混ざっている

特に 100K の checkpoint OOS ROI は各 seed でかなり小さく、平均もばらつきも同オーダーです。

- seed42: mean `0.032%`, best `0.113%`
- seed123: mean `0.078%`, best `0.154%`
- seed456: mean `0.049%`, best `0.148%`
- seed789: mean `-0.002%`, best `0.064%`

この状況では、「どの checkpoint が真に良いか」を 5,000 step proxy で決めること自体がかなり不安定です。  
つまり本丸は **val_ratio 問題 + model-selection 問題の複合** です。

### 3.2 `A1: best_model を G3 に使う` は必要だが、それだけでは勝ち筋にならない

ここも方向は正しいです。ただし、今の best_model は「弱い proxy で選んだ best」です。

よって実務上は、次の順が安全です。

1. `final_model` と `best_model` を **同じ full OOS** で両方評価する
2. `best_model` の方が consistently 良いことを確認する
3. その後に G3 の標準評価を `best_model` ベースへ寄せる

先に差し替えるだけだと、`5,000` step に偶然合った checkpoint を本採用する危険が残ります。

### 3.3 `100K × val_ratio=0.02` は昇格させない方がよい

`422#` では `F` 実験も候補に入っていますが、これは **diagnostic only** が妥当です。

理由は単純で、もしこれが PASS しても、結論は

- 「100K が悪い」
ではなく
- 「短い OOS なら通る」

に留まるからです。これは高収益システムの検証としては弱いです。市場理論的にも、短い favorable regime に適合しただけの policy は live で崩れやすいです。

---

## 4. 市場理論から見た解釈

### 4.1 100K 失敗は「alpha 不在」ではなく「alpha のレジーム非頑健性」の可能性が高い

100K でも `seed42` と `seed456` はプラスです。つまり、完全に無情報というよりは、**ある局面では効くが、広い局面では壊れる edge** に見えます。

これは BTC/JPY の microstructure ではよくある形です。

- 短期の order-flow 偏りには反応できる
- しかしボラ・流動性・時間帯・片側優勢が変わるとすぐ逆流する

したがって「もっと長く学習させれば良くなる」より、

- どの regime で参加するか
- どの regime では size を落とすか
- どの regime では execution 系に委ねるか

の方が重要です。

### 4.2 seed 分散は「乱数感度」だけでなく、学習された参加様式の差

100K で worst seed が崩れたのは、単に unlucky seed というより、

- 一方は trend-follow 寄りの参加様式
- 他方は mean-revert 寄りの参加様式

を学習し、そのどちらが長い OOS 期間の支配 regime と噛み合ったか、という説明でもかなり通ります。

この場合、必要なのは単純な平均化より、**regime-conditioned policy selection** です。

---

## 5. 設計面から見た追加提案

### 5.1 最優先は `single tail holdout` からの脱却

今の `train_val_split()` は末尾固定 holdout です。  
`scripts/v460/lib/sac_common.py:296`

これは実装としては簡潔ですが、金融時系列の policy 評価としては脆いです。次のいずれかを入れる価値があります。

1. **Rolling-origin / walk-forward G3**  
   例: `train A -> val B`, `train A+B -> val C`, `train A+B+C -> val D`
2. **Multi-slice OOS checkpoint selection**  
   full OOS が重いなら、先頭 5K 固定ではなく、複数区間から均等抽出する
3. **worst-slice penalty** を含む model selection  
   平均 ROI だけでなく、slice worst PF / Sharpe を見る

### 5.2 checkpoint 選択指標を ROI 単独から上げる

今の F6 は ROI 単独です。microstructure 系では、ROI 単独だとたまたま一方向に走った期間で誤選抜しやすいです。

候補は以下です。

- `score = ROI - λ * max_drawdown`
- `PF floor` を満たさない checkpoint は不採用
- `reward_profit_corr` は補助指標に留める

`reward_profit_corr` が良くても `PF` と `Sharpe` が落ちる例は、今回の 100K でも既に見えています。

### 5.3 architecture 変更は「評価系修正の後」でよい

`[128,128]`, `weight_decay`, `learning_starts` は候補としては妥当です。  
ただし 412# / 413# で既に整理されている通り、ここは **ハーネス修正後の比較** にするのが順序です。

現時点の優先度は以下です。

1. 評価系の公平化
2. 同条件で `[256,256]` vs `[128,128]`
3. `weight_decay` の小レンジ (`0 / 1e-5 / 1e-4`)
4. 必要なら actor/critic 非対称化

`Attention`, `Residual`, `Dropout` は今は後ろで十分です。

### 5.4 ensemble は「重み平均」ではなく「推論時選別」

`Checkpoint Ensemble` をやるなら、`state_dict` 平均より以下です。

1. **best-of-seed**: 同一プロトコルで最も robust な seed を採用
2. **top-K inference ensemble**: 複数 policy の action を推論時に集約
3. **regime switch ensemble**: regime に応じて policy を切替

SAC での重み平均は critic/target の整合が崩れやすく、412# / 413# の整理とも整合しません。

---

## 6. 422# に追加したい「別解」

### 6.1 SAC を direct trader ではなく sidecar に落とす

もし 100K でも direct policy の worst-seed 崩壊が続くなら、SAC の役割を縮めた方が勝ちやすいです。

- 参加可否フィルタ
- side bias (`buy/sell/flat`) 補助
- offset / aggressiveness 補助
- time-of-day / volatility regime の補助判定

この形なら、ph2 で積み上げた execution safety と衝突しにくく、market microstructure の現実にも沿います。

### 6.2 regime-balance training

100K の train window が regime 偏在しているなら、単純な timesteps 増加は逆効果です。  
以下のような再構成も候補です。

- 高ボラ / 低ボラの比率を意識した train slice 構成
- 時間帯を均等化したサンプリング
- `ranging` / `trending` を跨ぐ curriculum

### 6.3 profit-first の seed gate 強化

今の G3 は median と worst を見ていますが、モデル選別段階でも以下を入れる価値があります。

- `worst-seed PF < 0.95` なら候補昇格禁止
- `max_drawdown` が閾値近辺なら hold
- `seed dispersion` が大きすぎる候補は本番前に棄却

今回の 100K は、まさに「平均ではなく dispersion が問題」のケースです。

---

## 7. 推奨アクション順

### P0: すぐやる

1. **20K × val_ratio=0.20** を実行する  
   これが一番情報価値が高いです。
2. **final_model vs best_model の full OOS 再評価** を同条件で出す  
   A1 の採否判断に必要です。
3. **F6 checkpoint 選別を 5,000 固定から改善** する  
   少なくとも multi-slice 化、可能なら full OOS 化。

### P1: その次

1. 同一評価系で `[256,256]` vs `[128,128]` を比較
2. `weight_decay` は `0 / 1e-5 / 1e-4` の小レンジで比較
3. selection score に `PF / MaxDD` を組み込む

### P2: direct SAC がまだ不安定なら

1. SAC を sidecar 化
2. regime-conditioned selection を導入
3. execution policy と alpha policy を分離

### HOLD

- `100K × val_ratio=0.02` を本命候補に昇格
- `Attention / Residual / PPO / TD3` へ即ジャンプ
- checkpoint `state_dict` 平均

---

## 8. 結論

`422#` はかなり良い forensic です。特に **val_ratio 交絡** と **F6/G3 の乖離** を見抜いた点は、そのまま活かしてよいです。  
一方で、profit-first に整理し直すなら、結論は次の形がより正確です。

> **100K FAIL の主因は「val_ratio だけ」ではなく、**  
> **広い OOS でレジーム脆弱性が露出したこと、そして checkpoint 選別/評価設計がそれを適切に扱えていないこと** である。

よって次にやるべきは、step 数や新アルゴリズムへ飛ぶことではなく、

1. `20K × val_ratio=0.20` で交絡を切る  
2. `best_model vs final_model` を full OOS で公平比較する  
3. walk-forward 的な評価へ寄せる

です。ここを先に固めると、以後の `[128,128]`、`weight_decay`、sidecar 化の判断がかなり楽になります。
