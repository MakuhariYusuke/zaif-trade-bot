# 301# 292#-300# 横断レビュー

> **対象**: 292#-300#  
> **観点**: システム工学 / 市場微視構造 / 統計妥当性 / 運用安全性  
> **立場**: 実装者ではなくレビュワーとして、判断を誤らせる論点と今後の優先順位を整理する

---

## 1. Findings

### F1. `none` レジーム除外により F-4 解釈が楽観化している【HIGH】

`ab_judgment.py` の既定では `exclude_regimes=["none"]` となっており、299# の集計もこの前提で進んでいる。一方で 300# 付録では、`none` レジームは実運用上しっかり取引されており、しかも buy/sell とも AS 率が最悪水準で、sell は fill 14.0%, pnl30 -0.80bps とかなり悪い。

つまり現状の F-4 は「実際に出血している母集団」を既定で除外している。これは統計上のノイズ除去ではあるが、運用判断に使う場合は **実損失を薄めた評価** になりうる。

**推奨**:

1. F-4 出力を `excluding_none` と `including_none` の二系統に分離する
2. 両者の差が一定以上なら WATCH 扱いにする
3. `none` を除外したいなら、同時に「`none` 執行そのものを止める」設計に寄せる

### F2. 299# の `sell vs buy` は A/B テストではなく観測比較である【HIGH】

299# は `variant=sell`, `control=buy` として比較しているが、side はランダム割当ではなく、レジーム・残高・forced switch・在庫制約・時間順序の影響を強く受ける。さらに PnL 比較は filled-only であり、300# が指摘している通り生存者バイアスもある。

そのため「有意差なし」は「両 side は同等」とは読めない。せいぜい **現状運用下の観測差は強い証拠になっていない** という意味に留まる。

**推奨**:

1. `sell vs buy` は Gate 根拠ではなく診断レポートとして扱う
2. `attempted / all / filled` の3母集団を常に併記する
3. `alpha trade` と `inventory repair trade` を分離する
4. レジーム内・時間帯内の matched comparison か block bootstrap を優先する

### F3. 295# の「84.8% hot-reload coverage」は集合カバレッジであり、挙動保証ではない【MEDIUM】

295# のテストは主に `_HOT_RELOADABLE_FIELDS` への登録確認で、`maybe_reload()` 実行後に実際のコンポーネントへ変更が反映されるかまでは広く検証していない。`config_hot_reload.py` は prefix ベースの再構築と一部手動同期に依存しているため、**登録済み = 効く** とは限らない。

この点は 295# の価値を否定するものではないが、文書上の「カバレッジ向上」はやや強い表現で、実態は **登録カバレッジ向上** と読むのが正確。

**推奨**:

1. 代表カテゴリごとに E2E hot-reload テストを追加する
2. 少なくとも `forced_buy_delay`, `daily_drawdown`, `FFD`, `time_filter`, `MCB/SAD` は実際に再設定が効くか確認する
3. 将来的には「変更検知」「再構築」「次サイクルの挙動変化」を一気通貫で検証する

### F4. 294# は永久デッドロックを解消したが、期待値問題は未解決【MEDIUM】

`forced_buy_delay_max_consecutive` は liveness 修正として妥当で、永久停止を防ぐ効果はある。ただし、持続的に悪い velocity 条件下では「N 回待って 1 回だけ通す」という周期動作になるため、これは **勝ち筋の追加ではなく安全弁** である。

したがって 294# を「buy 改善」と解釈すると危険で、正しくは「停止不具合の解消」に留まる。以後は、強制通過した buy の損益を独立に追う必要がある。

**推奨**:

1. post-294 の `forced_buy_delay` 突破後 buy を別 KPI で集計する
2. inventory repair 専用モードとして lot 縮小・offset 保守化を組み合わせる
3. 294# の成功判定を「deadlock が消えた」ことと「収益が改善した」ことに分離する

### F5. 298# の統計強化は有用だが、まだ Gate の最終根拠には弱い【MEDIUM】

Mann-Whitney U / Cliff's delta / Holm-Bonferroni の追加は方向として正しい。ただし実装は正規近似ベースで、tie correction や exact/permutation path はない。また Holm 補正も 2 検定単位であり、300# が指摘する regime 横断の多重性や時系列自己相関までは扱っていない。

つまり 298# は **統計を良くした** が、まだ **統計的に十分と言い切れる段階ではない**。

**推奨**:

1. regime 横断では BH か階層補正を追加する
2. tied/rounded データ向けに permutation か tie-corrected path を用意する
3. 非独立性が強い比較には block bootstrap を併用する

### F6. 300# の offset パイプライン仮説は有望だが、定量裏付けがまだ不足【MEDIUM】

300# の「offset が拡大方向に偏って toxic fill only trap を作る」という仮説はかなり筋が良い。ただし 292# で増えた可観測性は `ev_score_pretrade`, `ev_offset_mult_applied`, `decision_path` に留まり、各 offset stage の寄与量までは取れていない。

現段階では 300# は **強い仮説** であって **定量確証** ではない。ここを混同しないほうがよい。

**推奨**:

1. `as_shift`, `vg_mult`, `kyle_mult`, `amihud_mult`, `imbalance_mult`, `loss_mult` を個別記録する
2. 最終 `effective_offset_ratio` だけでなく stage-by-stage で残す
3. その上で `trending_up sell` や `none sell` の offset 生成経路を再分析する

---

## 2. 妥当だった点

292#-296# は、可観測性・型安全・cancel_reason 定数化・例外可視化・hot-reload 拡張など、地味だが必要な基盤整備として妥当だった。特に 292# の `model_used` 誤プロキシ補正と ev 可観測性追加は、289# 系の誤読を止める意味で価値が高い。

298# も、297# の事前調査にあった G-2 誤認を自分で訂正しており、この点は健全。300# は今回の束の中で最も戦略的価値が高く、「統計で差が出ない」と「構造問題がない」を混同していない点が良い。

---

## 3. 優先順位

1. **P0**: F-4 / A/B 出力に `none` 含有版を追加する
2. **P0**: `sell vs buy` を擬似A/B扱いしないことを明文化する
3. **P1**: hot-reload の E2E テストを追加する
4. **P1**: forced buy を `alpha` と `repair` に分離して評価する
5. **P1**: offset stage 寄与量を FillRecord に保存する
6. **P2**: 統計は BH / block bootstrap / paired系へ段階的に拡張する

---

## 4. 総評

292#-300# は、雑に言えば「技術的な止血」と「統計・市場理論の言語化」が並行して進んだ束であり、方向性は悪くない。ただし、判断を誤らせる危険があるのは 299# の読まれ方である。  

この束から引くべき結論は「sell と buy は同じ」ではなく、以下の3点に尽きる。

1. **測れていないものがまだ多い**
2. **`none` と forced 系が評価を歪めている**
3. **構造問題は 300# が示す通り、なお execution 側に残っている**

---

## 5. 検証メモ

今回のレビューでは、関連コード (`ab_judgment.py`, `fill_cycle_executor.py`, `skip_gate_evaluator.py`, `order_monitor.py`, `fill_loop_orchestrator.py`, `config_hot_reload.py`) を照合した。  
また、対象ユニットテスト `tests/unit/v460/test_292_observability.py` と `tests/unit/v460/test_160_ab_judgment.py` は、仮想環境で `--no-cov` 実行により **125 passed** を確認した。
