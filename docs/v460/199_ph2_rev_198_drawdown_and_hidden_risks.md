# 199# 198レビュー: ドローダウン分析の検証 + 隠れた再発要因

> **対象**: `193#`〜`198#` の実装・事後分析、および `results/v460/fill_test/logs/fill_test.log` / `results/v460/fill_test/fill_records_20260301.jsonl` の実ログ。  
> **結論**: `193#`〜`197#` の「hard gate を soft offset へ落とす」方向は複雑性削減としては正しい。だが、`198#` の朝セッション損失は、**reprice / state 管理 / 速度指標の不整合 / 最小ロット起因の安全弁不発** が重なって再発している。  
> **要点**: 今回の負けは単一バグではない。`198#` で挙がっている `stale_order reprice` が主因だが、それを悪化させる周辺設計がまだ残っている。

---

## 1. 198# の主要主張は妥当

`2026-03-01 09:04–10:07 JST` の実ログ確認では、`198#` の骨子は概ね正しい。

- 12 fills 後に `daily_drawdown_guard` が hard halt
- sell 側の損失寄与が大きい
- reprice 付き約定が特に悪い
- `soft lot reduction` が `0.0010 -> 0.0010` で実質無効
- 09:31 以降、下落が続くのに `trending_down` へ遷移していない

特に `10:00:30–10:02:34` の Cycle 5229 は、ログ上でも以下の通り。

1. sell 発注
2. 下落方向への drift を検知
3. `stale_order` が cancel-reprice
4. 約 17bps 下方へ価格追随
5. 約定後 `-23.32bps`

この点は `198#` の診断通りで、最優先論点のまま維持してよい。

---

## 2. 198# に追加すべき重要事項

## 2.1 [HIGH] HALT 中に `fill_test_state.json` が更新されず、状態監視が壊れる

実ファイル確認:

- `results/v460/fill_test/fill_test_state.json` 更新時刻: `2026-03-01 02:07:41 JST`
- `results/v460/fill_test/fill_records_20260301.jsonl` 更新時刻: `2026-03-01 15:10:00 JST`

同一 run (`run_id=1772296493_9ffe8a80`) にもかかわらず、state は朝セッション前の時点で止まっている。  
これは `scripts/v460/lib/fill_loop_orchestrator.py` の HALT 分岐が、

- `daily_drawdown_halt` レコード追加
- batch flush
- sleep

のみを行い、**state 永続化を行っていない**ため。

### 影響

- `cycle_count=5200` のまま止まり、実際の `5231` まで進行した事実が state に反映されない
- `saved_at` が古いままなので、外部監視からは「止まったのか、HALT 中なのか」が判別しづらい
- 復旧判断や再起動判断を誤る

### 判断

`198#` は取引損益を正しく見ているが、**運用監視の観測面の破綻**は未記載。  
これは損益そのものではないが、長時間 HALT 運用では無視できない。

## 2.2 [HIGH] `daily_drawdown_halt` レコードが日次ファイルを過半占有し、分析ノイズになっている

`results/v460/fill_test/fill_records_20260301.jsonl` 実測:

| 区分 | 件数 |
|---|---:|
| 総レコード | 57 |
| `daily_drawdown_halt` | 31 |
| 実約定 | 12 |
| その他 (`skip_gate`, `timeout`, `spread_too_narrow`) | 14 |

つまり、この日の JSONL は **半分以上が「取引していない HALT 監査レコード」**。

### 問題

- 生ログとしては有益だが、日次集計では「実取引日」ではなく「HALT 待機ログ」に埋もれる
- 分析スクリプトが `side="none"` / `order_quantity=0.0` を常に除外する保証がない限り、件数系指標を歪める
- 将来の学習用抽出や運用 KPI で、`fill_rate` や cancel reason 構成比が過度に悪化して見える

### 判断

`daily_drawdown_halt` 自体は必要だが、**同じ fill_records 系ファイルへ高頻度で混在させる設計は見直し対象**。  
監査用途なら別ファイル or 別カテゴリへ分離した方が良い。

## 2.3 [HIGH] `VG` と `velocity_offset` が、同じ「速度」名なのに符号が逆転している

朝セッションで以下が実際に発生している。

### 2026-03-01 09:53:58 (buy)

- `maker_price` の VG ログ: `velocity=17.0bps`
- `skip_gate` の velocity ログ: `buy velocity=-8.85bps`

### 2026-03-01 10:00:30 (sell)

- `maker_price` の VG ログ: `velocity=-18.4bps`
- `skip_gate` の velocity ログ: `sell velocity=9.20bps`

### 含意

同一 cycle 内で、`scripts/v460/lib/maker_price.py` と `scripts/v460/lib/skip_gate_evaluator.py` が  
**逆符号の velocity を見て別々に offset を調整している**。

これは次のどちらか。

1. 定義が本当に異なる（mid-based と trade-based など）
2. 符号規約が揃っていない

いずれにせよ、現状の命名では「同じ velocity を見ている」ように読めてしまう。  
その結果、**保守化の根拠が二重化しているのに、内部では相互に矛盾し得る**。

### 判断

これは `198#` 未記載だが、実運用上かなり重要。  
`soft offset` 群が増えた今、**同じ概念名で違う信号を重ねるのは危険**。

## 2.4 [HIGH] `193#` の `ev_as_offset` は deadlock 解消には効くが、「負EVの取引禁止帯」を薄くしすぎた

`193#` で `ev_weighted` は hard gate から offset 修飾子へ降格した。  
これは `191#` / `192#` の複雑性問題に対する正しい応答だった。

ただし副作用として、**軽度〜中度の負EVでも、emergency 閾値に達しない限り原則は発注される**。

朝セッションでも、negative 側の EV 補正を受けつつ約定した負けトレードが複数ある。

- 09:43 sell: `ev_score=+1.687` でも `-6.87bps`
- 09:56 buy: `ev_score=-2.514` のまま約定して `-8.50bps`
- 10:06 buy: `ev_score=-1.485` のまま約定して `-5.91bps`

### 問題

- 旧方式は「止まりすぎ」だった
- 新方式は「通しすぎ」に寄りやすい
- しかも `0.001 BTC` が最小ロットのため、lot 半減で吸収できない

### 判断

`193#` は方向として正しいが、現状は  
**hard skip か fully tradable かの二極から、「全部 tradable 寄り」に振れすぎている**。

中間に、

- 軽度負EV: offset のみ
- 中度負EV: offset + 待機延長 or cancel-only
- 重度負EV: skip

という 3 段階が必要。

## 2.5 [MEDIUM] 主 Gate の「止める先」が朝セッションの実損益と噛み合っていない

`09:04–10:07` の `skip_gate` ログ上の hard skip は 6 件:

- buy: 5 件
- sell: 1 件

一方、`198#` 集計では損失の **74% が sell**。

### 含意

現時点の primary gate は、

- 負けた sell を十分に止められていない
- 逆に buy を多めに止めている

という **方向ミスアライン** を起こしている可能性が高い。

これは「sell が悪い」という単純な話ではなく、  
**実損失の出ている方向と、最も強く制御されている方向がズレている**ということ。

## 2.6 [MEDIUM] `postonly_guard` は「保護消失」だけでなく、「価格決定の責務逆転」を起こしている

`198#` は Cycle 5216 を「offset 保護消失」と整理している。これは正しい。  
ただし、問題はそれだけではない。

`scripts/v460/lib/fill_cycle_executor.py` の `postonly_guard` は、

- 計算済みの `order_price` が crossing した場合
- buy なら `best_bid`
- sell なら `best_ask`

へ **即座にスナップ** している。

つまり最終価格は、

- `maker_price`
- `ev_offset`
- `velocity_offset`
- `trend_offset`

で積み上げた結果ではなく、**最後の数行の板スナップで再定義される**。

### 含意

これは「微調整が消える」というより、

> 価格決定の主体が offset パイプラインではなく `postonly_guard` に乗っ取られる

という構造問題。

対処は単に「postonly を弱める」ではなく、

1. 先に crossing を検出
2. 最新 best bid/ask を起点に offset を再計算
3. その結果として post-only を満たす価格を作る

であるべき。

## 2.7 [MEDIUM] `low_vol_boost` は条件付き防御ではなく、ほぼ定数化している

朝セッションでは `vol_ratio=0.155–0.265` で、全サイクルが `0.75` を大きく下回っている。  
結果として `low_vol_boost x1.40` が常時発動。

これは `198#` の指摘通りだが、重要なのは「倍率が強い」こと以上に、

> 条件分岐のつもりで入れた guard が、実態としては baseline になっている

点。

この状態では「boost」ではなく実質の新しい base offset なので、  
以後の EV / velocity / trend 調整の基準もズレる。

---

## 3. 193#〜197# で評価できる点

今回の実装群には、方向として正しい改善もある。

### 3.1 良い点

1. `ev_weighted` の hard gate 廃止  
2. `velocity_skip` の soft 化  
3. `CycleGateAggregator` による hard blocker の整理  
4. `trending_sell` の bypass 条件削減

これは `191#` / `192#` で指摘した「distributed ownership」を確かに改善している。

### 3.2 ただし限界

今回の損失は、複雑性削減自体が間違っていたのではなく、

- 価格更新 (`reprice`)
- 状態観測 (`state persistence`)
- 指標意味論 (`velocity`)
- 最小ロット起因の risk control 不発

が未解消のまま残ったために発生している。

つまり、**構造整理は前進したが、損失発生点が別レイヤーに残っていた**ということ。

---

## 4. 次の優先順位

## 4.1 P0（最優先）

1. `scripts/v460/lib/order_monitor.py`  
   sell で不利方向の drift 時は `cancel-only` を追加し、reprice を止める
2. `scripts/v460/lib/fill_loop_orchestrator.py`  
   HALT 分岐でも state を定期保存する
3. `daily_drawdown_halt`  
   監査レコードを fill_records 本体から分離、または別 JSONL に逃がす

## 4.2 P1（次点）

1. `scripts/v460/lib/maker_price.py` と `scripts/v460/lib/skip_gate_evaluator.py`  
   velocity の命名と符号規約を分離・統一する
2. `ev_as_offset`  
   「中度負EVは soft veto」にする 3 段階化を導入する
3. `postonly_guard`  
   best bid/ask へスナップではなく、最新板を基準に offset 再計算へ変更する

## 4.3 P2（その後）

1. `low_vol_threshold` を実測分布ベースで再較正する
2. 最小ロット前提で効く安全弁（lot 半減ではなく skip / cooldown）へ切り替える
3. 朝セッションの sell 専用に、`PnL wait` と trend 判定を再検証する

---

## 5. 最終評価

`198#` は事後分析として有効で、主犯の特定も大きく外していない。  
ただし、再発防止という観点では次の 3 点を追加で押さえる必要がある。

1. **HALT 中の state 観測破綻**  
2. **速度指標の二重化と意味衝突**  
3. **最小ロット環境で、soft risk control が実質的に効いていないこと**

今回の損失は「複雑すぎたから負けた」というより、

> 複雑性を減らしたあとに残った、実運用層の穴で負けている

という整理が正確。

従って次の一手は、Guard の追加ではなく、**reprice / state / signal semantics の整備** を優先すべき。

---

## 6. 追記: 199# に対するセカンドオピニオンと「市場理論」に基づく抜本的批判 (Gemini 3.1 Pro)

### 6.1 総評: Codexの慧眼と「システム認知の分裂」

Codexが指摘した「HALT時の状態監視破綻」「監査ログによるデータ汚染」「\postonly_guard\による価格ハイジャック」は、198#の分析が見落としていた**「運用層の致命的バグ」**を見事に突き止めている。全面的に支持する。
直接ログやコードを検証したところ、特にVG (\maker_price.py\) とSkipGate (\skip_gate_evaluator.py\) で「同じVelocityの考え方でありながら、計算元と符号が逆転して喧嘩している」状況が確認された。これは**システムの認知が分裂**している証拠である。酒田五法に**「相場は相場に聞け」**という絶対原則があるが、いまのBotは相場の声を聞く「耳」を二つ持ち、都合よく解釈して逆選択に突っ込んでいる。直ちに一つの強牢な指標に統合せよ。

### 6.2 MM理論から見た「Reprice (Chase)」の狂気と自殺行為

マーケット・メーカー（MM）理論（Avellaneda-Stoikovモデル等）の根幹は、Inventory（在庫）リスクとAdverse Selection（逆選択）リスクのコントロールである。
相場が自分にとって不利な方向（例: Sell指値に対して価格が下落）に「逃げて」いるとき、\is_drifting_away\ をトリガーに指値を下げてまで追随（Reprice）する現行ロジックは、MM理論において**「自ら喜んでババ（逆選択）を掴みに行く自殺行為」「落ちてくるナイフを掴むな」**の典型である。Taker（モメンタム）なら追えばいいが、Makerが逃げる価格を追えば、最も不利な「底」で約定して死ぬだけだ。
Codexの言う通り、順方向のChaseのみ残し、不利方向のDriftは**直ちにCancel-only（撤退）**に切り替えるべきだ。

### 6.3 一目均衡表の「時間論」と postonly_guard の短絡性

一目均衡表の真髄は「価格と時間のバランス」にある。時間を味方につけるのがMakerの最大の優位性である。
しかし、現行の \postonly_guard\ は、193#等で精緻に計算し積み上げた「身を守るためのOffset（雲の厚さ）」を、発注の最後の最後で「板が交差するから」という理由だけで、**一瞬にして \best_ask\ や \best_bid\ にスナップ（短絡）させて台無しにしている**。
これは時間を放棄し、最前線に丸腰でテレポートする愚行である。「雲（強固なレジスタンス）」に阻まれて計算上の指値が板を突き抜けるなら、スナップして発注するのではなく、**一旦発注を見送る（「休むも相場」）**のが正解である。

### 6.4 最小ロット (0.001) の呪縛と「建玉の整理」

「Soft Drawdown リミットに触れたが、最小ロットのため半減（切り下げ）処理が作動しなかった」という事態は、資金管理（Money Management）の破綻を意味する。
建玉（ロット）がこれ以上減らせないという物理的限界に直面したなら、**時間のインターバルを伸ばす**か、あるいは直近の損失の74%を占めている**不調なサイド（今回はSell）を一定時間「完全封鎖」**するしか生き残る道はない（酒田五法の「三兵」における「見送り」）。
Soft Limitに触れた際のフェイルセーフとして、「ロットが下げられないなら、Cycleインターバルを強制的に3倍にする」や「ペナルティホールド」など、実効性のある「建玉の整理」代替案を直ちに実装せよ。

### 6.5 結論と即時アクション

小手SkipGateや計算式のパラメータ調整ではなく、以下の**「致命的な出血点の縫合」**を最優先で行うべきだ。

1. **不利方向へのChase（Reprice）の完全禁止**（MM理論に基づく逆選択特攻の阻止）
2. **\postonly_guard\ のスナップ廃止**（Offsetを加味して交差するなら発注キャンセル）
3. **Velocity指標の単一情報源（SSOT）化**（認知の分裂の解消）
4. **Soft Drawdown時の代替制限策（インターバルあるいはサイド制限）の実装**

これらを行わずして稼働を再開するのは、同じ穴に落ちるだけの無謀な試みであり、我々の大義である**「短期間での高収益性」**からは程遠い。まずはこの穴を塞ぐことに専念せよ。
