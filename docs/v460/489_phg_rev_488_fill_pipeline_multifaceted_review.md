# 489# 488# レビュー: Fill Test パイプライン論点の補強・補正・追加盲点

> 種別: review
> 対象: 488# `docs/v460/488_phg_fill_pipeline_deep_review.md`
> 日付: 2026-03-19

---

## 0. 総評

488# は、現状の fill test を「収益・可観測性・例外安全性・設定整合性」の観点から広く棚卸しした文書としては有益である。  
ただし、そのまま「根本原因の確定版」として読むには強すぎる断定や、現行コードでは既に古くなった指摘も混ざっている。

今回の整理は次の 5 点である。

1. **危機感そのものは正しい**  
   直近 `2026-03-10` 〜 `2026-03-18` の `fill_records` 集計でも  
   `rows=4120`, `fills=997`, `fill_rate=24.2%`, `cum_pnl30=-232.6bps`, `PF=0.900`  
   と厳しい。

2. **488# の P0/P1 候補には当たりがある**  
   特に `orchestrator_mid_cycle` の広域例外捕捉、`daily_drawdown_guard` の reanchor/rearm 設計、`sidecar_signal_io` の read race は真面目に扱う価値がある。

3. **ただし、中心仮説の一部は補正が必要**  
   488# が強く押している `max()->min()` 問題は、現行の二段 ceiling 設計を踏まえると、そのまま P0 バグとは言い切れない。

4. **3/19 ログで新しい緊急問題が出ている**  
   `Cycle execution error: name '_sidecar_signal' is not defined` が連続しており、これは 488# に書かれていない。

5. **市場理論的には「単一バグ」より「防御レイヤー過積載」が本丸に近い**  
   `ranging_low_vol_skip`、`sell_dynamic_kill`、`skip_gate`、`spread_too_narrow`、`preflight_insufficient`、`no_feasible_quote`、`cross_venue_lead_lag_veto` が同時に効いており、maker が板から降り過ぎている。

---

## 1. 488# の中で妥当性が高い論点

### 1.1 `_execute_and_track_cycle()` の広域例外捕捉

これは有効な指摘である。

`scripts/v460/lib/orchestrator_mid_cycle.py:448`

```python
except Exception as e:
    logger.error(f"Cycle execution error: {e}", exc_info=True)
    ...
    await self._effective_sleep()
    return
```

問題は 2 つある。

- 呼び出し元に失敗が伝播しない
- 一部状態だけ restore して sleep に入るため、失敗が「穏当に見える」

特に 3/19 ログではこれが現実害になっている。

```text
2026-03-19 14:47:10 以降
Cycle execution error: name '_sidecar_signal' is not defined
```

が 4 分周期で継続しており、例外分類の弱さが live 障害の見落としを助長している。

補足として、488# の

- 「`SystemExit` が捕捉される」

は誤り。`except Exception` は `SystemExit` を捕捉しない。  
ここは **問題の方向は正しいが、例外階層の説明は補正が必要**。

### 1.2 `sidecar_signal_io` の stat/read race

これも妥当。

`scripts/v460/lib/sidecar_signal_io.py:125`  
`scripts/v460/lib/sidecar_signal_io.py:149`

`stat()` と `read_text()` の間に race window があり、SAC sidecar が高頻度更新するなら、

- mtime は旧版
- 中身は新版

のズレが起こりうる。  
今すぐ主犯とまでは言えないが、**P1 の堅牢化対象**としては適切。

### 1.3 `daily_drawdown_guard` の reanchor / rearm 設計

488# の危惧は概ね妥当。

`scripts/v460/lib/daily_drawdown_guard.py:247`

```python
_effective_pnl = side_pnl - _reanchor
_threshold = self._per_side_reanchor_budget_bps if _reanchor != 0.0 else self._per_side_hard_limit_bps
```

これは設計上、

- 初回 halt 後は「絶対損失」ではなく「reanchor からの追加損失」で再 halt

になるので、488# の言う「再開後に追加損失ウィンドウがある」はその通り。

同様に `scripts/v460/lib/daily_drawdown_guard.py:272` の cooldown re-arm も、

- その日中に一方向へ固定されやすい

という批判は理解できる。  
これは **市場理論上の問題** でもある。  
極端な `halt → release → rearm` は「危険時のみ完全停止」というより、「流動性供給を再開した瞬間にまた引っ込む」挙動になりやすく、quote continuity を壊す。

### 1.4 直近の収益悪化は「防御系の詰まり」が支配的

ここも 488# の危機感は正しい。

`2026-03-10` 〜 `2026-03-18` の fill records 集計では top cancel reason が:

- `ranging_low_vol_skip = 718`
- `sell_dynamic_kill = 546`
- `skip_gate = 414`
- `spread_too_narrow = 378`
- `timeout = 199`
- `preflight_insufficient = 142`
- `no_feasible_quote = 112`

である。

統計的に見ると、今の fill rate / PF 劣化は「1 個の致命バグ」より、

- ゲート
- clamp
- spread guard
- veto
- inventory / balance 制約

の多重作用と読む方が自然である。

---

## 2. 488# の中で補正が必要な論点

### 2.1 `_effective_max_ratio()` の `max()->min()` は、そのまま P0 バグとは言い切れない

488# の最重要主張はここだが、現行コードではそのままは当たらない。

`scripts/v460/lib/maker_price.py:583`

```python
def _effective_max_ratio(self, side: str) -> float:
    ...
    # 中間段の探索幅を確保する設計
```

さらに現行では

- `scripts/v460/lib/fill_config.py:354` `resolve_offset_ceiling()`
- `scripts/v460/lib/offset_pipeline.py:280` `execution_final_clamp`

が入っている。

つまり現在の設計は

1. 中間段では `max(base, side_ceiling)` で exploration を残す
2. 最終段で `resolve_offset_ceiling()` により side-aware ceiling へ切り詰める

という二段構造である。

したがって、488# の

- 「`max()` だから ceiling が完全に無効」
- 「PF 0.883 の直接原因」

という断定は強すぎる。

正しい言い換えは、

- **中間段では緩い ceiling、最終段で本 ceiling という設計になっている**
- 問題は `max/min` 単体ではなく、**中間 exploration と最終 clamp と veto/no-feasible-quote の相互作用**

である。

### 2.2 logging 欠落の一部は既に修正済み

488# の 1 章は一部古い。

現行コードには既に:

- `scripts/v460/lib/fill_cycle_executor.py:569`
  - `cancel_reason`
  - `sidecar_signal_status`
  - `sidecar_offset_bps`
- `scripts/v460/lib/orchestrator_post_cycle.py:302`
  - sidecar 集計
- `scripts/v460/lib/orchestrator_post_cycle.py:316`
  - cancel reason top5

が入っている。

ただし実ログでは `[487# sidecar]` / `[487# cancel]` が十分に観測できていない。  
よって正しい整理は、

- **コード上は改善済み**
- **runtime / deployment / ログローテーション側で観測が不十分**

である。

### 2.3 sidecar 設定バリデーション欠如は古い

488# の

- `sidecar_max_boost_bps`
- `sidecar_dead_zone`

未検証という指摘は現行コードでは古い。

`scripts/v460/lib/fill_config_validation.py:349`

- `sidecar_max_boost_bps >= 0`
- hard ceiling `<= 0.20`
- `0 <= sidecar_dead_zone < 1`

が既に入っている。

### 2.4 `stale_reprice_min_delta_jpy` hot-reload 対象外という指摘は誤り

`scripts/v460/lib/config_hot_reload.py:170` に既に含まれている。

一方で 488# が挙げた

- `sigma_floor`
- `vol_ratio_floor`

は実際に hot-reload 対象外であり、この点は妥当。

### 2.5 `microprice_depth > weights 長で IndexError` は誤り

`scripts/v460/lib/maker_price.py:525`

```python
w = weights[k] if k < len(weights) else 0.0
```

となっているため、`depth=6` で即 `IndexError` にはならない。

ただし問題が無いわけではない。

- 6 段目以降は重み 0 で silent no-op
- 設定したつもりでも情報が増えない

ので、**validation or warning 対象**ではある。

---

## 3. 488# が見落としている新しい重要論点

### 3.1 3/19 の `_sidecar_signal` NameError

これは 488# に入っていないが、今は最重要。

`results/v460/fill_test/logs/fill_test.log`

- `2026-03-19 14:47:10`
- `2026-03-19 17:25:15`

の間、`Cycle execution error: name '_sidecar_signal' is not defined` が継続している。

現ワークツリー上の `scripts/v460/lib/orchestrator_mid_cycle.py:141` では `_sidecar_signal` は定義済みなので、

- live プロセスが古いコードで動いている
- 部分反映された runtime drift がある
- 別経路のコードが走っている

のいずれかが疑わしい。

これは 488# の各種監査項目より先に止血すべきである。

### 3.2 `cross_venue_lead_lag_veto` → `NO_FEASIBLE_QUOTE` 連鎖

`2026-03-19 04:09` 以降のログでは:

```text
NO_FEASIBLE_QUOTE ... last_reason=cross_venue_lead_lag_veto
```

が連続している。

これは市場理論上、

- 毒性回避そのものは正しい
- しかし回避レイヤーが強すぎると maker は市場から退場する

という典型パターンである。

したがって現状は、

- `sell_dynamic_kill` だけが問題

ではなく、

- **cross_venue veto**
- **min_spread**
- **no_feasible_quote**

の相互作用が新たな fill rate 低下要因になっている。

### 3.3 `balance insufficient` は alpha 問題ではなく inventory / financing 問題

3/19 ログでは `buy insufficient, switching to sell immediately` が反復している。

これは単なる gate 調整の話ではなく、

- JPY 枯渇
- BTC 偏在
- side 偏り

の inventory 問題である。

市場理論的には、これは

- 「情報劣位で負けている」

というより、

- **資金制約で本来取りたい side を取れない**

状態であり、alpha の評価を汚す。

488# の「5,703 balance insufficient」を重視する方向は正しいが、改善手段は gate 緩和より **inventory recycling / financing discipline** 側に寄せるべきである。

---

## 4. 統計学的・市場理論的補強

### 4.1 因果と相関を分けるべき

488# は

- VPIN continuous ramp → `979 sell_dynamic_kill` の直接原因
- inverse skew damping → sell 損失拡大

といった因果を強く書いているが、現状の証拠は観測相関である。

統計的に本当に言うなら、少なくとも:

1. `kill activated` vs `not activated` の条件付き平均
2. VPIN bucket ごとの kill odds
3. `inv_skew_factor < 0` 時の sell fill / pnl 条件付き比較
4. 同一 SHA / 同一 config での before/after

が必要である。

今のログから言えるのは、

- これらが有力仮説

までで、**単独犯の断定はまだ早い**。

### 4.2 直近フェーズの本丸は「防御過積載」

直近ログでは:

- `ranging_low_vol_skip`
- `sell_dynamic_kill`
- `skip_gate`
- `spread_too_narrow`
- `preflight_insufficient`
- `no_feasible_quote`
- `cross_venue_lead_lag_veto`

が多層で効いている。

maker の市場理論からすると、これは

- adverse selection 回避のための防御を重ね過ぎて
- 参加率を落とし
- spread capture 機会を自ら捨て
- inventory 修復も遅らせている

構図に近い。

よって次の一手は「さらに新しい防御を足す」ではなく、

- **どのレイヤーをどの regime/side/time 帯で止めるか**

を整理する participation budget の設計である。

### 4.3 recent sample では `sell` 単独問題に閉じない

`2026-03-10` 〜 `2026-03-18` の top cancel では `sell_dynamic_kill` が大きい一方、

- `ranging_low_vol_skip`
- `preflight_insufficient`
- `no_feasible_quote`

も重い。

したがって「sell を直せば戻る」というより、

- `buy` の資金制約
- `buy` の veto/no-feasible 連鎖
- 全体の participation 過少

も同時に解く必要がある。

---

## 5. 489# としての優先順位

### P0

1. **live runtime の `_sidecar_signal` NameError を止血**
2. **runtime drift / deploy drift の確認**
3. **`NO_FEASIBLE_QUOTE` の reason decomposition を first-class 観測に上げる**

### P1

1. `_execute_and_track_cycle()` の例外分類
2. `daily_drawdown_guard` の reanchor / rearm 設計見直し
3. `cross_venue veto × min_spread × no_feasible_quote` の同時作用分析
4. `VPIN` / `inv_skew` 仮説を same-SHA 条件付き統計で検証

### P2

1. hot-reload 対象フィールドの整理 (`sigma_floor`, `vol_ratio_floor` など)
2. `microprice_depth > len(_MICRO_WEIGHTS)` の warning / validation
3. progress log と cycle log の runtime 実観測整合確認

---

## 6. 結論

488# は「雑に否定すべき文書」ではない。  
特に、

- 例外安全性
- DD guard の状態設計
- inventory / balance friction の重さ

を正面から扱っている点は良い。

一方で、

- `max()->min()` を中心原因に置くこと
- 既に修正済みの validation 欠如を現行問題として扱うこと
- 観測相関から直接因果へ飛ぶこと

は補正が必要である。

いま最も重要なのは、488# の監査リストをそのまま実装 backlog 化することではなく、

1. **runtime をまず安定化させる**
2. **過積載防御のどれが participation を殺しているかを同一条件で分解する**
3. **inventory / financing 問題を alpha 問題と切り分ける**

である。

この順序で進めれば、488# の良い部分を活かしつつ、誤った優先順位づけを避けられる。
