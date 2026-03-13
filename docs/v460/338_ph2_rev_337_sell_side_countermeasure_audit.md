# 338# 337# レビュー — Sell-side 悪化対策の監査と追加 Finding

> **種別**: rev  
> **対象**: 337#  
> **起票**: 2026-03-08  
> **観点**: reviewer / profitability-first / systems engineering / market theory  
> **補足確認**: `docs/v460/335_ph2_gemini_31_pro_review_314_334_comprehensive_audit.md`, `docs/v460/336_ph2_rev_334_335_claims_validation_and_measures.md`, `analysis/333_sha_isolated_analysis.py`, `ztb/risk/sell_dynamic_kill.py`, `scripts/v460/lib/orchestrator_guards.py`, `scripts/v460/lib/cycle_gate_aggregator.py`, `scripts/v460/lib/orchestrator_post_cycle.py`, `configs/v460/fill_test.yaml`  
> **追加検証**: `analysis/333_sha_isolated_analysis.py --sha eb24cf4a`, `--sha fea79119`, `--sha eb24cf4a fea79119`  
> **テスト**: `tests/unit/v460/test_157_regime_features.py`, `tests/unit/v460/test_194_cycle_gate.py`, `tests/unit/v460/test_273_kill_time_limit_halt_untick_recovery_grace.py` → **94 passed**

---

## §1 Findings

| # | 重大度 | 対象 | Finding | 推奨対応 |
|---|---|---|---|---|
| 1 | **CRITICAL** | `ztb/risk/sell_dynamic_kill.py:510`, `scripts/v460/lib/orchestrator_guards.py:48`, `configs/v460/fill_test.yaml:628` | `threshold_offset_bps` の符号解釈が逆。正の offset は kill を緩和せず、**threshold を 0 側へ寄せて kill を厳格化**している | `threshold += offset` を見直し、緩和ならより負側へ動くよう修正。関連テストも全面修正 |
| 2 | **HIGH** | `docs/v460/337_ph2_rpt_sell_side_degradation_countermeasures.md`, `analysis/333_sha_isolated_analysis.py:148`, `scripts/v460/lib/orchestrator_guards.py:113` | 337# は **risk-control 用指標 (`post_fill_30s_pnl`)** と **収益評価指標 (`ev_weighted_pnl` 優先)** を混同しており、headline 数値が再現系と一致しない | 337# の PnL 集計基準を明示し、control metric と gate metric を分離して再集計 |
| 3 | **HIGH** | `scripts/v460/lib/orchestrator_guards.py:67`, `scripts/v460/lib/fill_config.py:575`, `scripts/v460/lib/fill_config_parser.py:470`, `configs/v460/fill_test.yaml:634`, `scripts/v460/lib/cycle_gate_aggregator.py:606` | 337# の「sell inv_relaxation が存在しない」は **現 HEAD では古い**。すでに sell 側 inv_relaxation が追加されており、さらに `sell_guard_inv_bypass_threshold` も別層に存在する | inventory 由来の sell 緩和ロジックを一箇所に統合。二重緩和経路を作らない |
| 4 | **HIGH** | `analysis/333_sha_isolated_analysis.py`, `configs/v460/fill_test.yaml:598`, `configs/v460/fill_test.yaml:615` | `sell_dynamic_kill` 単独原因に寄せ過ぎ。`eb24cf4a` 単独では cancel reason は `skip_gate=51` が `sell_dynamic_kill=42` を上回る | 「threshold だけ触れば直る」と考えず、filter stack 全体を階層化して評価 |
| 5 | **MEDIUM** | `docs/v460/337_ph2_rpt_sell_side_degradation_countermeasures.md:280`, `docs/v460/337_ph2_rpt_sell_side_degradation_countermeasures.md:397` | `-1.0bps` 提案は、観測窓の最悪 rolling mean `-0.888` をぎりぎり回避する **hindsight fit** に近い | 段階的 ladder (`-0.5/-0.8/-1.0`) で検証し、一足飛びに決めない |
| 6 | **MEDIUM** | `scripts/v460/lib/orchestrator_post_cycle.py:107`, `docs/v460/337_ph2_rpt_sell_side_degradation_countermeasures.md:330` | `balance_forced_switch` を rolling PnL から **完全除外** すると、forced trade の実コストを kill 制御が見失う | hard exclude ではなく、別 KPI 化または downweight を優先 |
| 7 | **MEDIUM** | `ztb/risk/sell_dynamic_kill.py:493`, `configs/v460/fill_test.yaml:602`, `docs/v460/337_ph2_rpt_sell_side_degradation_countermeasures.md:349` | `resume_window=10` は cycle 基準であり、337# の「約20分」解釈は固定 wall-clock ではない | cooldown は秒基準ログも併記し、必要なら時間基準へ移行 |

---

## §2 最重要 Finding — inv_relaxation の符号が逆

これは現時点で最も危険な見落としである。

`ztb/risk/sell_dynamic_kill.py:510` では、現在こう実装されている。

```python
# threshold_offset_bps > 0 は閾値を less negative にする (kill されにくくなる)
if threshold_offset_bps != 0.0:
    threshold += threshold_offset_bps
```

しかし条件は `rolling_mean < threshold` であり、threshold を `-0.5 -> -0.3` に動かすと **kill は起きやすくなる**。  
実際に最小再現でも以下を確認した。

| rolling_mean | base threshold | offset | threshold_used | killed |
|---|---:|---:|---:|---|
| -0.6 | -0.5 | 0.0 | -0.5 | True |
| -0.6 | -0.5 | +0.2 | -0.3 | **True (より厳格)** |
| -0.6 | -0.5 | -0.2 | -0.7 | **False (これが本当の緩和)** |

このため、現行の `buy_dynamic_kill_inv_relaxation` も `sell_dynamic_kill_inv_relaxation` も、**コメント上は緩和だが実挙動は逆** である可能性が高い。

特に sell 側は現在:

- `sell_dynamic_kill.threshold_bps = -0.3` (`configs/v460/fill_test.yaml:601`)
- `sell_dynamic_kill_inv_relaxation.max_bps = 0.3` (`configs/v460/fill_test.yaml:638`)

なので、在庫過剰時に offset が最大まで乗ると `threshold_used = 0.0` になる。  
これは **rolling mean が少しでも負なら kill** という意味であり、337# が狙った「sell を通しやすくする」と真逆である。

### 含意

337# の sell-side 問題は妥当だが、**その対策の一部は現 HEAD で逆向きに効いている**。  
ここを直さずに threshold だけ調整すると、効果判定自体が汚染される。

---

## §3 337# の数値はそのまま gate 判断に使えない

337# は `eb24cf4a` について

- buy: `+18.0bps`
- sell: `-24.5bps`

と置いているが、再現用に昇格済みの `analysis/333_sha_isolated_analysis.py` で同一 SHA を切ると、既定の集計では以下になった。

### `eb24cf4a` 単独

- records=306, fills=110, fill_rate=35.9%
- overall mean `-0.388bps`, sum `-42.66bps`
- sell mean `-0.586bps`, sum `-32.24bps`, p10 `-8.145`
- buy mean `-0.189bps`, sum `-10.41bps`, p10 `-4.626`
- cancel reasons: `skip_gate=51`, `sell_dynamic_kill=42`

### `eb24cf4a + fea79119`

- records=395, fills=129, fill_rate=32.7%
- overall mean `-0.394bps`, sum `-50.80bps`
- sell mean `-0.548bps`, sum `-35.59bps`
- buy mean `-0.238bps`, sum `-15.20bps`
- cancel reasons: `sell_dynamic_kill=72`, `skip_gate=58`

差の主因は、指標レイヤの違いだと見てよい。

- `scripts/v460/lib/orchestrator_guards.py:113` の kill 管理は `post_fill_30s_pnl` で回る
- `analysis/333_sha_isolated_analysis.py:148` の再現分析は `ev_weighted_pnl -> post_fill_30s_pnl -> pnl_bps` の順で拾う

### レビュアー判断

337# の「sell kill の制御問題を診る」目的自体は正しい。  
ただし、そのまま **戦略全体の profitability 判断** に接続してはいけない。

必要なのは次の 2 系統のレポート分離である。

1. **Control report**: `post_fill_30s_pnl` 基準で kill ループの挙動を診る
2. **Strategy report**: `ev_weighted_pnl` 基準で gate / 戦略収益性を診る

---

## §4 337# が過小評価している設計上の論点

### §4.1 sell 側の relief は「未実装」ではなく「重複実装リスク」

337# は `sell_dynamic_kill_inv_relaxation` の新設を提案したが、現 HEAD では既に以下がある。

1. `sell_dynamic_kill_inv_relaxation_*` in `scripts/v460/lib/orchestrator_guards.py:67`
2. `sell_guard_inv_bypass_threshold` in `scripts/v460/lib/cycle_gate_aggregator.py:606`

つまり sell 緩和はすでに二層化し始めている。  
ここへさらに別経路を足すと、**どの層が kill を最終決定しているのか分からなくなる**。

### §4.2 「double-filter」ではなく「overlapping filter stack」

337# は `skip_gate + dynamic_kill` を主に挙げているが、実際の sell pass はさらに

- `skip_sell_trending`
- `trending_sell_as_offset`
- `sell_guard_inv_bypass`
- `spread_too_narrow`
- `timeout`
- `stale_adverse_drift`
- `daily_drawdown` 系

の影響も受ける。

よって問題は「二重フィルタ」より、**複数 safety layer が非階層的に並列配置されていること**である。  
市場理論的に望ましい順序は、

1. quote を広げる
2. participation を落とす
3. duty-cycle 実行へ縮退する
4. 最後に hard kill

であって、いきなり複数の hard blocker が別層から飛んでくる構造は望ましくない。

### §4.3 `resume_window=10` は固定 20 分ではない

337# は resume を「約20分」と解釈しているが、実際は cycle 長が可変なので wall-clock は一定でない。  
また `ztb/risk/sell_dynamic_kill.py:431` 以降には `max_kill_duration_sec` もあり、kill は完全永久停止ではない。

したがって、「回復不能 deadlock」という表現は少し強い。  
より正確には **回復が極端に遅い stale control loop** である。

---

## §5 市場理論面での補強

### §5.1 337# の非対称性批判は正しい

`sell=-0.3` vs `buy=-1.5` の 5 倍非対称は、Glosten-Milgrom 的な sell 側 adverse selection を考慮しても大き過ぎる。  
ここは 337# の指摘通りで、**完全対称にする必要はないが、現状の非対称は過大** である。

### §5.2 ただし「-1.0 を即採用」は overfit に近い

337# の `-1.0bps` は、「今回観測された rolling mean の最悪 `-0.888` を跨がない値」として綺麗に見える。  
だがそれは裏返すと、**今回のサンプルにちょうど合う値**でもある。

profitability-first でやるなら、推奨は一足飛びではなく ladder である。

1. `-0.3 -> -0.5`
2. まだ loop が出るなら `-0.8`
3. それでも駄目なら `-1.0`

この方が、threshold と AS tail のトレードオフを読みやすい。

### §5.3 forced trade は「消す」のではなく「別建てで見る」

337# の `balance_forced_switch` 除外提案は理解できるが、完全除外は危ない。  
forced trade は確かに通常 MM 品質指標ではないが、**システムが生む実コスト** ではある。

既に `scripts/v460/lib/orchestrator_post_cycle.py:107` では buy forced/normal の KPI 分離が入っている。  
この方向を拡張し、

- buy forced
- buy normal
- sell forced
- sell normal

の 4 分割 KPI にすべきであり、kill 制御から完全に消してはいけない。

### §5.4 count window より time-decay を上げるべき

337# は EWMA を P2 に置いているが、私は **P1 相当** に引き上げてよいと見る。  
理由は単純で、fill が減るほど 50-fill window の意味が薄れ、**古い損失がいつまでも効く** からである。

特に hard kill と count window は相性が悪い。  
参加を止めるほど新データが入らず、制御が stale になる。  
この問題には、threshold 数値だけでは本質的に勝てない。

---

## §6 推奨アクション

### P0

1. **`threshold_offset_bps` の符号逆転を修正する**  
   これが最優先。`buy/sell_dynamic_kill_inv_relaxation` の効果判定は、この修正前には信用しない方がよい。

2. **`tests/unit/v460/test_286_comprehensive_resolution.py` の期待値を修正する**  
   現在は test 名と assertion が逆転しており、バグを隠している。

3. **337# の headline 数値に metric ラベルを付ける**  
   `post_fill_30s_pnl` ベースなのか `ev_weighted_pnl` ベースなのかを明記する。

### P1

4. **sell threshold は ladder で動かす**  
   `-0.5 -> -0.8 -> -1.0` の段階検証。最初から `-1.0` に飛ばない。

5. **forced trade KPI を sell 側にも拡張する**  
   hard exclude ではなく、forced/normal の dual window で見る。

6. **inventory relief を一箇所に統合する**  
   `sell_dynamic_kill_inv_relaxation` と `sell_guard_inv_bypass_threshold` を併存させない。

### P2

7. **count-based rolling を EWMA / time-decay へ置換または併設する**

8. **gate hierarchy を文書化する**  
   quote widening → participation reduction → duty-cycle → hard kill の順に責務を整理する。

---

## §7 総評

337# は、「sell-side 崩壊を sell_dynamic_kill 観点から捉え直した」点で価値がある。  
特に **5 倍非対称の危険性** と **rolling window の自己強化ループ** を捉えた点は妥当だった。

一方で、現 HEAD で見ると見落としも大きい。

1. `inv_relaxation` は既に一部入っている
2. しかもその符号が逆で、意図と反対方向に効いている可能性が高い
3. `sell_dynamic_kill` だけでなく `skip_gate` も同等以上に重い区間がある
4. headline PnL が metric 混在で再現しない

したがって、次にやるべきことは「sell threshold を今すぐ強く緩める」ことそのものではない。  
**まず control loop の符号と指標定義を正し、その上で threshold を段階的に触ること**である。

profitability-first に言い換えると、今の最短ルートは「パラメータを大きく動かすこと」ではなく、**逆向きに効いている制御を潰すこと**である。
