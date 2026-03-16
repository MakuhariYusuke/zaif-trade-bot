# 412# 410-411 SAC設定・Seed感度レビュー

**Date**: 2026-03-14  
**Phase**: phg (フェーズ横断)  
**Type**: rev (レビュー)  
**対象**: `410_ph4_rpt_g3_pass_deep_dive.md`, `411_phg_rpt_seed_sensitivity_analysis.md`  
**参照コード**: `scripts/v460/lib/sac_common.py`, `scripts/v460/lib/tasks/sac_train.py`, `ztb/trading/environment/components/calculators/reward_calculator.py`, `configs/v460/experiments/g2_sac_reward_clean*.yaml`

---

## §1 エグゼクティブサマリ

410# と 411# の方向性は概ね良い。特に以下は妥当である。

- `reward-clean` が `reward-tuned` より G3 適合性で明確に改善していること
- `policy_kwargs` 転送実装が入り、`net_arch` / `weight_decay` を YAML から制御できるようになったこと
- `[256,256]` のまま 20K を回し続けるより、容量と seed 安定性を見直すべきという問題意識

ただし、現時点でそのまま採用すると危うい論点もある。

1. **410# の `reward_profit_corr` 解釈は実装とズレている**
2. **411# の「100Kなら [128,128] が最適」は言い切りが強すぎる**
3. **`weight_decay=1e-4` は妥当な第一候補だが、最適値とまでは言えない**
4. **`learning_starts=5000` は 100K 本番には許容だが、20K 検証では重い**
5. **Checkpoint Ensemble (state_dict 平均) は SAC では実用第一の手ではない**

結論を先に書くと、**次の本命は `[128,128] + weight_decay小 + checkpoint選択継続`** でよいが、`learning_starts=5000` と `M5` は同列の確定事項として扱わない方がよい。  
また、**seed456 の `corr<0` は警戒材料ではあるが、deploy 観点では seed789 の checkpoint 依存の方が運用リスクが高い**。

---

## §2 重要な補正事項

### §2.1 410# の `reward_profit_corr` 解釈はそのままでは危険

410# §3.1 では `reward_profit_corr` を `corr(episode_mean_reward, episode_gross_pnl)` と置いているが、実装はそうなっていない。  
実際には `scripts/v460/lib/sac_common.py` で、**ステップごとの reward 累積** と **ステップごとの realized PnL 累積** の相関を計算している。

- reward 側: `all_reward_steps` を累積
- PnL 側: `total_pnl` の差分 (`all_pnl_steps`) を累積
- 算出箇所: `scripts/v460/lib/sac_common.py:269`

したがって `seed456 corr=-0.20` は、

- reward と realized PnL の**時系列整合が悪い**
- reward shaping が**完全に profit と逆方向**である
- seed456 が**初期値の偶然だけで勝っている**

のいずれか一つに即断できない。

特に `reward_calculator` は `gross_pnl = (current_price - previous_price) * position` のような**密な mark-to-market 型 reward** を返す一方、G3 の PnL 側は **realized PnL 差分** である。  
このため、以下のような現象だけでも `corr<0` は起こりうる。

1. 保有中は含み損が続くが、決済でまとめて利益化する
2. reward clip により magnitude 情報が潰れ、累積形状だけが残る
3. inventory 調整と alpha 取りが時間的にずれる
4. 1 episode 評価なので regime 偏りを強く受ける

**補正結論**: `seed456 corr<0` は「警戒シグナル」ではあるが、410# が置いているほど直接的な「学習失敗の証明」ではない。

### §2.2 deploy リスクは seed456 より seed789 も重い

410# は seed456 を主リスクとして強調しているが、OOS checkpoint 安定性だけ見ると seed789 の方が不安定である。

| Seed | corr | OOS正 checkpoint 数 | コメント |
|------|------|---------------------|----------|
| 42 | +0.54 | 1/4 | late bloomer |
| 123 | +0.56 | 4/4 | 最も安定 |
| 456 | -0.20 | 4/4 | alignment 警戒だが OOS は安定 |
| 789 | +0.61 | 1/4 | **best checkpoint 依存が強い** |

このため、運用優先度としては

1. `reward_profit_corr` の意味を正しく理解する
2. `seed456` の alignment 警戒を残す
3. それと同時に **seed789 型の checkpoint fragility** を監視する

の順で扱うのが妥当である。

### §2.3 411# の過パラメータ化認識は正しいが、比率は楽観的ですらある

411# の 215,044 / 58,372 / 16,900 は、20次元観測・1次元 action の **trainable params (actor + critic)** として整合している。  
ただし RL の 20K transition は i.i.d. データではなく、**時系列相関が強い**。  
よって 411# の「10.8x」はむしろ下限寄りで、**実効サンプルサイズで見れば過パラメータ化はもっと重い**。

この点は 411# の主張を弱めるのではなく、むしろ

- `[256,256]` 維持を正当化しにくい
- `[128,128]` へ縮める方向はかなり合理的

という意味で補強する。

---

## §3 個別論点への回答

### §3.1 `M1 net_arch [128,128] vs [64,64]`

**回答**: 「100K 前提なら `[128,128]` を第一候補にする」は妥当。ただし「最適」と断言するにはまだ不足。

理由は以下。

- `[256,256]` は現状明らかに大きすぎる
- 100K に伸ばすなら `[128,128]` は capacity を残しつつ trainable params を `58,372` まで落とせる
- 一方で、この問題は 20 feature / 1 action の比較的低次元問題であり、**seed 安定性最優先なら `[64,64]` も普通に有力**

整理すると次の通り。

| 構成 | trainable params | 20K比 | 100K比 | 評価 |
|------|------------------|-------|--------|------|
| [256,256] | 215,044 | 10.8x | 2.15x | 大きすぎる |
| [128,128] | 58,372 | 2.9x | 0.58x | **100K本命候補** |
| [64,64] | 16,900 | 0.85x | 0.17x | **安定性本命候補** |

**実務判断**:

- 1本しか回せないなら `[128,128]`
- 2本回せるなら `[128,128]` と `[64,64]` を両方比較
- 20K の短期判定だけで `[128,128] 最適` と確定しない

### §3.2 `weight_decay 1e-4` は適切か

**回答**: `1e-4` は「悪くない初手」だが、最適値と見なすには早い。  
また、現 SB3 SAC では `optimizer_kwargs` が actor/critic 両方に入るため、**効き方がやや鈍く雑**である。

確認できる点:

- `SACPolicy` は `optimizer_kwargs` を actor optimizer と critic optimizer の両方に渡している
- つまり `weight_decay=1e-4` は critic だけでなく actor にも同時適用される

解釈としては、

- **critic 過学習抑制**には一定の合理性がある
- ただし actor まで同じ L2 を掛けるのは、profit-first の観点では最適とは限らない

#### dropout / layer normalization との比較

- **dropout**: RL では第一選択にしない方がよい  
  理由は、off-policy 学習で既にノイズ源が多く、さらに stochasticity を増やすと収束解釈が難しくなるため
- **layer normalization**: dropout よりは有望  
  ただし現行 MLP policy にそのまま刺さる訳ではなく、custom policy 側の変更コストがある

したがって優先順位は次の順がよい。

1. net_arch 縮小
2. `weight_decay` の小レンジ比較 (`0`, `1e-5`, `1e-4`)
3. 必要なら actor/critic 非対称化
4. それでも不安定なら LayerNorm
5. dropout は後回し

### §3.3 `learning_starts=5000`

**回答**: 100K 本番なら許容。20K 検証ではやや重い。

20K では:

- `learning_starts=1000` → 学習可能区間 19K
- `learning_starts=5000` → 学習可能区間 15K

さらに `gradient_steps=2` なので、更新回数ベースでも

- `1000` → 約 38,000 updates
- `5000` → 約 30,000 updates

となり、**約21% 分の更新を失う**。

よって判断は以下がよい。

- **20K 高速実験**: `1000` か多くても `2000`
- **100K 本番候補**: `5000` を試す価値あり

つまり `M1/M2/M3` を一気に束ねるより、**`M3` だけは 100K 文脈で分けて評価**した方が attribution がきれいになる。

### §3.4 `seed456 の corr<0 なのに OOS 正`

**回答**: 「初期重みの偶然」以外にも、十分ありうる説明がある。

主な候補は以下。

1. **reward と評価 PnL の定義差**
   dense reward は mark-to-market、G3 PnL は realized 差分
2. **time-lag**
   学習上の reward では中盤が悪くても、後半の決済で realized PnL が回収される
3. **clip 歪み**
   `reward_clip [-1, 1]` により reward の大小関係が粗くなる
4. **inventory 修復の寄与**
   reward は短期的に悪く見えるが、ポジション調整が最終利益に効く
5. **val window 特有の regime**
   その seed の policy が OOS 区間の regime と偶然合った

よって、seed456 は

- 「完全な事故 seed」
- 「reward 定義の欠陥だけで勝っている」

のどちらかに決め打ちせず、**alignment 指標と OOS 安定性を切り分けて観察**すべきである。

### §3.5 `Checkpoint Ensemble (M5)` の実用性

**回答**: `policy.state_dict()` の単純平均は、SAC では通常おすすめしない。

理由:

1. actor だけでなく twin critic / target critic もあり、整合が崩れやすい
2. 離れた checkpoint 同士の平均は、中間の「どちらでもない policy」を作りやすい
3. optimizer state は平均しても意味が薄い

より実用的な代替は以下。

- **best checkpoint 選択を継続**
- **top-K checkpoint の inference ensemble**  
  action 平均、もしくは checkpoint vote
- **EMA/SWA 的な近接 checkpoint 平均**  
  ただし actor/critic の整合確認を必須にする

**結論**: M5 は研究テーマとしてはありだが、今の優先度では P2 ではなく **P3 寄り**。

---

## §4 411# に追加したい改善案

### §4.1 actor / critic 非対称アーキテクチャ

SB3 SAC は `net_arch` に dict を渡せるため、例えば

```yaml
policy_kwargs:
  net_arch:
    pi: [64, 64]
    qf: [128, 128]
```

のような非対称化が可能である。

この案件では、

- actor: 1次元 action なので小さめでも足りる可能性が高い
- critic: Q 推定の方が表現力を欲しやすい

ため、**両方を同じ幅で縮めるより自然**である。

### §4.2 `corr` だけでなく「正 OOS checkpoint 本数」を正式指標化

現状は `reward_profit_corr` が目立っているが、deploy 的には

- `reward_profit_corr`
- best checkpoint の ROI
- **正の OOS checkpoint 本数**
- OOS ROI の分散

をセットで見た方がよい。  
seed456 と seed789 の優先順位逆転も、ここを入れると見えやすい。

### §4.3 M1/M2 と M3 を同時にいじらない

411# の `reward_clean_small` は実務上の第一案としては良いが、レビュー観点では

- `net_arch` の効果
- `weight_decay` の効果
- `learning_starts` の効果

が混ざる。  
次の判断材料を得たいなら、最低でも

1. `[128,128], wd=0, ls=1000`
2. `[128,128], wd=1e-4, ls=1000`
3. `[128,128], wd=1e-4, ls=5000`

の順で attribution を取った方がよい。

### §4.4 100K 前に slippage / friction 感度を並行評価

410# が指摘した通り、今の G3 PASS は friction-free 条件での達成である。  
したがって 100K を伸ばすのと並行して、

- 1tick slippage
- spread widening
- fill miss

への耐性を同時に見るべきである。  
**SAC 側だけを磨いても、執行摩擦で優位が消えるなら profit-first ではない**。

---

## §5 優先順位付きの推奨アクション

### P0

1. 410# の `reward_profit_corr` の説明を実装準拠に補正する
2. `reward_clean_small` を「本命候補」と位置付けるが、「最適」表現は避ける
3. `M5 checkpoint ensemble` は保留し、best checkpoint 選択を維持する

### P1

4. 20K では `M1` と `M2` を優先、`M3` は切り離して検証する
5. 100K 本番候補は `[128,128]` を主軸にしつつ、比較対象として `[64,64]` も残す
6. `positive OOS checkpoint count` を seed 安定性メトリクスに追加する

### P2

7. actor / critic 非対称 `net_arch` を検討する
8. `weight_decay` は `0 / 1e-5 / 1e-4` の小レンジで比較する
9. LayerNorm は simple knobs が尽きてから検討する

---

## §6 最終判定

410# と 411# は、**「SAC をこのまま 100K に伸ばしてよいか」を判断する材料としては十分有益**である。  
ただし採用の仕方は少し修正した方がよい。

- **採用してよいもの**
  - `[256,256]` 見直し
  - `[128,128]` を第一候補にする
  - `weight_decay` 導入を試す
  - `policy_kwargs` YAML 化

- **採用を保留すべきもの**
  - `[128,128] が最適` という断定
  - `learning_starts=5000` の 20K 常用
  - `Checkpoint Ensemble` の即実戦投入

profit-first の観点で次の一手を一文で言うなら、  
**「100K本命は `[128,128]` 系でよい。ただし、`learning_starts` と checkpoint ensemble に飛びつく前に、reward/PnL 指標の意味を正し、`[64,64]` と OOS安定性も同時比較するべき」** である。
