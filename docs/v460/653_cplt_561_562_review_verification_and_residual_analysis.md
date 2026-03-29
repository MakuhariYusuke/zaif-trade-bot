# 653# 561-562 レビュー検証 — 650# 実測による提案の成否判定と残存盲点

- **日付**: 2026-03-30
- **目的**: 561#(DRC提案)・562#(統合レビュー)が提示した改善提案と盲点指摘を、650# の 2026-03-29 実測データ (13 RT, 27 fills, SHA 5832c87fe) で検証。605#/606# の「渙」の流れ—「理論より計測」(565# 結論) を尊重し、データに基づく成否判定を行う。
- **入力**: 561#, 562#, 563#, 565#, 605#, 606#, 650# (roundtrip analysis + deep dive)

---

## §0 総括

562# は16の改善提案 (P-A〜P-I + A1〜A6 + B1〜B5) と2つの構造改善案を提示した。650# の実測データにより、**実装済み施策の実効性**と**未実装提案の妥当性**を初めてデータで評価できる。

結論を先に述べる:

| 評価 | 内容 |
|------|------|
| **562# が正しかった** | P-B (ceiling引上げ) は実施済みだが **sell 0.40 でも 92% が飽和** → さらなる対策が必要。562# が「0.30 では不十分」と指摘した通り |
| **562# が正しかった** | §4.2 "乗算チェーンの多重計上" — VG boost が ceiling で頭打ちになり防御が形骸化 (650# 問題4) は、まさに「三層が互いを打ち消し合っている」実例 |
| **562# が過小評価した** | inv_skew の実質無効化 (25/27 fills = 0.0000) は 562# では言及されず 565# 盲点8 で指摘。650# で壊滅的に確認 |
| **562# が過小評価した** | preflight_insufficient — 562# P-F は「調査項目」としたが、650# では **46.2% 全キャンセルの最大原因**。JPY枯渇の深刻さを見誤った |
| **561# の DRC は方向性正解だがスコープが狭い** | 650# の 4構造問題 (inv_skew, sidecar, ceiling, VG) は DRC 単独では解決不能。562# が「既存機構の統合再設計」と提案した方が正しい |
| **565# の「計測が欠けている」は的中** | 650# の RT 単位分析により、初めて因果推論可能なデータが得られた |

---

## §1 562# 提案の実装状況と650#による実効性評価

### 1.1 実装済み提案の実効性

| # | 提案 | 実装 | 650# 実測結果 | 実効性評価 |
|---|------|------|-------------|-----------|
| **P-A** | CV tighten sell無効化 | ✅ 565# | sell tighten による -1.10bps 損失は解消。650# では CV widen によるsell損失の報告なし | **効果あり** |
| **P-B** | Ceiling sell:0.40, buy:0.35 | ✅ 565# | sell 12/13 が ceiling 0.40 固定 (92% 飽和)。pre_clamp 0.60 が 3件 → 33% 情報損失。buy は 0.16–0.35 に分散 | **buy は改善、sell は不十分** |
| **P-C** | Stage max_mult 2.0 | ✅ 565# | 各段上限 2.0 のハードコード。乗算膨張は抑制されたが ceiling 飽和は解消せず | **部分効果** |
| **P-D** | min_spread_jpy 動的化 | ✅ 625# | ATR×1.2, cap 3.0bps。spread_too_narrow cancel = 15.6% (49/314) — 依然多い | **部分効果** |

### 1.2 未実装提案の650#による妥当性再評価

| # | 提案 | 650# データによる判定 | 推奨 |
|---|------|---------------------|------|
| **P-E** | hour_lot_scale (時間帯lot縮小) | 650# のテール集中は 12-13時 (MCB HALT連鎖)。時間帯問題は ceiling/MCB の問題であり lot 縮小では解決しない | **保留** — MCB+position 対策 (650# I2) が先 |
| **P-G** | DRC (動的ceiling) | eDRC は実装済みだが α=β=0 で無効。sell ceiling 0.40 でも 92% 飽和 → DRC の必要性は持続 | **有効** — ただし576# インシデント後のα/β再推定が前提 |
| **P-H** | AS損失額削減 | 650# の Q1 (< 2.4bps) avg PnL = -0.86bps。ceiling 引上げ後も低スプレッド帯は依然負。損益分岐の前提は健全 | **有効** — low-spread guard (650# I3) が即効 |
| **P-I** | A-S参照スプレッド (δ*) | δ* は narrow guard で使用中。650# で spread_too_narrow 15.6% → delta_star が min_spread guard に寄与。ただし base_offset 出発点としての利用は未着手 | **長期有効** |

### 1.3 562# §4 構造改善案の検証

| # | 提案 | 650# による検証 | 判定 |
|---|------|---------------|------|
| **§4.2** | 乗算チェーン多重計上 | VG boost > 1.0 のとき sell offset は既に ceiling → VG boost が ceiling で吸収されて無意味化 (650# 問題4)。**まさに562#が予言した「三層が互いを打ち消す」構造** | **確認。未解決** |
| **§4.3** | AS Risk Score max結合 | 未実装。650# の sell 12/13 fills が ceiling 固定 = 9段の演算結果が1つの数値に潰されている。max結合は依然として合理的選択肢 | **方向性は有効** |

---

## §2 565# 盲点の650#時点ステータス

| # | 盲点 | 565# 時点 | 650# 時点 | 判定 |
|---|------|----------|----------|------|
| 1 | PnL計測窓非対称 (sell=90s) | Critical指摘 | E3崩壊は修正済。sell "30s PnL" = 実態90s の命名問題は残存 | **PARTIAL** |
| 2 | spread_capture/AS_cost 未活用 | 未活用 | section_execution_quality + section_spread_decomposition で活用 | **RESOLVED** |
| 3 | Regime遷移遅延AS (6分) | 未定量化 | 未定量化。650# RT#1 (ranging→trending_up で -2.45bps) が疑似例 | **UNRESOLVED** |
| 4 | AS burst 自己相関 | 未実装 | 未実装 | **UNRESOLVED** |
| 5 | 曜日効果 | 未分析 | 未分析 | **UNRESOLVED** |
| 6 | Kelly/lot_sizing 矛盾 | kelly=true, lot=false | 両方 false に統一（矛盾解消）。実運用は未着手 | **RESOLVED (矛盾のみ)** |
| 7 | Pre-clamp offset 分布 | 不在 | section_clamp_saturation で分布出力。ceiling 0.40 設定の根拠に使用 | **RESOLVED** |
| 8 | inv_skew trending無効化 | 249#未検証 | 650# で壊滅的に確認: 25/27 = 0.0000。regime_gate + neutral_band + decay_tau の三重無効化 | **UNRESOLVED (深刻化)** |

---

## §3 650# が新たに発見した562#/565#にない問題

650# の RT 単位分析で、562#/565# が予見できなかった構造問題が浮上した。

### 3.1 MCB HALT × open position 複合リスク

**562#/565# でのカバー**: なし（MCB は 606# で有効化されたばかり）

**650# 実測**: RT#2 で MCB HALT 4連鎖 × preflight_insufficient 8回 = 69.7分閉塞 → -14.27bps。MCB は高ボラ検知として正しく動作したが、open position を持った状態での HALT が在庫リスクを増幅した。

**理論的位置づけ**: Foucault et al. (2007) の halt-induced inventory risk。取引所の trading halt と同じ構造。ただし取引所の halt にはオークション型再開があるが、現行 MCB にはない。

**562# P-E (lot縮小) との関係**: lot 縮小では MCB HALT 中の open position リスクは縮小しない。MCB + position の問題は 562# のスコープ外だった。

### 3.2 VG × ceiling 相互無効化

**562# §4.1 での予見**: 「Gate層の判断をOffset層が上書きし、Offset層の計算をCeiling層が無効化する」— **部分的に予見していた**。

**650# 追加発見**: VG boost > 1.0 の14 fills のうち、sell 側は全て ceiling = 0.40 で頭打ち。**VG が sell 側に対して実質的な volatility 防御を提供できていない**。これは 562# §4 の「三層打ち消し問題」の具体例であり、VG boost の意味が ceiling 層で消滅する。

### 3.3 Sidecar stale 93% (649# 修正後もなお)

**562# §1.3.1 での予見**: 「SAC dead 期間の明示が必要」— ✅ 正しく指摘

**650# 追加発見**: 649# でデータ鮮度チェックを分離したが、F02 以降 (11:23–23:06) は全 stale。retrain 自体が成功していない可能性。全27 fills が `decision_path=primary_only`。

**562# B4 (SAC復旧計画) との関係**: B4 は「SAC retrain 復旧が前提」と正しく指摘した。649# は部分修正だが、retrain 成功→signal 更新の全パスが通っていない。

### 3.4 inv_skew の三重無効化

**565# 盲点8 での指摘**: regime_gate_enabled=true による trending 無効化のみ指摘

**650# 追加発見**: neutral_band (0.1) + decay_tau (1800s) + regime_gate が **三重に無効化**。fill 間隔が3-30分の場合、decay 単独で imbalance が neutral_band 以内に収束し、regime_gate に到達する前にスキップされる。結果として **JPY:BTC = 1:99 という極端な偏重でも skew 補正ゼロ**。

**562# / 565# との差分**: 565# は regime_gate のみ指摘。neutral_band と decay_tau の複合効果は誰も分析していなかった。

---

## §4 「渙」の第三段階 — 605#/606#からの継承

605# の三義を 650# 時点で再評価する。

### 4.1 氷の状態

| 605# の「凍った氷」 | 650# 時点 | 評価 |
|---|---|---|
| Pipeline ceiling 100% 飽和 | sell 92% 飽和 (0.40)、buy は改善 | ⚠️ **半溶** |
| 乗算膨張 (9段無制限) | stage_max_mult=2.0 導入済み | ✅ **溶けた** |
| Sidecar active率 ~8% | 7% (2/27 fillsのみfresh) | ❌ **凍ったまま** |
| sell_dynamic_kill 遅行性 | duration短縮済み (600s/900s) | ✅ **溶けた** |
| CV Widen buy側 -3.21bps | CV tighten sell無効化。buy widen は分析データ不足 | ⚠️ **要追跡** |
| entry_gate / SAD / MCB | MCB=true (606#)、SAD=true (606#)、entry_gate=observe | ✅ **溶けた** |
| inv_skew trending無効 | regime_gate + neutral_band + decay_tau 三重無効 | ❌ **凍結深化** |

### 4.2 新たに発見された氷

| 新氷 | 出典 | 深刻度 |
|------|------|--------|
| MCB HALT + open position → inventory risk amplification | 650# RT#2 | HIGH |
| VG boost × ceiling interaction → sell-side defense nullified | 650# 問題4 | MEDIUM |
| preflight_insufficient 46.2% → buy-side structural blockage | 650# cancel analysis | HIGH |
| Sidecar retrain 未成功 → 649# 修正の不十分さ | 650# all primary_only | MEDIUM |

### 4.3 廟（守るべき原則）の健全性

| 原則 | 650# 検証結果 |
|------|-------------|
| **Spread > AS Cost** | ⚠️ Q1 (< 2.4bps) avg PnL = -0.86bps → 低スプレッド帯で原則違反 |
| **Inventory Mean-Reversion** | ❌ inv_skew 三重無効 → 原則が実装上機能していない |
| **Catastrophic Loss Prevention** | ✅ MCB + Final Clamp + age_cap は稼働中。ただし MCB+position 問題あり |

---

## §5 次の一手 — 562#/565# 残存 + 650# 新規の統合アクション

565# の結論「理論はもう十分にある。欠けているのは計測である。」は 650# の RT 分析で部分的に解消された。計測が実施された今、行動に移す段階。

### Tier 0: 即時投入 (YAML/config 変更のみ)

| # | 施策 | 出典 | 変更 | 期待効果 |
|---|------|------|------|---------|
| **T0-1** | inv_skew neutral_band 引下げ | 650# I1 | `neutral_band: 0.1 → 0.05` | inv_skew 発動率 7%→推定30-40%。在庫偏重への最小限の応答 |
| **T0-2** | inv_skew decay_tau 延長 | 650# I1 | `decay_tau_sec: 1800 → 3600` | fill間のimbalance情報保持力を倍増 |

**リスク評価**: ranging sell戦略 (WR 62%) への影響は限定的。inv_skew は sell offset を減らす方向にしか作用しないため、sell の spread capture が若干低下する可能性があるが、buy-side の手仕舞い能力が改善する方が全体PnLに寄与。段階的に neutral_band: 0.05 から試行。

### Tier 1: 短期実装 (コード 1-30行)

| # | 施策 | 出典 | 前提 |
|---|------|------|------|
| **T1-1** | MCB HALT 時 open position 警告ログ | 650# I2 | micro_circuit_breaker.py の HALT 判定部。リスクなし |
| **T1-2** | low-spread sell ガード (トリプル条件) | 650# I3 + 562# P-H | spread<2.0 AND obi>0.25 AND vpin>0.65 → sell offset +0.02。RT#7 (-9.14bps) を捕捉、RT#5 (+3.44bps) は通過 |
| **T1-3** | Regime遷移AS分析セクション追加 | 565# 盲点3 | analyze_fill_logs.py に section_regime_transition_as() を追加。fill 前後のregime変化とPnLの相関 |
| **T1-4** | AS burst autocorrelation 分析 | 565# 盲点4 | analyze_fill_logs.py に section_as_burst() を追加。φ₁ 算出 |

### Tier 2: 中期 (構造変更・検証)

| # | 施策 | 出典 | 前提条件 |
|---|------|------|---------|
| **T2-1** | eDRC α/β 再推定・有効化 | 562# P-G + 561# | 576# インシデント後の安全なパラメータ探索。650# データで backtest 可能 |
| **T2-2** | Sidecar retrain 成功率調査・TTL 調整 | 650# / 562# B4 | retrain_scheduler ログ分析。signal 更新のフルパス確認 |
| **T2-3** | 曜日効果分析 | 565# 盲点5 | 4週分データ蓄積後 |
| **T2-4** | sell ceiling 0.40→VG連動動的拡張 | 650# I4 + 562# §4 | VG boost > 1.5 時のみ ceiling 0.50 に拡張。VG 形骸化の解消 |

### Tier 3: 長期 (アーキテクチャ)

| # | 施策 | 出典 | 判断ポイント |
|---|------|------|-------------|
| **T3-1** | AS Risk Score 統合 (max/RMS) | 562# §4.3 | 乗算多重計上の根本解決。pre_clamp 分布データが十分蓄積後に判断 |
| **T3-2** | MCB HALT 前 position pre-close | 650# I6 | fill_cycle_executor 連携、partial fill。実装HIGH |
| **T3-3** | Kelly lot_sizing 実運用化 | 565# 盲点6 | lot_sizing + kelly を true にしてA/Bテスト |
| **T3-4** | inv_skew 条件付き trending 復活 | 565# 盲点8 | 249# 設計判断の検証後、extreme在庫 (偏り>40pp) でのみ適用 |

---

## §6 562# 分析系タスクの棚卸し

562# §3.1 (A1-A6) の多くが「分析・文書化」タスクだった。650# により一部は自然解消。

| # | タスク | 650# による状態 |
|---|--------|---------------|
| **A1** | 現行設定値棚卸し | **部分解消** — 650# doc に MCB/inv_skew/ceiling/sidecar の設定値記載済み |
| **A2** | OFI-Lite boost 効果計測 | **未実施** — 650# fill records に OFI boost 値があるが boost ON/OFF 比較は未実施 |
| **A3** | SAC dead 期間明示 | **解消** — 650# で 93% stale、全 fill = primary_only を文書化 |
| **A4** | Composite Risk 効果分析 | **未実施** — composite_risk の block 率変化は未定量化 |
| **A5** | unclamped 反実仮想 PnL | **部分解消** — 650# で pre_clamp > 0.60 の3 fills を特定。ただし ceiling=[0.40,0.45,0.50] 別の反実仮想は未実施 |
| **A6** | CV tighten sell無効化検証 | **解消** — 565# で実装済み |

---

## §7 批判的視点 — 562# と650# の両方への自己批判

### 7.1 562# への批判

1. **preflight_insufficient の過小評価**: P-F で「調査項目」としたが、650# では 46.2%＝最大 cancel 原因。560# に引きずられて AS 偏重の分析フレームになった。563# が「参加制約の重さをもっと前に出すべき」と指摘した通り。
2. **inv_skew の見落とし**: 562# は inv_skew に全く言及していない。565# 盲点8 が初めて指摘し、650# で壊滅を確認。562# の「既存実装との照合」が不十分だった。
3. **MCB 有効化後の影響予測なし**: 562# 策定時は MCB = disabled。606# 有効化後の副作用（HALT + open position）を予見できなかった。

### 7.2 650# への批判

1. **n=13 RT の統計的限界**: 13 RT で WR/PF を論じるのは統計検出力が低い。562# §6.1 が指摘した Gate F1 スコアと同様、サンプルサイズ不足を認識した上での解釈が必要。
2. **単日分析のバイアス**: 3/29 は trending→ranging→trending→ranging の大振幅日。典型日ではない可能性がある。565# 盲点5 (曜日効果)、563# (same-SHA純度) の懸念が依然有効。
3. **因果推論の限界**: RT#2 の -14.27bps は MCB × preflight × ceiling の三重複合だが、「どれが主因か」の切り分けは不完全。反実仮想 (MCB なかったら？ preflight なかったら？) が必要。

### 7.3 563# への回帰

563# の根幹的指摘: 「sell 側だけが本丸」という整理は強過ぎる。650# でもこれは有効:
- 650# の 11 sell-entry RT は avg -1.06bps で損失だが、2 buy-entry RT は avg +0.13bps
- しかし **buy が少ないのは preflight_insufficient で buy 注文が構造的にブロックされている** からであり、buy の「品質」を測るサンプルが不足
- つまり「buy 側が良い」のではなく「buy 側のデータが少なすぎて判断できない」

---

## §8 結論

### 562# レビューの総合評価

**即効施策 (P-A/B/C/D) は全て実装され、部分的に効果を発揮している。** 特に P-B (ceiling 引上げ) と P-C (stage max_mult) は pipeline 飽和を 100%→92% (sell) に改善した。しかし **sell 側の飽和は構造的に解消されておらず、562# §4 の「三層打消し問題」は依然として最大のボトルネック**。

**中長期提案 (P-G DRC, §4 統合) は方向性として正しいが、eDRC の α/β 再推定と 576# 安全性確認が前提。** 650# の新発見 (MCB+position, VG+ceiling, inv_skew三重無効) は562#のスコープ外だったが、562# §4 の「同一現象の多重計上」フレームワークで自然に説明できる。

### 「渙」第三段階の指針

605# → 606# → 649# → 650# の流れは:
1. **凍結状態の棚卸し** (605#)
2. **安全な解凍** (606# SAD/MCB、565# ceiling/stage-cap)
3. **解凍後の観測** (650# RT単位分析)
4. **観測に基づく次の解凍候補の特定** (本文 §5)

565# が「理論はもう十分。欠けているのは計測。」と結論した。650# でRT単位の計測が実現した今、次は **「計測結果に基づく段階的パラメータ投入 (T0) → 効果観測 → 構造改善 (T1-T2)」** のサイクルを回す段階。

**即時アクション**: T0-1 / T0-2 (inv_skew パラメータ調整) を投入し、24h 後に inv_skew 発動率と sell RT PnL の変化を計測。

---

*以上*
