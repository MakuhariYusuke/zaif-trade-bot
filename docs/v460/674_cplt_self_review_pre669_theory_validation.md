# 674# セルフレビュー — 669#以前の理論有効性検証と体制総点検

- **日付**: 2026-03-31
- **前提**: 672# 深堀り分析結果を判定基準として、P0基線(dfbe3b539eaa, 3/20)以降の全変更を遡及検証
- **対象期間**: 536# 渙 〜 673# (約319コミット)

---

## §0 検証の基本姿勢

672# で確立した判定軸:
1. **Bootstrap CI**: avg PnL30 = [-0.527, +0.071] → H0 (μ=0) 棄却不可
2. **情報理論**: 全 ML 特徴量の MI ≤ 3% (ノイズ水準)
3. **Glosten-Milgrom**: 1500-2500 JPY 帯は α=21.6% (最低)、realized_hs = +1.01 bps (黒字)
4. **時系列**: P0 (3/20) fill rate 51%, NFQ=0 → 現行 fill rate ~7%, NFQ ~30%

**根本問題**: P0 基線から性能が劣化した。「改善」の積み重ねが実は劣化を引き起こしている可能性がある。

---

## §1 変更の分類と検証

### §1.1 有効性が確認された変更 (維持)

| 変更 | 内容 | 根拠 | 判定 |
|------|------|------|------|
| 120# early_exit 無効化 | EE 48件=-345.5bps → 全損失の137% | P0時点で既に無効 | ✅ 確定 |
| 421# execution_final_clamp | ceiling 後の executor 乗数が ceiling を迂回するバグ修正 | 構造欠陥の修正 | ✅ 確定 |
| 438# max_reprice_buy=0 | reprice 反復: 0→-0.73→-2.21 bps の累積損失 | 実測データ | ✅ 確定 |
| 454# micro_timeout | 快速<10s: +0.238bps, 遅延≥30s: -3.750bps | fill-speed vs PnL 相関 | ✅ 確定 |
| 522# inventory_escape/recovery_skew 撤廃 | balance-forcing は structural harm | 522# 分析 | ✅ 確定 (死コード確認済) |
| 549# ewma_input_clamp_bps=5.0 | 単一 AS=-13bps が EWMA 汚染→kill スパイラル | Winsorization 理論 | ✅ 確定 |
| 596# primary_max_consecutive_skip=5 | stale model による全件 skip → BTC=0 death spiral | P0 教訓 | ✅ 確定 |
| 622# SAD/MCB 有効化 | スプレッド異常・ボラ急騰の検出・回路遮断 | 防御機構として正当 | ✅ 確定 |
| 631# floor_bps 3.8→0.38 | 625# の計算ミス修正 (10倍誤り) | 算術修正 | ✅ 確定 |
| 641# CV offset_boost 1.25→1.0 | 7日間実測: buy=-0.56bps, sell=-1.56bps。boost は有害 | 672# MI=2.8% (弱信号) | ✅ 確定 |
| 646# side_min_samples 50→200 | 過学習防止 (n=229 で 24% が定数出力) | 645# degenerate model 事例 | ✅ 確定 |
| 660# halt_sigma 2.0→2.5 | 46回halt (6.3%) は偽陽性過剰 | 感度・特異度トレードオフ | ✅ 確定 |
| 664# deadlock_escape | BTC=0 膠着脱出機構 | 構造的デッドロック解消 | ✅ 確定 |
| 669# max_lot=0.001 | 1mBTC cap で PI/deadlock 軽減 | 実運用要件 | ✅ 確定 |
| 673# cap_bps 3.0→2.0 | 672# 分析で cap が拘束パラメータと判明 | G-M α 分析 | ✅ 確定 (観察中) |

### §1.2 効果不明・検証不十分な変更 (要観察)

| 変更 | 内容 | 疑義 | 判定 |
|------|------|------|------|
| 506# sell_age_cap_sec=25 | 30-50s バケットの -158.73JPY 回避 | サンプル数限定。但し害は少ない | ⚠️ 維持 (低リスク) |
| 535# preemptive_sell_kill | CV velocity 持続時の sell 事前ブロック | CV MI=2.8% だが防御論理は健全 | ⚠️ 維持 (防御的) |
| 540# composite_risk 1.5→1.0 | 2ゲート同時で block | 副作用不明だが安全側 | ⚠️ 維持 |
| 544# sidecar max_boost 0.15→0.20 | SAC 影響力拡大 | SAC α ≈ 8% (TTL補正後)。効果小 | ⚠️ 維持 (低Impact) |
| 546# sidecar shaping quadratic | 弱signal抑制、強signal集中 | 理論的に健全だが実測差不明 | ⚠️ 維持 |
| 654# toxic_sell_veto | G-M compound guard | 後述 §2.1 で詳細検討 | ⚠️ 調整必要 |

### §1.3 復元を検討すべき変更 (問題あり)

| 変更 | 現行値 | P0値 | 問題 | 推奨 |
|------|--------|------|------|------|
| **624# ATR floor 導入** | enabled + cap=2.0bps | 無し (固定700JPY) | **NFQ の根本原因**。cap 引下げで対処中だが本質は floor 自体の必要性 | 673# cap=2.0 で観察。効果不十分なら 1.5 へ |
| **630# VG velocity 12→6** | 6.0 bps | 12.0 bps | 閾値半減は過激。VG 発動頻度が大幅増加 → fill rate 低下の一因 | **8.0 に緩和** を検討 |
| **630# sell_velocity_skip 6→4** | 4.0 bps | 6.0 bps | 同上。4bps は日常的な velocity で常時発動リスク | **5.0 に緩和** を検討 |
| **630# trend_threshold 0.5→0.20** | 0.20% | 0.5% | 低閾値でほぼ常時 trending 判定。ranging 判定が確認不足 | 実測 regime 分布を確認してから判断 |
| **634# sell_ranging_offset=0.5** | 0.5 | 無し (0) | sell+ranging に +0.5 ペナルティは重い。1500-2500帯でも排除される可能性 | 672# α 分析と照合し **0.3 に緩和** 検討 |
| **641# max_skip_rate 0.3→0.4** | 0.4 | 0.3 | skip 率上限引上げ → **fill rate 低下直結**。672# CI が 0 を含むなら skip は PnL 改善に寄与せず | **0.3 に復元** |
| **565# offset_ceiling_sell 0.20→0.40** | 0.40 | 0.20 | ceiling 2倍拡大。sell offset が 0.40 まで到達 → 約定不能な乖離 | **0.30 に戻す** (中間値) |
| **565# offset_ceiling_buy 0.25→0.35** | 0.35 | 0.25 | 同上 buy 側。pre_clamp p50=0.31 で clamp が常態化していた | **0.30 に戻す** (中間値) |
| **519# sell_kill window 50→30** | 30 | 50 | 反応速度向上は良いが、30 fills は統計的に不安定 | ⚠️ 維持 (35-40 も検討) |
| **540# sell_kill max_duration 1800→600** | 600s | 1800s | 535# pre-emptive sell kill 前提の短縮だが、kill 発動時確実に復帰必要 | ⚠️ 維持 (900s も検討) |

### §1.4 誤った理論に基づく変更 (認識すべき)

| 変更 | 問題 | 672# による否定 |
|------|------|----------------|
| 632# atr_mult 2.0→1.2 | 「Roll proxy 循環緩和」が目的だが cap_bps が拘束なので mult 変更は無効 | **効果ゼロ** (673# で確認) |
| 645# sell model 無効化 | degenerate model 除去は正しいが「unified model で代替」は 672# MI=2.2% で否定 | モデル自体が無効 |
| 660# trending_up/down skip offsets | n=18, n=10 の微小サンプルで閾値変更 | サンプル不足。CI 計算なし |
| 641# buy/trending_down hard_skip_mult=4.0 | n=40 で「唯一の収益バケット」と判断 | **過剰適合リスク**。分散大 |

---

## §2 構造的問題の深堀り

### §2.1 toxic_sell_veto と 672# 矛盾

654# で導入した toxic_sell_veto の条件:
- sell + spread < 2.3bps + OBI > 0.25 + VPIN > 0.65

**672# が示すこと**:
- 1500-2500 JPY 帯 (≈ 1.2-2.0 bps) は **α 最低 (21.6%)、realized_hs 黒字**
- 2.3bps 閾値は 1500-2500 帯に重なる → **最良帯の sell を排除する可能性**

**判定**: toxic_sell_veto は spread_bps=2.3 の閾値が広すぎる。**1.5 に縮小** するか、OBI/VPIN 条件を厳格化 (OBI>0.4, VPIN>0.75) して false positive を減らすべき。

### §2.2 narrow_spread_boost と 672# 矛盾

183# で導入:
- spread < 2000 JPY → buy offset ×2.0, sell offset ×2.5
- skip_gate narrow_spread_offset +0.2

**672# が示すこと**:
- 0-1500 帯: α=21.2% (最低)、AS cost = +0.07 (ほぼ中立)
- 狭スプレッド = 情報非対称性が低い = **積極的に約定すべき**

**判定**: narrow_spread_boost は **672# と方向が逆**。狭スプレッドでは offset を縮小 (fill 促進) すべきなのに拡大している。
- **narrow_spread_boost_buy: 2.0 → 1.0** (ブースト無効化)
- **narrow_spread_boost_sell: 2.5 → 1.5** (最低限の防御のみ)
- **skip_gate_narrow_spread_offset: 0.2 → 0.0** (ペナルティ撤廃)

### §2.3 SkipGate の価値

672# の結論: 全特徴量 MI ≤ 3%。SkipGate の予測精度はノイズと同等。

しかし、SkipGate は複数の役割を持つ:
1. **ML 予測** → 672# で否定 (MI ≤ 3%)
2. **AS binary 分類** → Cohen's d = -2.08, MI = 21% (有効)
3. **velocity/regime/hour ルールベース** → 実コード上 offset 修正の主要チャネル

**判定**: SkipGate 自体の廃止は不要。ML モデルの予測値 (score) への依存を減らし、ルールベース (velocity skip, hour offset, AS binary) を主軸にする方向が正しい。

### §2.4 hard_skip_utc_hours 廃止 (623#) の妥当性

P0: `hard_skip_utc_hours: [16, 21]` (JST 01時, 06時 を全停止)
623#: `hard_skip_utc_hours: []` → hour_ceiling_mult + sell_hour_offset_boost に委譲

**672# が示すこと**: AS rate は時間帯で大きく変動 (UTC14: AS 100%, UTC21: PnL -125.8bps/日)。
しかし 672# のデータ期間は 623# 以降なので、hard_skip 廃止の影響は「廃止後のデータ」で測定されている。

**判定**: 623# の委譲先 (ceiling_mult, hour_offset_boost) が UTC16, UTC21 をカバーしているため、完全な復元は不要。ただし **UTC21 (JST06) の ceiling_mult=1.5 は過弱**。この時間帯は PnL -125.8bps/日であり、**ceiling_mult=2.5 以上** または hard_skip 復元を検討。

### §2.5 370# M2-M5 モジュール群の意義

bayesian_regime, sigma_clustering, glft_dynamic_k, vpin_vol_sync — 全て enabled=true で稼働中。

**672# が示すこと**: これらのモジュールが生成する特徴量の MI はいずれも ≤ 3%。

**但し**: これらは特徴量生成だけでなく、adaptation_engine への直接介入 (sigma_clustering → interval/offset 調整) や maker_price への直接介入 (glft_dynamic_k → AS δ* 計算) も行う。

**判定**: 出力が直接 maker_price に影響するものは MI で測れない (offset パイプラインを通らない)。**glft_dynamic_k は maker_microstructure.py L235 で AS δ* 計算に使用** → 効果検証はスプレッド分布の比較が必要。mi ベースの否定は早計。

---

## §3 優先順位付きアクションプラン

### P0 (即時実施 — 次回コミット)

| # | アクション | 根拠 | 影響 |
|---|-----------|------|------|
| A | **max_skip_rate 0.4 → 0.3** 復元 | 672# CI が 0 を含む → skip は収益改善に寄与せず。fill rate 直結 | fill rate +数% |
| B | **narrow_spread_boost_buy 2.0→1.0, sell 2.5→1.5** | 672# α 最低帯を boost で排除は逆効果 | 狭スプレッド fill 促進 |
| C | **skip_gate_narrow_spread_offset 0.2→0.0** | 同上 | SkipGate 狭帯ペナルティ撤廃 |

### P1 (データ確認後)

| # | アクション | 根拠 | 条件 |
|---|-----------|------|------|
| D | **VG velocity 6→8** | 630# の半減は過激。fill rate 回復 | regime 分布確認後 |
| E | **sell_velocity_skip 4→5** | 同上 | 同上 |
| F | **offset_ceiling_sell 0.40→0.30** | 474# の 0.20 は過小だが 0.40 は過大 | offset 分布確認後 |
| G | **offset_ceiling_buy 0.35→0.30** | 同上 | 同上 |
| H | **toxic_sell_veto spread 2.3→1.5** | 672# と矛盾する帯域ブロック | veto 発動ログ確認後 |
| I | **sell_ranging_offset 0.5→0.3** | 重すぎるペナルティ | ranging fill rate 確認後 |
| J | **UTC21 ceiling_mult 1.5→2.5** | PnL -125.8bps/日の最危険帯 | 安全側変更のため即実施可 |

### P2 (構造改善)

| # | アクション | 根拠 |
|---|-----------|------|
| K | **SkipGate ML score 依存度低減** | MI ≤ 3%。ルールベースを主軸に |
| L | **cap_bps 2.0→1.5 段階引下げ** | 673# 効果確認後 |
| M | **ev_emergency_skip -8.0→-5.0** | 670# P0 提案未適用 |

---

## §4 P0 基線との対照表 — 復元 vs 維持 の最終判定

| パラメータ | P0 値 | 現行値 | 判定 | 理由 |
|-----------|-------|--------|------|------|
| min_spread_jpy | 700 | 100 | **維持** | ATR floor が主防衛に昇格済み |
| min_spread_atr_* | 無し | enabled, cap=2.0 | **維持** | 673# で cap 調整済み。構造として正しい |
| hard_skip_utc_hours | [16,21] | [] | **維持** (ceiling 強化) | 委譲先が機能。UTC21 ceiling は強化 |
| max_skip_rate | 0.3 | 0.4 | **復元 → 0.3** | skip は PnL 改善に寄与せず |
| offset_ceiling_buy | 0.25 | 0.35 | **緩和 → 0.30** | 0.25 は過小、0.35 は過大 |
| offset_ceiling_sell | 0.20 | 0.40 | **緩和 → 0.30** | 0.20 は過小、0.40 は過大 |
| side_offset.sell | 0.18 | 0.14 | **維持** | 506# データ根拠あり |
| velocity_threshold_bps | 12.0 | 6.0 | **緩和 → 8.0** | 半減は過激 |
| sell_velocity_skip | 6.0 | 4.0 | **緩和 → 5.0** | 同上 |
| sell_dynamic_kill.window | 50 | 30 | **維持** | 反応速度の改善は正当 |
| sell_dynamic_kill.max_duration | 1800 | 600 | **維持** | pre-emptive kill 前提 |
| balance_freeze_cycles | 3 | 1 | **維持** | 641# 検証済み |
| kelly.enabled | true | false | **維持** | lot_sizing=false との整合 |
| narrow_spread_boost_buy | 2.0 | 2.0 | **縮小 → 1.0** | 672# α 最低帯を排除は逆効果 |
| narrow_spread_boost_sell | 2.5 | 2.5 | **縮小 → 1.5** | 同上 (最低限防御) |
| inv_skew.neutral_band | 0.1 | 0.05 | **維持** | 在庫管理精度向上 |
| inv_skew.decay_tau | 1800 | 3600 | **維持** | 長期記憶は ranging 保全 |
| trending_up skip offset | -0.1 | 0.3 | **要観察** | n=18 のため不確実 |
| composite_risk | 1.5 | 1.0 | **維持** | 安全側 |

---

## §5 自己批判

### 何が間違っていたか
1. **Parameter golf**: 閾値の微調整を繰り返したが、672# MI≤3% が示すように ML 予測は無効。閾値調整の効果も同様にノイズレベル
2. **累積的過修正**: 個別には合理的に見える変更が、積み重なると fill rate 51%→7% の大幅劣化を引き起こした。「千票の切り傷」
3. **確証バイアス**: 各変更は「この変更で PnL 改善」の物語で正当化されたが、Bootstrap CI が 0 を含むということは、**どの変更にも統計的有意性がない**
4. **小サンプル依存**: 660# (n=18,10)、641# (n=40) で重大な閾値変更を実施。最低でも n>100, できれば n>500 が必要
5. **fill rate への無関心**: fill rate が 51%→7% に低下する過程で、各スプリントは「PnL 品質」に注目し、「数量」を軽視した。MM にとって fill rate は生命線

### 何が正しかったか
1. **構造欠陥の修正** (421#, 549#, 596#, 631#) は明確な効果
2. **有害機能の無効化** (120# early_exit, 522# balance-forcing) は確実に正の貢献
3. **防御機構の段階的導入** (SAD, MCB, deadlock_escape) は安全性向上
4. **672# 分析による根本的再考** — cap_bps 発見は本質的

### 教訓
- **変更は減算優先**: 「足す」より「引く」方が安全。機能追加ではなく、有害機能の除去で改善
- **統計的有意性**: n=18 で判断しない。少なくとも CI を計算してから
- **fill rate 目標**: MM では fill rate × avg PnL が収益。どちらかの最適化は不十分
