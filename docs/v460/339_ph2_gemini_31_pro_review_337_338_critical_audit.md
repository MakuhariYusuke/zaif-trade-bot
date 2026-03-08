# 339_ph2_gemini_31_pro_review_337_338_critical_audit.md

# Gemini 3.1 Pro: 337#（Sell側改善）の自己監査とCodex 338# の総括レビュー

## 1. 結論

Codex（338#）による監査報告において指摘された **「CRITICAL: 	hreshold_offset_bps の符号逆転バグ」は完全に事実であり、稼働中のBotにおいて致命的な誤作動を引き起こしています。**
337#において私たちが良かれと思って導入したBuy/Sell双方への「在庫連動のThreshold緩和（inv_relaxation）」は、数学的に**「緩和」ではなく「極端な厳格化（逆選択への突撃）」として機能している**ことがコードから裏付けられました。

また、Codexが指摘した他アーキテクチャ上の混線（PnL指標の二重管理、複数Gateの並行発火、二重の緩和ルート）も極めて妥当です。337# での「強い言葉」を反省した直後ではありますが、システム工学的に現在の実装は破綻状態にあるため、即座のHotfix対応を要請します。

---

## 2. 致命的バグ（CRITICAL）の検証と裏付け

### §2.1 緩和（Relaxation）が実は厳格化（Tightening）だった事実
ztb/risk/sell_dynamic_kill.py (L510周辺) において、以下の記述が存在します。
`python
# 例: threshold=-0.8, offset=+0.3 → effective=-0.5 (緩和)
if threshold_offset_bps != 0.0:
    threshold += threshold_offset_bps

if rolling_mean < threshold:
    self._cooldown = ... # kill!
`

*   **事実と盲点**: コメントには「effective=-0.5 (緩和)」と書かれていますが、**-0.5 を下回るだけでKillされるようになるのは「緩和」ではなく「著しい厳格化（早期Killの誘発）」**です。
    本来、閾値を -0.8bps から緩和（損失の許容幅を広げる）するのであれば、閾値は -1.1bps 等、**より深い負の値**にならなければなりません。
*   **影響**: これにより、Buy/Sell問わず「在庫に偏りが出たため約定させたい（緩和したい）」はずの局面で、「普段以上に少しのマイナスPnLで即座にBotをKillする」という**完全に逆方向の挙動**が走っています。337# による sell_dynamic_kill_inv_relaxation の導入は、このバグをSell側にも拡散させる結果となりました。

---

## 3. その他の Codex 指摘に対する見解

### §3.1 評価判定基準の混同 (HIGH)
Codex指摘の通り、337#でのログ分析は実際の内部リスク管理用指标 (post_fill_30s_pnl) と再集計用指标 (ev_weighted_pnl) を混同していました。実稼働のキル判定ロジックに対して、外部からアナリスト目線で評価を下す際によく発生する「ML Opsの罠」であり、Codexの批判を全面的に受け入れます。

### §3.2 未整理な二重緩和ルート (HIGH)
337# で sell_dynamic_kill_inv_relaxation を追加しましたが、scripts/v460/lib/cycle_gate_aggregator.py 内部に既に **sell_guard_inv_bypass_threshold (ハードオーバーライド)** が存在することが確認されました。
これにより、「微小緩和（逆方向に作動中）」と「全回避」が異なるレイヤーで並存し、システムの制御フロー（Control Loop）がカオス化しています。防御ルートの階層化（Hierarchy）が欠落しています。

### §3.3 強制取引（Balance Forced）の排除問題 (MEDIUM)
337# で「強制取引の損失をローリングPnLから完全除外する」方針を挙げ（一部コミット済み）ですが、Codexの懸念の通り、**システムが生み出す実コスト（毒）をキル管理から隠蔽することは長期的には極めて危険**です。これは完全除外ではなく、デュアルKPIとして分離監視するか、ウェイトを下げて計算する（Decay）アプローチへ引き戻すべきです。

---

## 4. 即時対応計画（Hotfix Action Plan）

現状のBotは「在庫を補填しなければならない時に限って即死する」という自爆仕様（LivenessとSafetyの完全な矛盾）を抱えて稼働しています。以下のHotfixを**P0**で実行する必要があります。

1. **バグ修正 (P0)**:
   ztb/risk/sell_dynamic_kill.py 等で稼働中の Kill Manager の閾値加算ロジックの符号を反転させる。
   *修正案*: 	hreshold -= threshold_offset_bps （緩和＝損失許容枠の拡張、よりNegative側へ動かす）。また、これによって壊れるであろう（テスト名と定義が逆転している）ユニットテストの修正を追随させる。
2. **二重ルートの整理 (P1)**:
   sell_guard_inv_bypass_threshold（強権発動）と sell_dynamic_kill_inv_relaxation（動的緩和）のどちらかに方針を統一し、役割が重複する実装の一方を破棄する。
3. **強制除外(§6.3)のロールバック (P1)**:
   強制取引を完全にWindowから除外した実装を巻き戻し、別枠のKPI追跡に留める。

Codex（338#）のレビューは冷徹ですが技術的に100%正確です。システムの基礎的数学（符号の向き）が破綻している以上、YAMLのパラメータ調整で試行錯誤しても意味がありません。直ちにコードの修正プロセスへ移行しましょう。
