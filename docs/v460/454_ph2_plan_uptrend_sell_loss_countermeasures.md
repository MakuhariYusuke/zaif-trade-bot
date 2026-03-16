# 454# 上昇トレンドにおける Sell 損失対策 — 包括設計メモ

**種別**: plan  
**日付**: 2026-03-17  
**関連**: 000# (プロジェクト方針), 337# (Sell-side degradation), 429# (三層アーキテクチャ), 440# (Toxicity Veto 棄却), 447# (新パラダイム提案), 450# (残課題), 453# (Micro-timeout 実装)  
**ステータス**: レビュー待ち（実装未着手）

---

## §0 Executive Summary

fill_test ライブログ分析により、BTC/JPY が **5 日間で +4.6%** 上昇（3/12: 11,192k → 3/16: 11,709k）する中、Bot が売り続けて利益を逸失している問題を特定。

**根本原因は 3 層**:

| # | 原因 | データ | 影響 |
|---|------|--------|------|
| R1 | Micro regime 盲点 | 40 分 lookback × 0.5% 閾値 → regime=ranging **91.6%** | 日足レベルのトレンドが検知不能 |
| R2 | Macro regime 感度不足 | slope_threshold=1.0 bps/min vs 実測 slope_5m 平均 ≈0.1 → macro_trend=None **70%** | マクロも機能しない |
| R3 | Macro→Sell 保護経路欠如 | macro_trend=UP でも sell offset boost/suppress する**コードが存在しない** | 検知しても保護不能 |

**結果**:
- Sell 側 AS (Adverse Selection) 率: 22% → **38%** （上昇日ほど悪化）
- Sell pnl120 週平均: **-0.11 bps** （Buy +0.18 bps 対比）
- 337# で指摘された sell-side 劣化パターンの再現

本ドキュメントでは **A-E (YAML 調整)** + **F-J (コード変更・新機構)** の 10 案を深堀りし、000# で示された三層アーキテクチャとの整合性を踏まえた推奨実装順を示す。

---

## §1 問題の定量分析

### 1.1 価格推移と fill_records の乖離

```
日付       midprice(k)  regime分布              sell_fill数  sell_AS率  sell_pnl120(bps)
3/12(Wed)  11,192       ranging=94% trend=6%     ~180        22%        -0.04
3/13(Thu)  11,350       ranging=90% trend=10%    ~165        28%        -0.08
3/14(Fri)  11,480       ranging=92% trend=8%     ~190        35%        -0.15
3/15(Sat)  11,580       ranging=91% trend=9%     ~145        33%        -0.12
3/16(Sun)  11,709       ranging=91% trend=9%     ~120        38%        -0.18
```

日次上昇率は 0.8-1.4%/day だが、40 分窓 × 0.5% 閾値では 98.6% のサイクルで検知不能（3/14 実測: |trend_pct|>0.5% は全体の 1.4%）。

### 1.2 Regime 閾値と実測分布の乖離

**Micro regime (trend_threshold_pct)**:

| 閾値 | 検知率(3/14) | 検知率(3/15) | 検知率(3/16) |
|------|-------------|-------------|-------------|
| 0.5% (現行) | 1.4% | 1.5% | 1.2% |
| 0.4% | 3.1% | 3.4% | 2.8% |
| 0.3% | 8.2% | 8.7% | 7.4% |
| 0.2% | 18.5% | 19.2% | 17.1% |
| 0.15% | 28.1% | 29.0% | 26.3% |

**Macro regime (slope_threshold)**:

| slope閾値 | slope_5m 検知率 | slope_15m 検知率 |
|-----------|----------------|-----------------|
| 1.0 bps/min (現行) | 1.1-4.9% | <1% |
| 0.5 bps/min | 10-11% | 3-5% |
| 0.3 bps/min | 20-25% | 8-12% |
| 0.1 bps/min | 50-55% | 25-30% |

### 1.3 ガード発火統計

```
guard_fire_counts (3/12-3/16 累計):
  gate_sell_dynamic_kill     = 921
  gate_buy_dynamic_kill      = 491
  gate_ranging_low_vol_skip  = 398
  gate_toxicity_budget       = 112
  gate_cross_venue_lead_lag  = 67
```

sell_dynamic_kill が最多発火（921 回）。threshold=-0.5 で trending_up も同値のため、上昇トレンドでも同じ感度で sell が抑制される。ただしこれは **売りを止める側** なので問題の方向性とは異なる — 問題は「止めるべき不利な売りが止まらず、止めなくてよい有利な売りが止まる」という非対称性にある。

---

## §2 案 A-E: YAML 調整のみで可能な施策

### 案 A — Micro regime 閾値引き下げ

**内容**: `regime.trend_threshold_pct` を 0.5% → 0.25-0.3% に引き下げ

**メカニズム**: 40 分窓で ±0.3% の傾きがあれば trending_up/down 判定。trending_up 時は `skip_sell_trending=true` により sell 注文がスキップされる（現在この機構は存在するが、発火率が極めて低い）。

**期待効果**:
- trending_up 検知: 1.4% → 8-9% のサイクルで発動
- 上昇トレンドの 1/10 程度は保護可能

**リスク**:
- 日中のノイズ的な上昇でも trending_up 判定 → buy 機会の逸失（buy_dynamic_kill の trending_up 閾値が -1.5 に緩和されているが）
- trending_down 誤検知も増加 → 売るべき局面で売れない可能性
- **根本問題**: 40 分窓では日足レベルのトレンドを構造的に捉えられない

**評価**: ★★☆☆☆ — 効果限定的。窓を広げない限り日足トレンドには届かない。ただし最も実装コストが低い。

---

### 案 B — Macro regime 閾値引き下げ

**内容**: `macro_regime.slope_threshold` を 1.0 → 0.3-0.5 bps/min に引き下げ

**メカニズム**: 5 分足 / 15 分足の OLS 回帰スロープで macro_trend を判定。閾値を下げることで WEAK_UP/STRONG_UP が出やすくなる。

**期待効果**:
- macro_trend 検知: 現行 1-5% → 10-25% に改善
- compose_regimes() でミクロ/マクロ矛盾チェックが有効化

**リスク**:
- 0.3 bps/min は **かなり弱い傾き** — ノイズでも検知してしまう
- 15 分足 slope はより安定だが反応は遅い → 急反転キャッチが遅延
- **根本問題 (R3)**: macro_trend=UP を検知しても **sell を抑制/offset を上げるコードが存在しない**。単独では効果ゼロ。案 F と併用必須。

**評価**: ★★☆☆☆ — 案 F なしでは no-op。閾値調整の方向性は正しいが、コード変更が前提。

---

### 案 C — trending_sell_offset_boost_factor 強化

**内容**: `trending_sell_offset_boost_factor` を 1.5 → 2.5-3.0 に引き上げ

**メカニズム**: micro regime が trending_up/down 判定時、sell offset に乗数をかけて不利約定から守る。440# で実装済の regime-side offset 非対称化の延長。

**期待効果**:
- trending_up 検知時（現行 1.4%）の sell offset が 1.5x → 2.5-3.0x に拡大
- より広い spread を要求 → 不利約定の AS 率低下

**リスク**:
- trending_up 検知率が 1.4% のままでは適用機会が極小 — **案 A との併用必須**
- 乗数が 3.0x だと sell 側が「事実上スキップ」に近くなる（offset が大きすぎて約定しない）
- skip_sell_trending=true と機能重複 → 明確な役割分担が必要

**評価**: ★★☆☆☆ — 案 A と同時適用で初めて意味を持つ。単独では効果なし。

---

### 案 D — sell_dynamic_kill 閾値の非対称化

**内容**: `sell_dynamic_kill.regime_thresholds.trending_up` を -0.5 → -0.3 に引き上げ（より厳しく）

**メカニズム**: sell_dynamic_kill は EWMA ベースの rolling PnL が閾値を下回ると sell をキルする。trending_up 時に閾値を引き上げることで「少しでも sell が不利ならすぐ止める」挙動にする。

**期待効果**:
- 上昇トレンド時の売り損失を早期検知・自動停止
- 既存の EWMA インフラを利用、新規コード不要

**リスク**:
- 再び案 A の「trending_up 検知率 1.4%」問題に直面
- -0.3 は相当タイト → 正常な sell でも EWMA がたまたま下振れすれば kill される
- kill 後の recovery ロジック（inv_relaxation）との整合性
- 337# で指摘: kill が頻発すると「PnL EWMA が回復しない → 永久 kill」の悪循環

**評価**: ★★★☆☆ — 方向性は正しいが trending_up 検知率が前提。337# の悪循環リスクを考慮すると、PnL EWMA のリセット機構とセットで検討すべき。

---

### 案 E — buy_dynamic_kill trending_up 緩和

**内容**: `buy_dynamic_kill.regime_thresholds.trending_up` を -1.5 → -2.0 以下に

**メカニズム**: 上昇トレンド時は buy が有利なので kill 閾値を緩和し、buy 継続を優先。

**期待効果**:
- trending_up 時の buy 約定数増加 → inventory 蓄積 → 上昇の恩恵享受
- 不利な売りの減少と合わせて全体 PnL 改善

**リスク**:
- トレンド判定ミス時に buy over-exposure → 反転時の損失拡大
- inv_relaxation との相互作用（既に buy_trending_up は enable_relaxation=true）
- 案 A と同じ検知率問題

**評価**: ★★☆☆☆ — 案 A-D と同時適用で補完的に機能するが、単独インパクトは小さい。

---

## §3 案 F-J: コード変更を伴う構造的施策

### 案 F — Macro→Sell Offset Boost / Suppress 経路の新設 ★推奨

**内容**: macro_trend=WEAK_UP/STRONG_UP 時に sell offset を自動ブーストし、STRONG_UP 時は sell を完全スキップするコード経路を新設。

**アーキテクチャ変更**:

```
fill_cycle_executor.py :: run_single_cycle()
  └─ macro_regime_detector.detect()
       ├─ STRONG_UP → sell_skip = True (skip_sell_trending と同等)
       ├─ WEAK_UP   → sell_offset *= macro_sell_boost_factor (config)
       ├─ NEUTRAL   → no change
       ├─ WEAK_DOWN → buy_offset *= macro_buy_boost_factor (config)
       └─ STRONG_DOWN → buy_skip = True

pre_order_adjustments.py :: _apply_offset_multiplier()
  └─ macro_trend 基づく新 multiplier パスを追加
```

**YAML config 追加案**:
```yaml
macro_regime:
  sell_boost_weak_up: 2.0        # WEAK_UP 時の sell offset 乗数
  sell_skip_strong_up: true      # STRONG_UP 時の sell 完全スキップ
  buy_boost_weak_down: 2.0       # WEAK_DOWN 時の buy offset 乗数
  buy_skip_strong_down: true     # STRONG_DOWN 時の buy 完全スキップ
```

**000# との整合**:
- 000# §0.1 三層分離の「Execution 層」に属する変更。Alpha/Safety 層は不変。
- regime 検知（Alpha 寄り）と offset 適用（Execution）のインターフェースを明確にする好機。

**期待効果**:
- R3 (Macro→Sell 保護経路欠如) を構造的に解決
- 案 B と組み合わせることで macro_trend 検知率 10-25% × 保護動作 → 実効的なカバレッジ確保

**リスク**:
- macro_trend のラグ（5 分足 / 15 分足基準）→ 急反転への対応遅延
- STRONG_UP/WEAK_UP の境界がフラットだと振動（頻繁な ON/OFF）→ ヒステリシス機構が必要
- テスト必要量: fill_cycle_executor + pre_order_adjustments + macro_regime 3 ファイル

**337# との関連**: 337# は sell_dynamic_kill 閾値の非対称化で同種の問題に対処したが、offset ブーストは未実装だった。本案は 337# の延長線上にある自然な進化。

**実装コスト**: 中（60-100 行の新規コード + YAML 追加 + テスト）

**評価**: ★★★★★ — R3 を直接解決する唯一の案。案 B と組み合わせて最大効果。

---

### 案 G — Inventory 非対称化（Macro 連動） ★推奨

**内容**: macro_trend=UP 時に target_inventory を正方向にシフトし、BTC 保有を促進。

**メカニズム**: 000# §2.4 の inventory 管理思想（中立ポジション維持）を拡張し、マクロトレンドに応じてターゲットを動的に変動させる。

```
通常時:  target_inv = 0 (中立)
WEAK_UP:   target_inv = +0.001 BTC (微量ロング傾斜)
STRONG_UP: target_inv = +0.003 BTC (積極ロング傾斜)
```

**offset への影響**:
- inv > target_inv の場合のみ sell offset が縮小（売りやすく）
- inv < target_inv では sell offset が拡大（売りにくく）
- buy 側は逆の動作

**000# との整合**:
- 000# §0.1 の「マーケットメーカーは方向性ベットを避ける」原則からの **意図的逸脱**
- ただし 447# (Gemini) の「Inventory Sponging」提案と同根 — MM が緩やかなトレンドフォロー成分を持つことは学術的にも支持されている (Guéant et al. 2013)
- リスク上限は config で硬く制限すべき

**リスク**:
- トレンド誤判定 → 反転時に inventory が不利方向に偏る
- inventory 硬制限 (max_inv) との干渉
- 「MM なのにポジションを持つ」は哲学的転換 → 慎重な段階的導入が必要

**実装コスト**: 中-高（inv management ロジック変更 + 新 config + fill_cycle_executor のターゲット計算修正）

**評価**: ★★★★☆ — 効果は高いが MM 原則からの逸脱度が大きい。案 F を先行実装後に検証データを見て判断すべき。

---

### 案 H — Micro-timeout トレンド連動強化

**内容**: 453# で実装済の micro-timeout (TIF emulation) を、macro_trend に連動させて売り側のタイムアウトを動的に短縮。

**メカニズム**:
- 通常: sell timeout = 3.0s, buy timeout = 3.0s
- macro_trend=WEAK_UP: sell timeout = 1.5s (早めにキャンセル → 不利約定回避)
- macro_trend=STRONG_UP: sell timeout = 0.5s (事実上即キャンセル)

**453# との差分**:
- 453# は固定 timeout（config 値）。本案は macro_trend に応じた動的調整。
- `MicroTimeoutManager` の `get_timeout()` メソッドに macro_trend パラメータを追加。

**期待効果**:
- 上昇トレンド時の sell 約定を素早くキャンセル → AS 率低下
- timeout による自然なスキップなので skip_sell_trending より柔軟

**リスク**:
- 453# が disabled-by-default の段階 → まず 453# 本体の有効化・検証が先
- timeout が短すぎると「出す意味がない注文」になる → 最低限の fill 確率は確保すべき

**実装コスト**: 低（453# 既存コードに条件分岐追加のみ）

**評価**: ★★★☆☆ — 453# 有効化後の自然な拡張。即効性はあるが前提条件あり。

---

### 案 I — Spread Shadowing（板厚基づく動的 spread）

**内容**: 447# (Gemini) で提案された「他の大口板に shadow して spread を設定」する手法を、上昇トレンド時の sell 側に特化適用。

**メカニズム**:
- L5 板深度データ（442# で取得基盤構築済）から sell 側の最良気配の厚みを監視
- 大口 sell が存在する場合（厚い板）→ その手前に自分の sell を出す（shadow）
- 大口 sell が薄い場合（上昇圧力強い）→ sell offset を大幅拡大 or スキップ

**000# との整合**:
- 000# §2.2 の「マイクロストラクチャ情報活用」に直接合致
- Execution 層の高度化として位置づけ可能

**リスク**:
- Coincheck の板データ更新頻度に依存（L5 のみ）
- 板操作（見せ板）に対する脆弱性
- 実装量が大きい（板解析ロジック新設 + offset 連携）

**実装コスト**: 高（新規モジュール + 板データパイプライン拡張）

**評価**: ★★★☆☆ — 理論的には最も洗練されているが実装コストが大。中期的な検討事項。

---

### 案 J — Regime Window 拡張 / Multi-Timeframe Fusion

**内容**: micro regime の lookback 窓を 40 分 → 2-4 時間に拡張、または複数タイムフレームの加重合成。

**メカニズム**:
```
regime_short  = OLS(window=40min)  → 短期ノイズ検知
regime_medium = OLS(window=2h)     → 時足トレンド
regime_long   = OLS(window=8h)     → 日足トレンド相当

composite_trend = weighted_vote([short, medium, long], weights=[0.2, 0.4, 0.4])
```

**期待効果**:
- R1（40 分窓の構造的限界）を根本解決
- 日足トレンドが composite で検知可能に

**リスク**:
- 長い窓は反応が遅い → 急反転への対応能力低下
- メモリ使用量増加（価格データ蓄積量: 8h × 1秒 = 28,800 データポイント）
- OLS 回帰の 8h 窓は計算コスト非自明 → EMA 近似が現実的
- 既存 macro_regime.py (5m/15m slope) との役割重複を整理する必要

**000# との対応**:
- 000# §1.3 「特徴量を先に検証せよ」→ 新窓サイズの有効性をオフラインで検証してから導入
- 429# の Sidecar 思想とも親和性高い（α シグナルの多タイムフレーム化）

**実装コスト**: 中-高（regime_detector リファクタリング + 新 config + 検証フレームワーク）

**評価**: ★★★★☆ — R1 直接解決策。ただし検証フェーズが必須で即効性は低い。

---

## §4 000# プロジェクト方針との整合性チェック

### 4.1 三層分離（000# §0.1）

| 案 | Alpha 層への影響 | Execution 層への影響 | Safety 層への影響 |
|----|-----------------|--------------------|--------------------|
| A-E (YAML) | 閾値変更のみ | なし | sell_dynamic_kill 閾値変更 |
| F | なし | **offset パイプライン拡張** | なし |
| G | なし | **inventory ターゲット変更** | inv 制限との整合要確認 |
| H | なし | micro-timeout 条件分岐 | なし |
| I | なし | **板解析 + offset 連動** | なし |
| J | **regime 検知拡張** | なし | regime 変更の波及 |

案 F/G は Execution 層、案 J は Alpha 層の変更。三層分離は維持可能。

### 4.2 「特徴量を先に検証せよ」（000# §1.3）

- 案 B/J は新しい特徴量（slope 閾値 / 新窓サイズ）を導入 → **オフライン検証必須**
- 案 F は既存 macro_trend を活用 → 検証済み特徴量の延長で可
- 案 G は inventory ターゲットという「特徴量」ではなく「制御変数」→ A/B テストで live 検証

### 4.3 「SAC は Sidecar であって Driver ではない」（429#）

- 本ドキュメントの全案は SAC に非依存。regime 検知 + offset 制御の範囲で完結。
- SAC の出力（offset_ratio）は引き続き外部入力として受け取るが、sell 保護ロジックは SAC の判断を上書きする safety 的位置づけ。

### 4.4 短期高収益性 vs 中長期健全性（AGENTS.md）

| 時間軸 | 推奨施策 | 理由 |
|--------|---------|------|
| 即効（1-2 日） | A + B + F | YAML + 最小コード変更で sell 保護効果発揮 |
| 短中期（1-2 週） | D + H | sell_dynamic_kill 非対称化 + micro-timeout 連動 |
| 中長期（1 ヶ月〜） | G + J | inventory 戦略転換 + regime 基盤刷新 |
| バックログ | I | 板データ基盤が整ってから |

---

## §5 推奨実装順序

### Phase 1: 即効性重視（1-2 日）

| 順 | 案 | 変更量 | 依存関係 |
|----|----|--------|---------|
| 1-1 | B | YAML のみ | なし |
| 1-2 | F | 60-100 行 + YAML | B の閾値設定に依存 |
| 1-3 | A | YAML のみ | F と同時でも可 |

**根拠**: B で macro_trend を有効化し、F でそれを sell 保護に接続する。これだけで R2+R3 を同時解決。A は補助的に micro 感度も上げる。

**検証指標**:
- macro_trend=UP/STRONG_UP の発生率: 目標 15-25%
- sell AS 率: 38% → 目標 25% 以下
- sell pnl120: -0.11 bps → 目標 0 bps 以上
- 全体 PnL: 悪化しないこと

### Phase 2: 防御強化（1-2 週）

| 順 | 案 | 変更量 | 依存関係 |
|----|----|--------|---------|
| 2-1 | D | YAML のみ | A の trending_up 検知率に依存 |
| 2-2 | H | 20-30 行 | 453# 有効化が前提 |
| 2-3 | E | YAML のみ | A の trending_up 検知率に依存 |

**根拠**: Phase 1 の効果観測後、sell_dynamic_kill 非対称化と micro-timeout 連動で多層防御を構築。

### Phase 3: 構造的進化（1 ヶ月〜）

| 順 | 案 | 変更量 | 依存関係 |
|----|----|--------|---------|
| 3-1 | J | 中-高 | オフライン検証完了が前提 |
| 3-2 | G | 中-高 | J の regime 精度に依存 |
| 3-3 | I | 高 | 板データ基盤整備が前提 |

**根拠**: regime 基盤を刷新してから inventory 戦略を転換し、最終的に板データ活用で Microstructure Edge を完成させる。

---

## §6 リスク評価

### 6.1 最大リスク: トレンド反転時の inventory 偏り

- 案 F/G は「上昇トレンド時に sell を抑制/inventory を偏らせる」→ **反転時に大きな含み損**
- **緩和策**:
  - Circuit Breaker の即時発動条件を確認・強化
  - macro_trend=DOWN 転換時の emergency sell ロジック
  - inventory 上限の hardcap は変更しない

### 6.2 regime 判定ノイズ

- 閾値引き下げ（A/B）によるノイズ増加 → 頻繁な trending/ranging 切り替え
- **緩和策**:
  - ヒステリシス機構（trending に入る閾値と出る閾値を分離）
  - 最低持続時間（例: trending 判定後 5 分は維持）

### 6.3 sell_dynamic_kill との干渉

- 337# で分析された「kill 悪循環」: sell kill → PnL 回復なし → 永久 kill
- **緩和策**:
  - EWMA リセットタイマー（48h 無リセットなら強制リセット — 337# §3 推奨）
  - inv_relaxation の sell_trending_up 設定の再検証

### 6.4 テスト不足リスク

- fill_cycle_executor.py のテストカバレッジが不十分な場合、offset 計算のバグが本番直撃
- **緩和策**:
  - 案 F 実装時に unit test を同時作成（macro_trend 全 enum パターン × sell/buy）
  - dry-run モードでの 24h 検証後に本番投入

---

## §7 他ドキュメントとの相互参照

| ドキュメント | 関連する本案 | 参照ポイント |
|------------|------------|-------------|
| 000# | 全案 | 三層アーキテクチャ、特徴量検証原則 |
| 337# | D, F | sell-side degradation パターン、kill 悪循環リスク |
| 429# | F, G, J | Sidecar 思想、SAC 非依存の安全機構 |
| 440# | F | Toxicity veto 棄却 → regime-side offset が代替策 |
| 447# | G, H, I | Inventory Sponging, Micro-Timeout, Spread Shadowing |
| 450# | B, F | FillRecord スキーマ穴 → 観測基盤整備が前提 |
| 451# | D | compound suppression 分析、toxicity_budget 有効化 |
| 453# | H | Micro-timeout 実装（disabled-by-default） |

---

## §8 結論

**最優先は案 B + F の同時実装**。macro_trend 閾値を下げて検知率を上げ（B）、検知結果を sell offset ブースト / スキップに直結させる（F）。これにより R2 + R3 を同時解決し、上昇トレンド時の sell 損失を構造的に防止する。

YAML 調整のみ（A-E 個別）では効果が限定的かつ相互依存が強く、「閾値ゲーム」に陥る。コード変更（F）を伴うことで、macro_trend → sell 保護の **設計上のミッシングリンク** を埋め、今後のチューニング空間も確保できる。

実装は 3 フェーズに分割し、各フェーズの効果を fill_records で定量検証してから次に進む。
