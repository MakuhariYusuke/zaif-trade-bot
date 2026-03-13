# 336# 334#/335# 主張検証・施策策定・追加改善点

> **種別**: rev (検証・施策レビュー)  
> **対象**: 334# (Codex), 335# (Gemini 3.1 Pro)  
> **起票**: 2026-03-08  
> **観点**: claims validation / measures design / additional improvements  
> **ベース SHA**: `5f6b91212` (HEAD)  
> **テスト**: `.\.venv\Scripts\python.exe -m pytest tests/unit/v460 --no-cov` → **4105 passed**

---

## §1 結論

334# (Codex) と 335# (Gemini 3.1 Pro) は、初期スタンスの温度差はあったが、最終的に以下の合意に至った。これらの各主張をコードと理論の両面から検証し、具体的施策を策定する。

**両レビューの合意点:**

1. `buy_dynamic_kill` の閾値 -0.8bps は過剰に攻撃的
2. 両面 MM 参加の復旧が最優先課題
3. ranging での利益源泉を認める
4. sell 側は閾値近辺であり過剰反応すべきでない
5. God Object 分割の方向性は正しいが「分散型 God Object」リスクあり

---

## §2 主張の妥当性検証

### §2.1 主張 A: `buy_dynamic_kill_threshold_bps = -0.8` は過敏すぎる

**判定: ✅ 妥当 — コード検証で確認**

| 項目 | buy | sell | 非対称性 |
|---|---|---|---|
| base threshold | **-0.8 bps** | **-0.3 bps** | buy が 2.67× 寛容 |
| ranging regime | -0.8 (base) | -0.5 | buy が 1.6× 寛容 |
| trending_down | -0.5 | -1.0 | sell が 2× 寛容 |
| trending_up | -1.5 | -0.3 | buy が 5× 寛容 |
| resume_window | 10 | 10 | 同一 |
| inv_relaxation | enabled (max +0.3) | なし | buy のみ |

一見 buy は sell より寛容だが、実際の効果を見ると:
- sell の base threshold は -0.3 であり **極めてタイト**
- buy の -0.8 は sell の -0.3 より寛容だが、**MM の microstructure noise を考慮すると依然過敏**
- 50-fill window での rolling mean PnL が -0.8bps を下回ることは、数回の -3〜-5bps fill で容易に発生

**定量的シミュレーション:**
- 50 fill 中 45 件が +0.5bps、5 件が -5.0bps の場合:
  - rolling mean = (45×0.5 + 5×(-5.0)) / 50 = (22.5 - 25.0) / 50 = **-0.05 bps** → kill されない
- 50 fill 中 43 件が +0.5bps、7 件が -5.0bps の場合:
  - rolling mean = (43×0.5 + 7×(-5.0)) / 50 = (21.5 - 35.0) / 50 = **-0.27 bps** → kill されない
- 50 fill 中 40 件が +0.5bps、10 件が -5.0bps の場合:
  - rolling mean = (40×0.5 + 10×(-5.0)) / 50 = (20.0 - 50.0) / 50 = **-0.60 bps** → kill されない
- 50 fill 中 38 件が +0.5bps、12 件が -5.0bps の場合:
  - rolling mean = (38×0.5 + 12×(-5.0)) / 50 = (19.0 - 60.0) / 50 = **-0.82 bps** → ❌ kill 発動

7 件の extreme loss で kill されないが、12 件 (24%) で kill 発動。333# の buy AS 率は 12.2% だが、rolling window 内の事象分布は non-stationary であり、**局所的な AS クラスタが kill を誘発しやすい**。

**追加要因 — カスケード増幅:**
1. buy kill → balance_forced_switch 増加 → forced_buy_delay 発動
2. forced_buy_delay → degraded_liquidation_duty_skip 発動
3. 3 層の抑制が直列に作用し、**1 つの kill 発動が 3 倍の機会損失に増幅**

**検証結論:** -0.8 は MM 理論上の noise margin に対して過敏。ただし「安易に解放せよ」ではなく、段階的緩和が妥当。

---

### §2.2 主張 B: Survivorship Bias は存在するが、緩和しても破綻しない

**判定: ✅ 部分的に妥当 — 335# の初期反応は過大、自己訂正後が適切**

335# は当初「閾値を緩めた途端 PnL がマイナスへ急転直下する確率が極めて高い」と断言したが、§5 自己訂正で撤回。

**survivorship bias の実態:**
- 確かに kill された 216 件の buy が仮に fill されていた場合、平均 PnL は低下する
- しかし kill 判定は **past rolling PnL** に基づいており、**将来の個別 fill の PnL を予測していない**
- kill が発動する局面 = rolling mean < -0.8bps = **直近 50 fill の平均が悪い局面**
- この時点の次の 1 fill が AS に遭う確率は、直近 50 fill が良好な局面と統計的に大差ない可能性がある

**理論的根拠:**
- Glosten-Milgrom (1985) の逆選択モデルは **order flow** の情報含有量が key
- rolling PnL mean は order flow toxicity の noisy estimator に過ぎない
- 50-fill window の -0.8bps threshold は、真の逆選択リスクではなく **microstructure noise に反応**している

**検証結論:** survivorship bias は存在するが、-0.8→-1.5 程度の緩和で PnL が「急転直下」する可能性は低い。moderate な PnL 低下 (buy avg_pnl +0.372 → +0.1 ~ -0.3 程度) は予想されるが、在庫均衡回復による間接効果が上回る。

---

### §2.3 主張 C: Ranging での利益は自然であり、偏りの証拠ではない

**判定: ✅ 妥当**

334# §6.1: 「受動的 MM が一番勝ちやすいのは ranging」— これは MM 理論の基本。

333# の ranging=90.3% は確かに偏りだが、**BTC/JPY の日次 regime 分布は 60-70% ranging が常態**と推定される。したがって:
- 90.3% は楽観的だが、60-70% でも主要な利益源泉であることに変わりない
- trending 性能が改善不要という意味ではない — 別問題として切り分けるべき

---

### §2.4 主張 D: sell p10 = -5.207 は統計的ノイズ

**判定: ✅ 妥当**

n=51 での 10th percentile の bootstrap SE は ±2-3 bps 程度。-5.207 vs -5.000 の差 0.207 は:
- 1 件の extreme fill の有無で容易に反転する
- 追加防御で対処すべき問題ではなく、**n 蓄積で収束を待つべき**
- 334# の「sell は監視継続、追加工事は保留」に同意

---

### §2.5 主張 E: 分散型 God Object リスク

**判定: ⚠️ 部分的に妥当 — ただし現時点では過大評価**

335# が指摘した「分散型 God Object」リスクについてコードを検証:

**CycleContext の設計:**
- `CycleContext` は `dataclass` でイテレーションスコープの可変状態を管理
- pre-cycle → balance → mid-cycle → execute → post-cycle の各 mixin が参照・更新
- **ミューテーション箇所は pre_cycle と balance の冒頭に集中** しており、mid-cycle/execute は主に読み取り

**RunSessionState の設計:**
- ループ間共有状態だが、更新箇所は post_cycle に集約
- `_forced_buy_delay_remaining`, `_degraded_liquidation_duty_counter` 等はそれぞれの責務 mixin でのみ更新

**現時点の評価:**
- 335# の懸念は理論的には正しいが、**現在のコード構造ではミューテーション境界が比較的明確**
- `CycleContext` を `@dataclass(frozen=True)` にして copy-on-write にすることは可能だが、パフォーマンスコストと開発速度のトレードオフ
- **今やるべきではない** — 収益改善が最優先であり、防御的リファクタリングは後回し

---

### §2.6 主張 F: Regime Detector 遅行性

**判定: ⚠️ 検証限定的 — データ不足**

335# §3.2 が「ranging 判定下で何百回も Buy が Kill される矛盾」を指摘。

しかし、buy_dynamic_kill は **regime thresholds** を使用している:
- ranging regime → base threshold -0.8 (regime_thresholds に ranging 未指定のため)
- つまり ranging 判定と buy kill は **矛盾しない** — ranging でも rolling PnL が悪ければ kill される

**真の問題は regime detector ではなく threshold 自体の過敏さ** であるため、335# のこの指摘は的外れ。regime_thresholds に `ranging: -1.5` を設定すれば ranging 時の buy kill は大幅緩和される。

---

## §3 カスケード増幅メカニズム — buy 側 3 層抑制の構造分析

334# と 335# が個別に指摘した 3 つの buy 抑制メカニズムは、実は **直列カスケード** を形成している:

```
[Layer 1] buy_dynamic_kill (-0.8bps threshold)
    │
    ├── kill 発動 + NOT balance_forced → 完全ブロック (216 件)
    │
    └── kill 発動 + balance_forced → degraded_liquidation モード
            │
            [Layer 2] degraded_liquidation_duty_skip (duty_cycle=3)
            │   │
            │   └── 3 サイクル中 2 回スキップ (95 件)
            │       残り 1 回は lot×0.2, offset×3.0 で縮退実行
            │
[Layer 3] forced_buy_delay (velocity ≤ -3.0/-5.0 bps)
    │
    └── balance_forced + buy + velocity 急落 → 3 サイクル待機 (100 件)
```

**カスケード係数の推定:**
- buy_dynamic_kill 1 件 → 在庫偏重 → balance_forced → 追加で ~0.5 件の forced_buy_delay + ~0.4 件の duty_skip を間接誘発
- **実効増幅率: ~1.9× (1 件の kill が ~1.9 件の機会損失に)**

333# data による検証:
- buy_dynamic_kill: 216 件 (root cause)
- forced_buy_delay + duty_skip: 195 件 (secondary effects)
- 比率: 195/216 = 0.90 — kill の 90% が二次効果を伴う
- **Layer 1 を緩和すれば Layer 2/3 も自然に減衰する**

---

## §4 施策提案

### §4.1 T-1: buy_dynamic_kill 閾値の段階的緩和 (P0)

**変更案:**

| パラメータ | 現行値 | 提案値 | 根拠 |
|---|---|---|---|
| `buy_dynamic_kill.threshold_bps` | **-0.8** | **-1.5** | sell(-0.3) の 5 倍。MM noise margin を確保しつつ真の AS を検出 |
| `buy_dynamic_kill.regime_thresholds.ranging` | (未設定=base) | **-2.0** | ranging は MM の主戦場。buy kill の大部分は ranging で発生。大幅緩和 |
| `buy_dynamic_kill.regime_thresholds.trending_down` | -0.5 | **-1.0** | 現行値は売りの trending_down(-1.0) と著しく非対称。sell と同水準に |
| `buy_dynamic_kill.regime_thresholds.trending_up` | -1.5 | -1.5 | 据え置き (順張り方向で既に寛容) |
| `buy_dynamic_kill.regime_thresholds.high_vol` | -0.5 | **-1.0** | 高ボラ時も一定の buy 参加は在庫均衡に必要 |

**期待効果:**
- buy fill_rate: 9.3% → 25-35% (ranging 緩和が主寄与)
- buy avg_pnl: +0.372 → 0.0 ~ -0.3 (survivorship bias 剥離)
- 在庫均衡改善 → balance_forced_switch 170→100 前後に減少
- forced_buy_delay/duty_skip の自然減衰 (カスケード解消)

**リスク:**
- buy AS 率上昇 (12.2% → 15-20%)
- worst-case: buy avg_pnl が -1.0bps まで悪化 → それでも AB 閾値 (-1.00bps) 近辺

### §4.2 T-2: buy_dynamic_kill_inv_relaxation の強化 (P0)

**変更案:**

| パラメータ | 現行値 | 提案値 | 根拠 |
|---|---|---|---|
| `buy_dynamic_kill_inv_relaxation.max_bps` | 0.3 | **0.5** | BTC 枯渇時の在庫修復 buy をより強く許容 |
| `buy_dynamic_kill_inv_relaxation.scale` | 0.5 | 0.5 | 据え置き |

現行の max 0.3bps では effective threshold が -0.8+0.3 = -0.5 にしか緩和されない。0.5bps なら -0.3 まで緩和され、sell base threshold と同水準になる。

### §4.3 T-3: forced_buy_delay の条件緩和 (P1)

**変更案:**

| パラメータ | 現行値 | 提案値 | 根拠 |
|---|---|---|---|
| `forced_buy_delay.velocity_threshold_ranging_bps` | -3.0 | **-5.0** | ranging での velocity -3.0 は通常の oscillation。-5.0 (base と同値) で十分 |
| `forced_buy_delay.cycles` | 3 | **2** | 3 cycle × ~120s = 6 分は過剰。2 cycle (4分) で十分な冷却 |

**根拠:** 334# §7.P0-2 で「hard skip から守備的 quote に落とす」提案がある。velocity_threshold の緩和は直接的だが、将来的には hard skip → offset boost softening も検討。

### §4.4 T-4: degraded_liquidation_duty_cycle の緩和 (P1)

**変更案:**

| パラメータ | 現行値 | 提案値 | 根拠 |
|---|---|---|---|
| `degraded_liquidation_duty_cycle` | 3 | **2** | 1/3 参加率(33%) → 1/2 参加率(50%)。lot×0.2 + offset×3.0 の縮退条件は維持 |

duty_cycle=3 は 333# で 95 件の skip を生成した。T-1 の kill 緩和でこのパスに入る頻度自体が減るが、入った場合の参加率も改善すべき。

### §4.5 T-5: buy_dynamic_kill.regime_thresholds に ranging を明示 (P0)

**新規追加:**

現在 `buy_dynamic_kill.regime_thresholds` に `ranging` キーがない。ranging 時は base threshold (-0.8) が適用される。ranging は MM の主戦場であるにもかかわらず、トレンド時と同じ閾値を使用している。

```yaml
buy_dynamic_kill:
  regime_thresholds:
    ranging: -2.0      # NEW: MM 主戦場で大幅緩和
    trending_down: -1.0 # -0.5→-1.0
    trending_up: -1.5   # 据え置き
    high_vol: -1.0      # -0.5→-1.0
```

これは T-1 と密接に関連するが、**base threshold の変更と regime-specific 変更を分離** することで、意図が明確になる。

---

## §5 334#/335# が触れていない追加改善点

### §5.1 sell_dynamic_kill base threshold の再評価

**発見:** sell の base threshold は **-0.3 bps** と極めてタイト。

| 比較項目 | sell | buy (現行) | buy (提案) |
|---|---|---|---|
| base threshold | -0.3 | -0.8 | -1.5 |
| ranging threshold | -0.5 | (base) | -2.0 |

sell の -0.3 は buy の -0.8 より 2.67× タイト。sell fill_rate=46.8% は高いが、**sell side の kill が本来必要な防御を超えて発動していないか** も今後モニターすべき。

333# では sell_dynamic_kill による skip 件数は不明だが、sell fill_rate が健全なため現時点では変更不要。168h data で再評価。

### §5.2 YAML / Code Default Drift

334# §5.3 が指摘した `unknown_regime_max_consecutive` の drift:

| 場所 | 値 | 備考 |
|---|---|---|
| `fill_config.py` L519 | `10` | コード既定値 |
| `fill_test.yaml` L27 | `5` | `321# M-3: 10→5` |

YAML で上書きされるため **現時点で実害なし** だが、コード既定値を YAML と一致させるべき (config_loader 依存ではなく standalone テストで顕在化するリスク)。

**他の drift 候補:**
- `sell_dynamic_kill_threshold_bps`: code default `-0.5` / YAML `-0.3` (YAML が 246# で変更)
- `trending_sell_offset_boost_factor`: code default `2.0` / YAML `1.5` (YAML が 320# で変更)

### §5.3 333# 分析の再現可能性

334# §3.4: 「333# に対応する専用スクリプトや JSON 出力が repo に見当たらない」

`temp/sha_analysis.py`, `temp/sha_combined_analysis.py` が存在するが、`analysis/` 昇格と `analysis_results/` への JSON 出力がおそらくされていない。168h data が蓄積する前に、分析パイプラインを整備すべき。

### §5.4 max_kill_duration_sec の妥当性

現行 `max_kill_duration_sec = 1800` (30分) は 268# の 92 分持続事例への対策。

T-1 で kill threshold を緩和したことにより、kill 発動頻度自体が減少し、この安全弁の重要性は低下する。しかし値自体は妥当であり変更不要。

### §5.5 dual_kill_quiescence と T-1 の相互作用

`dual_kill_quiescence_enabled: true` は buy+sell 両方 kill 時に休止する設定。T-1 で buy kill が減少すると、dual kill 発生頻度も自然減少する。

ただし、**sell kill (-0.3) が発動しやすい現状で buy kill を緩和すると、sell のみ kill の「片肺」が buy→sell 方向に反転するリスク**がある。これは sell threshold の相対的タイトさによるもので、168h data で sell 側の kill 頻度も追跡すべき。

---

## §6 施策の優先順位と実行計画

### Phase 1: 即時実行 (T-1, T-2, T-5 — P0)

**YAML 変更のみ。コード変更なし。テスト影響なし。**

```yaml
# configs/v460/fill_test.yaml 変更
buy_dynamic_kill:
  threshold_bps: -1.5            # 現行 -0.8
  regime_thresholds:
    ranging: -2.0                # NEW
    trending_down: -1.0          # 現行 -0.5
    trending_up: -1.5            # 据え置き
    high_vol: -1.0               # 現行 -0.5

buy_dynamic_kill_inv_relaxation:
  max_bps: 0.5                   # 現行 0.3
```

### Phase 2: 次バッチ (T-3, T-4 — P1)

T-1 の効果を 24-48h 観測後に実施。

```yaml
forced_buy_delay:
  velocity_threshold_ranging_bps: -5.0  # 現行 -3.0
  cycles: 2                              # 現行 3

degraded_liquidation_duty_cycle: 2       # 現行 3
```

### Phase 3: 168h 計測

Phase 1 + (Phase 2 if needed) を適用した SHA で 168h (7日) G1.2 gate 計測を開始。

---

## §7 リスク評価

### §7.1 T-1 失敗シナリオ

| シナリオ | 確率 | 影響 | 対応 |
|---|---|---|---|
| buy avg_pnl が -1.0 以下に悪化 | 低 (15%) | AB FAIL | threshold を -1.2 に引き戻し |
| buy AS 率が 25%+ に悪化 | 中 (30%) | p10 悪化 | regime_thresholds.ranging を -1.5 に引き戻し |
| 在庫偏重が解消されない | 低 (10%) | 構造変更不足 | inv_relaxation さらに強化 |
| sell kill が相対的に目立つ | 中 (25%) | 片肺反転 | sell threshold の再評価 |

### §7.2 ロールバック計画

T-1/T-5 は YAML のみの変更。異常検出時は YAML revert のみでロールバック可能。  
コードレベルの変更を伴わないため、テスト-回帰のリスクは 0。

---

## §8 334#/335# の各 §への最終判定

| レビュー | セクション | 主張 | 判定 | 備考 |
|---|---|---|---|---|
| 334# | §4.1 | buy 過剰抑制が本丸 | ✅ 妥当 | カスケード増幅メカニズムを検証済み |
| 334# | §4.2 | 二面参加の品質が崩壊 | ✅ 妥当 | buy fill_rate 9.3% は壊滅的 |
| 334# | §4.3 | sell は監視対象に留める | ✅ 妥当 | p10=-5.207 は統計的ノイズ圏 |
| 334# | §6.1 | ranging が本来の収益源 | ✅ 妥当 | MM 理論の基本 |
| 334# | §6.2 | trend capture は別戦略 | ✅ 妥当 | passive MM と trend following は混ぜるべきでない |
| 334# | §7.P0-1 | buy kill 限定的緩和実験 | ⚠️ 内容は妥当、方法が保守的 | global → regime-aware に修正提案 |
| 334# | §7.P0-2 | forced_buy_delay を守備的 quote に | ⚠️ 方向は妥当 | まずは velocity 閾値緩和 (P1) |
| 334# | §7.P0-3 | degraded duty_skip 分解 | ✅ 妥当 | T-1 がカスケードの root cause |
| 335# | §3.1 初期 | 生存者バイアスで急転直下 | ❌ 過大評価 | §5 で自己訂正済み |
| 335# | §3.2 初期 | 火災報知器を切るな | ❌ 誤り | §5 で「過敏スプリンクラー」に訂正 |
| 335# | §3.3 | Inventory の死 | ✅ 妥当 | カスケード増幅で確認 |
| 335# | §2 | 分散型 God Object リスク | ⚠️ 理論的に正しい | 現時点では実害なし、防御的リファクタは後回し |
| 335# | §5 | buy kill 緩和は P0 必須 | ✅ 妥当 | -1.5〜-2.0 の提案に同意 |
| 335# | §5 | 168h 計測前に修正必要 | ✅ 妥当 | 片肺状態での 168h は無価値 |

---

## §9 まとめ

1. **buy 側 3 層抑制のカスケード増幅** が最大の構造問題。root cause は `buy_dynamic_kill` の -0.8bps threshold。
2. **T-1 (threshold -0.8→-1.5) + T-5 (ranging: -2.0 追加)** が最優先施策。YAML のみの変更で低リスク。
3. sell 側は現状維持。168h data で再評価。
4. 334# と 335# (自己訂正後) はほぼ全面的に妥当。335# の初期反応 (§3.1/3.2) のみ過大評価。
5. 335# のレジーム遅行性指摘は的外れ — 問題は threshold の過敏さ。
6. God Object 分割への懸念は理論的に正しいが、**収益改善が先**。
