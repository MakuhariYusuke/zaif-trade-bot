# 214# 213# の Codex/Gemini 指摘に対する実コード・実データ検証レポート

> **日付**: 2026-03-02
> **対象**: `213_ph2_rev_205_212_validation_and_proposals.md` (Codex §1–7 + Gemini §8)
> **手法**: ソースコード精読、state file 実値照合、fill records 実集計、config/hot-reload 照合

---

## 0. 検証総括

| 213# 指摘 | 判定 | 実証方法 |
|---|---|---|
| §2.1 sleep clamp / halt log / persist 動作確認 | **正しい** | state file + ログで確認済 (211# 検証と一致) |
| §2.2 新 guard 未発火 | **正しい** | fill_records git_sha が旧コード (`03bdcfbf1a09`, `ac180d4f47f0`) のみ |
| §2.3 fill records が新コードの結果でない | **正しい** | 103 全件が旧 SHA。新 SHA は halt loop のみ |
| §3.1 DD state 移行穴 | **正しい — 実データで裏付け** | 下記 §1 |
| §3.2 Velocity SSOT 名称問題 | **部分的に正しい** | 下記 §2 |
| §3.3 Hot-reload 漏れ | **正しい** | 下記 §3 |
| §3.4 事実/仕様の分離 | **概ね正しい** | 下記 §4 |
| §8.2 Gemini velocity 批判 | **名称は問題、重大度は過大** | 下記 §2 |
| §8.5 Gemini DEFCON 提案 | **構想は正当、ただし §3.4 の指摘と矛盾** | 下記 §4 |
| §8.6 Gemini P0 4点 | **概ね妥当、優先度に修正あり** | 下記 §5 |

---

## 1. DD state 整合性欠陥 — 実データで完全裏付け

### 1.1 state file の現在値

```json
{
  "daily_pnl_bps": -110.94,
  "daily_fill_count": 29,
  "halted": true,
  "soft_triggered_today": false,
  "daily_pnl_bps_buy": 0.0,
  "daily_pnl_bps_sell": 0.0,
  "side_halted_buy": false,
  "side_halted_sell": false
}
```

### 1.2 fill records からの実計算

```
buy:  -36.64 bps (15 fills)
sell: -74.31 bps (14 fills)
total: -110.94 bps (29 fills)
```

### 1.3 不整合の特定

| フィールド | state file | あるべき値 | 不整合? |
|---|---|---|---|
| `soft_triggered_today` | `false` | `true` (PnL -110.94 < soft limit -30.0) | **YES** |
| `daily_pnl_bps_buy` | `0.0` | `-36.64` | **YES** |
| `daily_pnl_bps_sell` | `0.0` | `-74.31` | **YES** |
| `side_halted_buy` | `false` | `true` (buy -36.64 < per_side_hard -30.0) | **YES** |
| `side_halted_sell` | `false` | `true` (sell -74.31 < per_side_hard -30.0) | **YES** |
| `daily_pnl_bps` | `-110.94` | `-110.94` | OK |
| `daily_fill_count` | `29` | `29` | OK |
| `halted` | `true` | `true` | OK |

**5 フィールドが不整合。Codex の指摘は完全に正しい。**

### 1.4 根本原因の特定

1. **fill records が旧コード (`03bdcfbf1a09`, `ac180d4f47f0`) で生成されている**。これらの git SHA は 207# (per-side PnL 追加) より前のコード。そのため、`export_state()` に `daily_pnl_bps_buy/sell` フィールドが含まれていなかった。
2. **211# のボットリスタート時**、新コードの `import_state()` が実行される。存在しないフィールドは `float(data.get("daily_pnl_bps_buy", 0.0))` でデフォルト `0.0` になる。
3. **warmup は `daily_fill_count == 0` のときだけ実行される** ([fill_loop_orchestrator.py](scripts/v460/lib/fill_loop_orchestrator.py#L537-L541))。state file から `daily_fill_count=29` がロードされるため、warmup は一切走らず、不整合は修復されない。

```python
# fill_loop_orchestrator.py L537-541 — warmup 条件
if (
    self._daily_drawdown_guard.enabled
    and self._daily_drawdown_guard.state.daily_fill_count == 0  # ← この条件が厳しすぎる
    and existing_records
):
    self._warmup_daily_drawdown_from_records(existing_records)
```

### 1.5 `soft_triggered_today=false` の原因仮説

`soft_triggered_today` は 203# で `export_state()` に追加されているため、旧コードにも存在したはず。有力な仮説は:

- **`update_pnl()` の if/elif 構造バグ**: hard check が先、soft が elif で繋がっている。PnL が急落して1回の `update_pnl()` 呼出し中に soft limit と hard limit を同時に超えた場合、hard halt が発動し soft の elif はスキップされる。
- しかし 29 fills で -110.94 bps (平均 -3.8 bps/fill) なので、1 fill で -30 → -50 を跳び越す可能性は低い。
- **より可能性の高い仮説**: リスタート時にstate fileが「soft limit 未到達時の古いスナップショット」から復元された。211# で発覚した「8.3h 保存間隔」問題により、soft trigger 発動後の状態が永続化されていなかった可能性。

いずれにせよ、**import 後の整合性検証ロジックがない**ことが根本問題。

### 1.6 推奨修正

```python
# import_state 直後に整合性を検証し、不整合なら fill records から再構築
def _validate_dd_state_consistency(self, records: list["FillRecord"]) -> None:
    guard = self._daily_drawdown_guard
    if not guard.enabled or guard.state.daily_fill_count == 0:
        return

    # 条件1: PnL が soft limit 以下なのに soft_triggered が false
    pnl_below_soft = guard.state.daily_pnl_bps <= guard._soft_limit_bps
    if pnl_below_soft and not guard._soft_triggered_today:
        logger.warning("[DD-consistency] soft_triggered=false despite PnL below soft limit → rebuild")
        self._warmup_daily_drawdown_from_records(records)
        return

    # 条件2: per-side PnL の合計が total と乖離
    side_sum = guard.state.daily_pnl_bps_buy + guard.state.daily_pnl_bps_sell
    if abs(side_sum - guard.state.daily_pnl_bps) > 0.01 and abs(side_sum) < 0.01:
        logger.warning("[DD-consistency] per-side PnL sum=0 despite fills → rebuild")
        self._warmup_daily_drawdown_from_records(records)
```

---

## 2. Velocity SSOT 名称問題 — 機能は正常、名称は要改善

### 2.1 実態の確認

[velocity_math.py](scripts/v460/lib/velocity_math.py#L8-L20) は明確に 2 つの velocity を定義:

| 名前 | データソース | 時間窓 | 用途 |
|---|---|---|---|
| `instant_vel_bps` | orderbook mid Δ / Δt | ~2-5s (point-to-point) | VG offset boost |
| `trade_vel_60s` | 約定価格 first↔last | 60s | SG skip/offset |

### 2.2 現在の配線

```
maker_price.compute()
  → compute_instant_velocity_bps()  ← instant_vel_bps を計算
  → self._last_mid_trend_bps に保存

fill_loop_orchestrator L1121:
  price_velocity_60s=self._maker_price.last_mid_trend_bps  ← ★ instant_vel を 60s の名前で渡す
  # NOTE コメントでデータソース相違を明記

skip_gate_evaluator → velocity skip/offset 判定に使用
fill_cycle_executor → VG supplement にも使用
```

### 2.3 判定

| 観点 | 判定 |
|---|---|
| **コード動作の正確性** | **正常**。閾値 (`sell_velocity_skip_threshold_bps: 6.0`) の YAML コメントに `VG vel adverse med=-0.95` と記載 — 実際に instant_vel_bps のデータに対して校正されている |
| **命名の正確性** | **不正確**。`price_velocity_60s` という名前は 60s trade-based velocity を意味するが、実データは instant OB velocity |
| **Codex 指摘の妥当性** | **正しい** — 名称・ログ・閾値解釈で誤読を招く |
| **Gemini 「時間次元の冒涜」** | **修辞的に過激だが核心は正当** — $\frac{dP}{dt}$ (瞬時微分) と $\frac{\Delta P}{\Delta t_{60s}}$ (60s 区間平均) は物理的に異なる量。ただし閾値が実データで校正されているため「致命的な誤発注の温床」という断定は**過大**。 |

### 2.4 影響範囲の限定

現時点で `trade_vel_60s` (本来の 60s trade velocity) を使っている箇所はない。`velocity_math.py` の docstring に存在を文書化しているが、データパイプラインには接続されていない。したがって:

- **「二重人格」ではなく「一つの値に間違った名前を付けている」** が正確な描写
- 将来 `trade_vel_60s` を実際に接続する際に混乱するリスクは高い

### 2.5 推奨修正

1. **引数名の変更**: `price_velocity_60s` → `price_velocity_bps` (時間窓を名前から外す)
2. **CycleGate dataclass フィールド名変更**: 同上
3. **影響範囲**: skip_gate_evaluator.py (~15箇所), fill_cycle_executor.py (~6箇所), fill_loop_orchestrator.py (~1箇所), 関連テスト
4. **リスク**: 機能変更なし、純粋なリネーム。テストの閾値変更不要

---

## 3. Hot-reload 漏れ — 完全裏付け

[config_hot_reload.py](scripts/v460/lib/config_hot_reload.py#L58-L223) の `_HOT_RELOADABLE_FIELDS` を全件照合。
以下のフィールドは **確実に hot-reload 対象外**:

### 3.1 HIGH (新防御パラメータ — 事故直後に最初に触りたい)

| フィールド | 定義場所 | 用途 |
|---|---|---|
| `loss_cooldown_threshold_bps` | fill_config.py L181 | loss cooldown 発動閾値 |
| `loss_cooldown_interval_mult` | fill_config.py L182 | loss cooldown 時の interval 乗数 |
| `loss_boost_offset_mult` | fill_config.py L183 | 211# loss boost offset 乗数 |
| `toxic_fill_veto_threshold_bps` | fill_config.py L185 | toxic fill 拒否閾値 |
| `toxic_fill_veto_cycles` | fill_config.py L186 | toxic fill 拒否持続サイクル数 |
| `one_sided_consecutive_limit` | fill_config.py L188 | 片側連続実行制限 |
| `one_sided_consecutive_interval_mult` | fill_config.py L189 | 片側連続時の interval 乗数 |

### 3.2 MEDIUM (運用時に調整したい)

| フィールド | 定義場所 | 用途 |
|---|---|---|
| `per_side_dd_enabled` | fill_config.py L174 | 片側 DD on/off |
| `per_side_dd_hard_limit_bps` | fill_config.py L175 | 片側 DD 閾値 |
| `per_side_dd_halt_cycles` | fill_config.py L176 | 片側封鎖サイクル数 |
| `hard_skip_utc_hours_buy` | fill_config.py L192 | 時間帯完全スキップ (buy) |
| `hard_skip_utc_hours_sell` | fill_config.py L193 | 時間帯完全スキップ (sell) |
| `max_cycle_sleep_sec` | fill_config.py L-- | sleep 上限 |

### 3.3 Codex/Gemini の指摘に対する判定

**完全に正しい。** 新しく積んだ防御ほど live で調整しにくいという逆転が起きている。
`_HOT_RELOADABLE_FIELDS` への追加は単純な作業 (frozenset に文字列を追加するだけ) であり、リスクは極めて低い。

**補足**: `daily_drawdown_hard_limit_bps` と `daily_drawdown_soft_limit_bps` は hot-reload 対象に**含まれている**。しかし `_COMPONENT_REBUILD_PREFIXES` に `"daily_drawdown_": "_rebuild_daily_drawdown_guard"` があるため、DD guard 本体の再構築が走る。per_side 系は DD guard の初期化パラメータなので、同じ rebuild パスで処理される必要がある。

→ per_side 系は `_HOT_RELOADABLE_FIELDS` に追加しつつ `_rebuild_daily_drawdown_guard` で再構築されるようにすれば安全。

---

## 4. 事実/仕様の分離 & 外部イベント叙述

### 4.1 Codex の指摘

> `alert_mode.json` という発想自体は合理的だが、外部イベントの叙述が仕様文書に混ざりすぎている

### 4.2 Gemini の反論 (§8.5)

> `alert_mode.json` こそが Jump Risk 環境下における「唯一の生存手段」であり、仕様に混ぜることに問題はない

### 4.3 我々の判定

**両者の核心は矛盾していない:**

- **Codex の論点**: 事実叙述 (Operation Epic Fury、BTC 価格推移) は仕様ではなく背景セクションに分離すべき → **妥当**
- **Gemini の論点**: alert_mode の仕組み自体は最優先で実装すべき → **妥当**

つまり:
1. 211# §8 の「仕組み定義」部分 (alert_mode.json のスキーマ、発動手順) は仕様として残す
2. 「イラン攻撃」の固有叙述は根拠として残しつつ「外部事実セクション」として分離する
3. `Operation Epic Fury` のオペレーション名は**ウェブ検索で確認できない** (Codex の §4 指摘に同意)。複数のニュースソースで「イスラエルがイランを攻撃」自体は確認できるが、作戦名は未確認のため**削除が安全**

### 4.4 BTC 価格経路の検証

211# §8 に記載された `$67K → $63K 急落 → $68K 反発 → $66K` という価格経路は、同期間の CoinDesk 記事見出し ("Bitcoin nears $63,000", "Bitcoin tops $68,000") と概ね整合する。ただし正確な価格は取引所・タイムスタンプにより異なるため、参考値として扱うのが適切。

---

## 5. P0 提案の妥当性評価

### Codex §5 vs Gemini §8.6 の比較

| # | Codex §5 提案 | Gemini §8.6 提案 | 判定 |
|---|---|---|---|
| **1** | DD state 整合修復 | State マイグレーション穴塞ぎ | **同一指摘。P0 で合意。** |
| **2** | 206#–211# 専用検証 run | (言及なし) | **妥当だが DD halt 解除後。P1** |
| **3** | Hot-reload 対象拡張 | 防具群の Hot-reload 完全統合 | **同一指摘。P0 で合意。** |
| **4** | Velocity 名前分離 | Velocity 完全分離 | **同一指摘。Codex=P1、Gemini=P0。** |
| **5** | Guard 発火カウンタ state 追加 | (言及なし) | **有用。P1。** |
| **6** | alert_mode.json 最小実装 | DEFCON 即時実装 | **同一指摘。P0 で合意。** |

### 5.1 我々の P0 優先度決定

Gemini は velocity を P0 (即時) としているが、§2 で検証した通り**機能的には正常動作**しており、命名変更は純粋なリファクタリング。一方、DD state 整合性は**現在進行形のデータ不整合**であり、halt 解除時に即座に影響する。

**確定した P0 (即時):**

1. **DD state 整合性修復** (§1.6) — 再起動時に fill records から per-side PnL を再構築
2. **Hot-reload フィールド追加** (§3.1 HIGH 7 フィールド) — 即変更可能にする
3. **alert_mode.json 最小実装** (211# §8) — 手動緊急停止機構

**確定した P1 (短期):**

4. **Velocity リネーム** — `price_velocity_60s` → `price_velocity_bps`
5. **Guard 発火カウンタ永続化** — state に `hard_skip_count`, `toxic_veto_count` 等を追加
6. **206#–211# 検証 run** — halt 解除後に新 guard の発火確認

---

## 6. Gemini §8 の学術的主張に対する技術的評価

### 6.1 Merton のジャンプ拡散過程 (§8.3)

> Jump が発生した瞬間、流動性は完全に蒸発し、過去のデータから計算された σ や Offset は無意味

**学術的には正しい。** Black-Scholes の連続拡散仮定は地政学ショックで崩壊する。しかし本システムは Black-Scholes に依存しておらず、maker limit order + offset という構造で Jump Risk に対しては以下の防御が既存:

- VG (Volatility Guard): ATR 急騰 → offset 拡大
- DD Guard: 累積損失 → halt
- Loss cooldown: 直前損失 → interval 延長

**Gap**: これらは全て「Jump の結果を観測してから反応」。Merton の λ (Jump 強度) を推定するコンポーネントは存在しない。alert_mode.json が人間による λ 推定の代替となる。

### 6.2 Hawkes 過程 (§8.4)

> 一度のショックがトリガーとなり、連鎖的なロスカットとパニック売りを巻き起こす

**これは実際に 2026-02-28 の BTC 価格推移で観測されている。** $63K への急落後の乱高下パターンは自己励起的。現在の bot でこれを検知する機構は**ない**。P1-B Micro Circuit Breaker の理論的根拠として有用。

### 6.3 Gemini の修辞について

§8.1 の「エンジニアリングとしての怠慢」「Claudeは無神経なコードを残している」は、§3 の hot-reload 漏れに対する批判としては**内容の核心は正当**だが、修辞的に**チーム内コミュニケーションとして不適切**。技術的事実に基づく批判は歓迎するが、人格攻撃は改善に寄与しない。

---

## 7. 是正措置まとめ

| # | 対応内容 | 根拠 | 優先度 | 難易度 |
|---|---|---|---|---|
| A | DD state 整合性検証 + 自動修復 | §1 | **P0** | 低 (~30行) |
| B | Hot-reload 7 フィールド追加 | §3 | **P0** | 極低 (~7行) |
| C | alert_mode.json 最小実装 | 211# §8 + §4 | **P0** | 低 (~50行) |
| D | Velocity 引数リネーム | §2 | P1 | 中 (~20箇所) |
| E | Guard 発火カウンタ永続化 | §5 | P1 | 低 (~20行) |
| F | 211# §8 事実/仕様分離 | §4 | P1 | 極低 (文書編集) |
| G | 206#–211# 検証 run | Codex §5.2 | P1 | 中 (halt 解除待ち) |
| H | `update_pnl` if/elif 構造見直し | §1.5 | P2 | 低 (~5行) |

---

## 8. 212# 命名規則修正

`212_codebase_audit_review.md` → `212_ph2_audit_codebase_quality.md` に是正。

v460 ドキュメントの命名規則: `{number}_ph{phase}_{type}_{description}.md`

- `ph2` = 現在のフェーズ
- `audit` = 種別 (監査)
- `codebase_quality` = 説明

index.md のリンクも同期更新済み。
