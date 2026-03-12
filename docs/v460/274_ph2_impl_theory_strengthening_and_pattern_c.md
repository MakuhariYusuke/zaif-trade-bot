# 274# 市場理論補強 + MacroRegime/Kelly 有効化 + Pattern C 検証

> **フェーズ**: ph2 (G1.1-exec)  
> **種別**: impl (実装 + 理論補強)  
> **日付**: 2026-03-04  
> **前提**: 269# レビュー → 270# Gemini レビュー → 272# DRY + 残課題拾い上げ → 273# I3/I5/I6 解決 → **本 274# で理論補強 + 残課題解消**

---

## 1. 背景

273# で 268# インシデントの全 6 件 (I1–I6) + Pattern B が解決済み。本 274# では:

1. **市場理論 docstring 補強** — 中核 3 モジュールに学術引用を追加
2. **MacroRegime 観測モード有効化** — 269#/270#/272# 全レビューが推奨
3. **Kelly Criterion YAML 配線** — 264# で実装済みだが YAML セクション未追加
4. **Pattern C 検証テスト** — dual-kill + aggregate halt + balance_forced 3 層同時
5. **deprecated CLI 引数削除** — `--api-key`/`--api-secret` セキュリティリスク排除

---

## 2. 市場理論 docstring 補強

### 2.1 対象モジュールと追加理論

| モジュール | 追加理論 | 学術引用 |
|---|---|---|
| `daily_drawdown_guard.py` | Optimal Stopping Theory | Chow, Robbins & Siegmund (1971) |
| | Holding Risk (在庫方向リスク) | Stoll (1978) §3, Ho & Stoll (1981) |
| | Per-side halt の情報非対称性解釈 | Glosten & Milgrom (1985) |
| `fill_loop_orchestrator.py` | Inventory Risk Management | Stoll (1978), Ho & Stoll (1981) |
| | Liveness vs Safety トレードオフ | 273# I3/I5/I6 で実証 |
| | Optimal market maker モデル | Avellaneda & Stoikov (2008) |
| `cycle_gate_aggregator.py` | Hard Gates 理論根拠 | Glosten-Milgrom (1985) kill, Roll (1984) effective spread |
| | Soft Gates 理論根拠 | Kyle (1985) λ velocity, regime conditional expectation |

### 2.2 Gate Soft/Hard 分類体系

274# で cycle_gate_aggregator の 9 ゲートを理論的に分類:

| Gate # | 名称 | 分類 | 理論根拠 |
|---|---|---|---|
| 1 | unknown_regime_buy_skip | **Soft** | Regime 不確実性 → recovery でバイパス可 |
| 2 | ranging_low_vol_buy_skip | **Soft** | 条件付き期待値変化 |
| 3 | trending_sell_skip | **Soft** | Regime conditional expectation |
| 4 | buy_dynamic_kill | **Hard** | Glosten-Milgrom 逆選択 → recovery でもブロック |
| 5 | sell_dynamic_kill | **Hard** | Glosten-Milgrom 逆選択 → recovery でもブロック |
| 6 | velocity_sell_skip | **Soft** | Kyle λ velocity → recovery でバイパス可 |
| 7 | unknown_regime_sell_skip | **Soft** | Regime 不確実性 → recovery でバイパス可 |
| 8 | narrow_spread | **Hard** | Roll (1984) effective spread < 取引コスト |
| 9 | confidence_gate | **Hard** | 信頼度が基準未満 |

**設計原則**: Hard gates は市場構造的リスク（逆選択・コスト超過）を反映するため、recovery 状態でもバイパス不可。
Soft gates は情報不確実性 or 条件付き判断であり、halt recovery 中は Liveness 確保のためバイパスを許容。

---

## 3. MacroRegime 観測モード有効化

### 3.1 経緯

| Issue | 推奨内容 |
|---|---|
| 269# | MacroRegime 有効化を推奨 (conflict_action: log) |
| 270# | Gemini 31 Pro も同意 |
| 272# | 残課題として再掲 |

189# で実装されたが `enabled: false` のまま放置。274# で `enabled: true` + `conflict_action: log` に変更。
**観測モードのため、ゲート判定には影響しない**。ログ出力のみで判断材料を蓄積。

### 3.2 変更

```yaml
# configs/v460/fill_test.yaml
macro:
  enabled: true    # 274# 観測モード有効化
  conflict_action: log
```

---

## 4. Kelly Criterion YAML 配線

### 4.1 問題

264# で `lot_sizer.py` に Kelly Criterion 実装済み。`adaptation_engine.py` L423 で
`self._yaml_cfg.get("kelly", {})` として YAML から読み取るが、**YAML にセクションが存在しなかった**。

### 4.2 理論的根拠

> J.L. Kelly Jr. (1956) "A New Interpretation of Information Rate"  
> — 情報の質に比例した最適ベット比率。full Kelly は変動が大きいため **half-Kelly (fraction: 0.5)** を採用。

### 4.3 追加 YAML セクション

```yaml
# configs/v460/fill_test.yaml
kelly:
  enabled: true
  equity_btc: 0.002      # 総資産 (BTC 換算)
  fraction: 0.5           # half-Kelly — 分散を 1/2 に抑制
  max_fraction: 0.25      # 安全上限
  min_win_samples: 30     # 推定に必要な最小約定数
```

---

## 5. Pattern C 検証テスト

### 5.1 Pattern C とは

268# で未検証だった **3 層同時デッドロック**:

```
Layer 1: dual-kill (buy + sell 両方 kill 状態)
Layer 2: aggregate halt または per-side halt
Layer 3: balance_forced (在庫圧迫で強制決済要求)
```

273# で I3/I5/I6 を解決したことにより、各層の解除機構が整備された。
274# ではこれらが同時発生するケースを網羅テスト。

### 5.2 テストケース

| テスト名 | シナリオ | 期待結果 |
|---|---|---|
| `test_dual_kill_plus_balance_forced_degraded` | dual-kill + balance_forced | degraded liquidation で通過 |
| `test_single_kill_balance_forced_halt_recovery` | kill + balance_forced + recovery | degraded で通過、ソフトゲートバイパス |
| `test_dual_kill_balance_forced_halt_recovery` | 3 層全部 + recovery | dual_kill_bypassed + ソフトゲートバイパス |
| `test_per_side_halt_with_untick_during_pattern_c` | per-side halt + untick 10 回 | halt カウンタ保持 (273# I3) |
| `test_aggregate_halt_blocks_before_gate_evaluation` | aggregate halt 先行 | gate 評価到達前に停止 |
| `test_cooldown_release_with_dual_kill` | cooldown 経過 + dual-kill | cooldown_released + lot 縮小で再開 |

---

## 6. deprecated CLI 引数削除

### 6.1 問題

`--api-key`/`--api-secret` 引数が残存。プロセスリストに認証情報が露出するセキュリティリスク。
`.env` ファイル経由が正規の認証方法であり、CLI 引数は不要。

### 6.2 変更

- `fill_test_cli.py`: `--api-key`, `--api-secret` の argparse 定義を削除
- `create_adapter()` 呼び出しから `api_key`/`api_secret` kwargs を除去
- `.env` ファイルへのフォールバックが正規パスとなる

---

## 7. type:ignore 検証結果

コードベースの `# type: ignore` を調査。2 件の substantive な使用を確認:

| 箇所 | 理由 | 判定 |
|---|---|---|
| `fill_loop_orchestrator.py` L144 | Mixin クラスのクラスレベル `deque` デフォルト値 | **正当** — mypy は Mixin で解決不能 |
| `event_logger.py` L117 | `sys.stderr = TeeWriter(...)` duck-typing | **正当** — TextIO vs TeeWriter の型互換 |

両方とも正当な使用であり、修正不要と判定。

---

## 8. テスト結果

| スコープ | 結果 |
|---|---|
| 274# 新規テスト (21 件) | ✅ 21 passed |
| v460 全体回帰テスト | ✅ 3764 passed, 0 failed |

### テストファイル内訳

```
test_274_pattern_c_theory_cleanup.py
├── TestPatternCTripleDeadlock        (6 tests)
├── TestGateSoftHardClassification    (5 tests)
├── TestKellyYAMLWiring              (3 tests)
├── TestMacroRegimeYAMLWiring        (2 tests)
├── TestDeprecatedCLIRemoval         (2 tests)
└── TestMarketTheoryDocstrings       (3 tests)
```

---

## 9. 変更ファイル一覧

| ファイル | 行数変化 | 対象 |
|---|---|---|
| `configs/v460/fill_test.yaml` | +12 | MacroRegime 有効化, Kelly YAML 追加 |
| `scripts/v460/lib/daily_drawdown_guard.py` | +22 (docstring) | 市場理論補強 |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +18 (docstring) | 市場理論補強 |
| `scripts/v460/lib/cycle_gate_aggregator.py` | +20 (docstring) | 市場理論補強 |
| `scripts/v460/lib/fill_test_cli.py` | -15 | deprecated CLI 削除 |
| `tests/unit/v460/test_274_*.py` | +400 (新規) | 全テスト |

---

## 10. 残課題 (将来対応)

274# スコープ外で特定された改善候補:

| 優先度 | 課題 | 出典 |
|---|---|---|
| P2 | Pattern E: halt 中 veto 時間減衰速度見直し | 269# |
| P2 | InventoryEscapePolicy / KillGateRescue 責務分離 | 272# |
| P2 | BlockingPolicy 抽出 (orchestrator 行数削減) | 272# |
| P2 | `_is_sell_killed` / `_is_buy_killed` → side パラメータ化 | 272# |
| P2 | Evaluation horizon 拡張 (5min/15min) | 269# |
| P3 | broad `except Exception` → 具体例外化 | scan |
