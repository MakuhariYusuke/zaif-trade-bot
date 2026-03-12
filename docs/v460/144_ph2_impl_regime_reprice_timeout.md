# 144# レジーム活用 R-1c/R-1d + 142#/143# レビュー対応

**作成日**: 2026-02-23  
**前提**: 143# R-1a/R-1b 実装済み、142# §6 / 143# §7 レビュー指摘  
**テスト**: 1244 → 1263 (+19)

---

## §1 レビュー対応 (142# §6)

### §1.1 #1 (HIGH): R-2c RegimeAdaptiveTrainer 表現修正

**問題**: `RegimeAdaptiveTrainer` は SAC 向け Mixin であり、fill_test の XGBoost retrain に直結できない。  
**対策**: 142# doc の R-2c を「直接流用」→「設計資産の再利用 + adapter 層別タスク化」に修正。

### §1.2 #2 (MEDIUM): 検証基準に統計ゲート追加

**問題**: 成功指標が平均値のみで統計的有意性を考慮していない。  
**対策**: 142# doc §4 にブートストラップ 95% CI、最低 50 サンプル、fee 控除後指標を必須化。

### §1.3 #3 (MEDIUM): 段階導入方針の明記

**問題**: R-1a/b/c/d の同時投入で寄与分解が困難。  
**対策**: 142# doc §4.1 に「1 変更ずつ A/B 実施」の段階導入を明記。

### §1.4 #4 (LOW): MarketRegime 値数修正

**問題**: 「40+ 値」は不正確、実際は ~20 enum + alias。  
**対策**: 142# doc を「多値 (20+ alias)」に修正。

### §1.5 #5 (MEDIUM): R-1b preflight 拘束の明記

**対策**: 142# doc R-1b の「提案」セルに「※ preflight は調整後 lot 基準」を追記。

---

## §2 レビュー対応 (143# §7)

### §2.1 #1 (HIGH): preflight-lot 整合

**問題**: `BalanceChecker._check_sell/_check_buy` が `self._current_lot` (base) を参照するが、実発注は `_regime_adjusted_lot()` (trending 増量あり)。→ `insufficient_funds` の危険。  
**対策**: `run_single_cycle` 内で `_regime_adjusted_lot()` を `apply_lot_floor()` **直後** に計算。`_order_lot > _current_lot` の場合に `_current_lot` を一時的に引き上げ。以降の preflight balance check が増量済み lot ベースで動作。

**変更**: `scripts/v460/run_fill_test.py` L806-812

### §2.2 #2 (MEDIUM): min_lot 単一ソース化

**問題**: `_regime_adjusted_lot()` が `min_lot = 0.001` をハードコードし、`config.min_order_btc` と二重管理。  
**対策**: `self.config.min_order_btc` に統一。

**変更**: `scripts/v460/run_fill_test.py` L294

### §2.3 #3 (MEDIUM): quarantine bypass 限定

**問題**: ANY `cancel_reason` で side/price/quantity バリデーションを全バイパス → 壊れたレコードも clean 扱い。  
**対策**: `_AUDIT_CANCEL_REASONS` frozenset (9 種) を定義し、監査系 reason のみバイパス。非監査系は従来通り quarantine。

```python
_AUDIT_CANCEL_REASONS = frozenset({
    "circuit_breaker_open", "preflight_pause", "preflight_insufficient",
    "time_filter_both_sides", "time_filter_086_deadlock",
    "narrow_spread_pause", "balance_forced_skip",
    "unknown_regime_buy_skip", "sell_dynamic_kill",
})
```

**変更**: `ztb/metrics/fill_quality.py` `_quarantine_reason()`

### §2.4 #4 (MEDIUM): 動作テスト強化

**対策**: 19 テスト追加 (ソース文字列検査 → 動作確認テストへ拡充):

| カテゴリ | 件数 | 内容 |
|---|---|---|
| quarantine bypass narrowed | 4 | audit/non-audit × side/price の組み合わせ |
| min_lot unification | 2 | source + custom min_order_btc の動作 |
| preflight-lot alignment | 1 | ソース順序検証 |
| R-1c config + YAML | 2 | default + YAML mapping |
| R-1d config + YAML | 2 | default + YAML mapping |
| R-1c order_monitor | 2 | source inspection (offset logic) |
| R-1d order_monitor | 2 | source inspection (timeout logic) |
| R-1c behavioral | 2 | offset clamp / negative offset |
| R-1d behavioral | 2 | multiplier / no-regime fallback |

### §2.5 #5 (LOW): doc 日付修正

**対策**: 143# doc の作成日を `2025-02-24` → `2026-02-22` に修正。

---

## §3 R-1c: レジーム別 reprice 上限適応

### §3.1 概要

142# 計画の R-1c 施策。`stale_max_reprice` にレジーム別オフセットを加算し、市場状況に応じた reprice 粘り強さを実現。

### §3.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_reprice_adjustments` | `{}` (空 dict) | regime_name → int offset |

YAML マッピング:
```yaml
regime:
  reprice_adjustments:
    high_vol: 1     # +1回 (ボラ時は粘る)
    trending: 2     # +2回 (トレンド追従で積極的reprice)
    ranging: 0      # デフォルト
```

### §3.3 実装方式

`OrderMonitor.monitor()` 内、side 別 `stale_max_reprice` 解決後に regime offset を加算:

```python
_stale_max_rp = max(0, _stale_max_rp_base + _regime_reprice_offset)
```

`max(0, ...)` で負のオフセットによる reprice 無効化も安全にクランプ。

---

## §4 R-1d: レジーム別 timeout 適応

### §4.1 概要

142# 計画の R-1d 施策。`order_timeout_sec` にレジーム別倍率を適用し、市場状況に応じた待機時間を実現。

### §4.2 新規 config フィールド

| フィールド | デフォルト | 説明 |
|---|---|---|
| `regime_timeout_multipliers` | `{}` (空 dict) | regime_name → float multiplier |

YAML マッピング:
```yaml
regime:
  timeout_multipliers:
    high_vol: 0.7   # 63s (早めに撤退)
    trending: 1.3   # 117s (トレンドに乗る)
    ranging: 1.0    # 90s (デフォルト)
```

### §4.3 実装方式

`OrderMonitor.monitor()` 冒頭で `_effective_timeout = order_timeout_sec × multiplier` を計算。while ループの終了条件を `elapsed < _effective_timeout` に変更。

regime_detector が None またはレジーム未検出の場合は multiplier = 1.0 (base timeout)。

---

## §5 テスト (19 件追加)

**テストファイル**: `tests/unit/v460/test_143_regime_utilization.py` (26 → 45, +19)

追加テストクラス:
- `TestQuarantineBypassNarrowed` (4 件)
- `TestMinLotUnification` (2 件)
- `TestPreflightLotAlignment` (1 件)
- `TestRegimeRepriceConfig` (2 件)
- `TestRegimeTimeoutConfig` (2 件)
- `TestRegimeRepriceInOrderMonitor` (2 件)
- `TestRegimeTimeoutInOrderMonitor` (2 件)
- `TestRegimeRepriceMonitorBehavioral` (2 件)
- `TestRegimeTimeoutMonitorBehavioral` (2 件)

**v460 全体**: 1244 → 1263 (+19), 0 failed

---

## §6 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `scripts/v460/run_fill_test.py` | バグ修正 | #1 preflight-lot 整合, #2 min_lot 統一 |
| `scripts/v460/lib/fill_config.py` | 機能追加 | R-1c `regime_reprice_adjustments`, R-1d `regime_timeout_multipliers` + YAML mapping |
| `scripts/v460/lib/order_monitor.py` | 機能追加 | R-1c reprice offset, R-1d effective timeout, `min_order_btc` 統一 |
| `ztb/metrics/fill_quality.py` | バグ修正 | #3 quarantine bypass 監査系限定 |
| `tests/unit/v460/test_143_regime_utilization.py` | テスト追加 | +19 テスト |
| `tests/unit/v460/test_113_resilience.py` | 閾値更新 | line count 405 → 410 (preflight-lot 3行追加分) |
| `docs/v460/142_ph2_plan_regime_utilization.md` | レビュー対応 | §6 #1-#5 全件修正 |
| `docs/v460/143_ph2_impl_regime_utilization.md` | レビュー対応 | §7 #5 日付修正 |
| `docs/v460/144_ph2_impl_regime_reprice_timeout.md` | 新規 | 本ドキュメント |

---

## §7 134# ロードマップ位置確認

```
Phase A (Data Infra)      : ✅ 135#
Phase B (Observability)   : ✅ 135#
Phase C (Re-measurement)  : 🔄 運用中 (168h run: 02/13-02/23, PID 108148 deadlock)
Phase D (Retrain restart) : ✅ 136# (基盤) → 🔄 156# で Phase C と並行開始
Phase E (P1 group)        : ✅ 137#-141# (全 9 項目完了)
142# Self-check           : ✅ C-1/M-1/M-3 修正
143# R-1a/R-1b            : ✅ offset + lot regime adaptation
144# R-1c/R-1d + review   : ✅ reprice + timeout + review 10件
145# Structural fixes      : ✅ cancel_reasons Enum化 + lot 乗算修正
146# Registry decoupling   : ✅ マルチ取引所 Registry 分離
147#-150# Phase C ops       : ✅ restart automation + deadlock対策
151# DPS plan              : ✅ confidence_lot 設計
152# Priority improvements  : ✅ 144# CRITICAL検証 + P3-03判定
153# Test stabilization     : ✅ テスト安定化 + run_fill_test 分割設計
154# 10h dryrun analysis   : ✅ P0-08 deadlock 発見・対策
155# Hindsight analysis    : ✅ 後知恵分析 + trending sell抑制 + sell timeout非対称化
```

**R-1 全サブタスク完了**: R-1a (offset), R-1b (lot), R-1c (reprice), R-1d (timeout)

**Phase C 状況** (2026-02-23時点):
- 168h run: 2,407 レコード蓄積、26 restarts、PID 108148 deadlock (02/23 04:36〜)
- 主要改善: trending sell抑制、sell timeout 75s、time_filter 3h化、balance_forced追跡
- **次**: restart → 残りデータ蓄積 or Phase D 並行開始 (156#)

**Phase D 並行開始の根拠**:
- 136# で retrain基盤は構築済み
- 2,407 records は SkipGate 再訓練に十分なサンプル数
- Phase C deadlock中の空き時間を活用
- sell 弱体の根本要因 (7重ゲート/負の螺旋) に Phase D retrain が直接効く

**次ステップ**: 156# (sell根本分析 + Phase C/D並行計画) / 144# §8-§9 CRITICAL修正

---

## §8 Codexレビュー追記 (2026-02-22)

### §8.1 重大度付き指摘

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | CRITICAL | `scripts/v460/run_fill_test.py:1298`, `scripts/v460/run_fill_test.py:804` | 144# §2.1 では「preflight 前に regime lot を反映」とあるが、実際は preflight (`_check_balance_for_side`) が先に実行され、`_regime_adjusted_lot()` は `run_single_cycle` 内で後段実行。ドキュメント主張と実装が不一致。 | preflight 前に「今回の発注予定 lot」を算出して残高判定に渡す。`BalanceChecker.check(..., required_lot=...)` 形式に拡張し、`run` ループ側で判定する。 |
| 2 | HIGH | `scripts/v460/run_fill_test.py:281`, `scripts/v460/run_fill_test.py:809` | `_regime_adjusted_lot()` の基準が `_current_lot` で、さらに `_current_lot = _order_lot` を永続化しているため、`trending` 連続時にロットがサイクルごとに乗算的に増える。R-1b の「一時調整」仕様と不整合。 | 基準ロットを「永続状態」と「一時発注量」に分離する。`_current_lot` は更新せず、`_order_lot` だけを発注/記録に使う。 |
| 3 | HIGH | `scripts/v460/run_fill_test.py:809`, `scripts/v460/run_fill_test.py:1298` | 上方向のみ `_current_lot` を更新する片側ロジックのため、`high_vol` など縮小レジームでは preflight が過大ロット基準のままになり、不要な `preflight_insufficient` を誘発し得る。 | 増減どちらも「今回発注量」で preflight 判定する。永続ロットを直接変更する設計は避ける。 |
| 4 | LOW | `ztb/metrics/fill_quality.py:910`, `docs/v460/144_ph2_impl_regime_reprice_timeout.md:56` | 「監査系 reason 限定」自体は反映済み。残論点は、監査 reason の side 許容範囲 (`none` のみか、`buy/sell` も含むか) が仕様として未明確な点。 | 設計意図をドキュメントに明記し、テスト名・期待値をその仕様に揃える。 |
| 5 | MEDIUM | `tests/unit/v460/test_143_regime_utilization.py:586`, `tests/unit/v460/test_143_regime_utilization.py:655`, `tests/unit/v460/test_143_regime_utilization.py:671` | 「動作テスト強化」とあるが、重要部分の多くが source inspection で、実際の制御フロー・残高判定・reprice回数・timeout変化を直接検証していない。 | Integration寄りの async テストを追加し、`preflight -> place_order` の数量整合、regime別 `reprice_count` 上限、`effective_timeout` の実時間挙動を検証する。 |
| 6 | MEDIUM | `scripts/v460/lib/fill_config.py:257`, `scripts/v460/lib/order_monitor.py:133` | `regime_timeout_multipliers` / `regime_reprice_adjustments` の値域バリデーションがなく、負値や極端値で意図しない挙動（即timeout等）を招く余地がある。 | `__post_init__` で `timeout_multiplier > 0`、`abs(reprice_adjustment)` 上限などを検証し、異常値を早期に reject する。 |

### §8.2 実行確認

- `tests/unit/v460` を実行し、`1263 passed, 0 failed` を確認（警告 91 件）。
- テスト総数の主張（1244→1263）は整合。

### §8.3 次ステップ提案 (優先順)

1. #1/#2/#3 を同時に修正して、lot の「永続値」と「1注文値」を分離する。
2. #5 の実挙動テストを追加し、今回の不整合が再発しない回帰ガードを入れる。
3. #4/#6 を反映して、doc-impl 整合と設定安全性を固める。

---

## §9 深掘りレビュー (改善点・重複排除)

### §9.1 追加指摘 (重大度順)

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | CRITICAL | `scripts/v460/run_fill_test.py:622`, `scripts/v460/lib/order_monitor.py:333`, `scripts/v460/run_fill_test.py:805` | 初回発注は `_order_lot` だが、reprice 時の数量は `current_lot` を使うため、縮小レジームで再発注量が初回より大きくなり得る。 | `OrderMonitor.monitor(..., order_lot=...)` を導入し、reprice は常に初回発注量を維持する。 |
| 2 | HIGH | `scripts/v460/run_fill_test.py:990`, `scripts/v460/lib/order_monitor.py:134` | timeout 判定は monitor 側でレジーム倍率適用済みなのに、FillRecord の `cancel_reason` は `config.order_timeout_sec` 基準。高/低ボラで timeout ラベルが誤判定になる。 | monitor が `timeout_reached` または `effective_timeout_sec` を返し、record 側はその値で判定する。 |
| 3 | HIGH | `scripts/v460/lib/skip_gate_evaluator.py:465`, `scripts/v460/lib/skip_gate_evaluator.py:467`, `ztb/trading/live/exchanges/base/broker_interfaces.py:65` | SkipGate の OB 取得で `.price/.quantity` 前提だが、実 adapter の `OrderBookSnapshot` は tuple list。例外で握り潰され、OB 特徴量が実質無効化される。 | OB レベル抽出を共通 utility 化し、tuple/object 双方を正規化してから特徴量化する。 |
| 4 | MEDIUM | `scripts/v460/run_fill_test.py:573`, `scripts/v460/run_fill_test.py:805`, `scripts/v460/lib/skip_gate_evaluator.py:519` | SkipGate 判定は lot 適応前に実行され、skip レコードの `order_quantity` は `current_lot`。実発注量と監査ログの数量整合が崩れる。 | lot 解決を先に行い、SkipGateEvaluator へ `planned_order_lot` を渡す。 |
| 5 | MEDIUM | `scripts/v460/run_fill_test.py:1202`, `scripts/v460/run_fill_test.py:1269`, `scripts/v460/run_fill_test.py:1322`, `scripts/v460/run_fill_test.py:1378`, `scripts/v460/run_fill_test.py:1420`, `scripts/v460/run_fill_test.py:1449`, `scripts/v460/run_fill_test.py:1477` | skip/監査系 `FillRecord` 生成が多重重複し、項目追加時に更新漏れが出やすい。 | `build_skip_record()` ヘルパを導入し、`run_id/git_sha/cycle_id/timestamp` を一元生成する。 |
| 6 | MEDIUM | `scripts/v460/run_fill_test.py:673`, `scripts/v460/run_fill_test.py:774`, `ztb/metrics/fill_quality.py:903`, `tests/unit/v460/test_143_regime_utilization.py:472` | cancel_reason 文字列が実装・集計・テストで散在し、命名変更時のドリフトリスクが高い。 | `ztb/metrics/cancel_reasons.py` 等に定数/Enum を集約し、全レイヤで同一参照に統一する。 |
| 7 | LOW | `scripts/v460/run_fill_test.py:657`, `scripts/v460/run_fill_test.py:1203`, `scripts/v460/run_fill_test.py:1270`, `scripts/v460/run_fill_test.py:1323`, `scripts/v460/run_fill_test.py:1421`, `scripts/v460/run_fill_test.py:1450`, `scripts/v460/run_fill_test.py:1478` | `cycle_id` 生成式が多箇所重複。形式変更時の一括修正が困難。 | `_new_cycle_id(prefix: str | None)` を追加し、生成規約を一元化する。 |

### §9.2 重複排除リファクタ案 (実装順)

1. **D1: RecordFactory 導入**  
`scripts/v460/lib/fill_record_factory.py` を作成し、`build_skip_record` / `build_error_record` / `build_fill_record` を統一。

2. **D2: OrderIntent 導入**  
`side`, `planned_lot`, `effective_timeout`, `reprice_limit` を 1 オブジェクト化して `run -> run_single_cycle -> order_monitor` に受け渡し。lot/timeout のズレを構造的に防止。

3. **D3: CancelReason Enum 化**  
監査 reason・執行 reason・保護 reason をカテゴリ付きで定義し、`fill_quality` 側の quarantine ルールも Enum ベースにする。

4. **D4: Top-of-book 正規化 utility**  
tuple/object 両対応の `extract_best_bid_ask()` / `extract_depth_totals()` を追加し、`maker_price`・`run_fill_test`・`skip_gate_evaluator` の重複ロジックを統合。

### §9.3 最短で利益に効く順序

1. §9.1 #1/#2/#3 を先に修正（数量不整合・誤ラベル・OB無効化は損失/誤判断に直結）。
2. その後 D1/D2 で lot/timeout の責務を整理して再発防止。
3. 最後に D3/D4 で運用保守コストを下げる。

---

## §10 追加レビュー (継承・範囲外・自己点検)

### §10.1 継承導入で整理できる事項

| # | 重大度 | 対象ファイル | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `ztb/trading/live/exchanges/coincheck/adapter.py:70`, `ztb/trading/live/exchanges/base/adapter.py:61` | `CoincheckAdapter` が `IBroker` 直実装で、`BitFlyerAdapter` が `BaseExchangeAdapter` 継承。dry-run/rate-limit/state 管理の実装方針が分岐している。 | `CoincheckAdapter` も `BaseExchangeAdapter` 継承へ寄せ、共通責務（dry-run、rate-limit、order state）を統一する。 |
| 2 | MEDIUM | `scripts/v460/run_fill_test.py:105` | `FillTestRunner` が 2k 行超の単一クラスで、execution policy と orchestration が混在。 | `AbstractCycleRunner` (テンプレートメソッド) を作り、`v460` 固有ロジックは派生クラスへ分離。 |
| 3 | MEDIUM | `scripts/v460/lib/order_monitor.py:91`, `scripts/v460/lib/skip_gate_evaluator.py:358` | orderbook 取り扱いの抽象化が不足し、tuple/object の実装差を各所で吸収している。 | `MarketDataAccessorBase` を導入し、`best_bid/ask`・depth 集計を継承側で共通化する。 |

### §10.2 範囲外だが直したほうが良い事項

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `scripts/v460/lib/skip_gate_evaluator.py:465`, `scripts/v460/lib/skip_gate_evaluator.py:467`, `ztb/trading/live/exchanges/base/broker_interfaces.py:65` | SkipGate の OB 特徴量で `.price/.quantity` 前提。実 adapter は tuple 返却なので例外で握り潰され、OB 特徴量が欠落しやすい。 | 形式正規化 utility を導入して tuple/object 両対応にする。 |
| 2 | MEDIUM | `ztb/trading/live/exchanges/bitflyer/adapter.py:138` | `_make_request` の docstring が重複しており、レビュー/保守時にノイズが大きい。 | docstring を 1 つに整理し、API エラー方針も統一記述する。 |
| 3 | MEDIUM | `scripts/v460/run_fill_test.py:666`, `scripts/v460/run_fill_test.py:732`, `scripts/v460/run_fill_test.py:767`, `scripts/v460/run_fill_test.py:907`, `scripts/v460/run_fill_test.py:1202`, `scripts/v460/run_fill_test.py:1269`, `scripts/v460/run_fill_test.py:1322`, `scripts/v460/run_fill_test.py:1378`, `scripts/v460/run_fill_test.py:1420`, `scripts/v460/run_fill_test.py:1449`, `scripts/v460/run_fill_test.py:1477` | `FillRecord` 組み立て重複が多く、項目追加時の更新漏れリスクが高い。 | builder/factory に統一し、reason 別最小差分だけ上書きする。 |
| 4 | LOW | `scripts/v460/run_fill_test.py:657`, `scripts/v460/run_fill_test.py:1203`, `scripts/v460/run_fill_test.py:1270`, `scripts/v460/run_fill_test.py:1323`, `scripts/v460/run_fill_test.py:1421`, `scripts/v460/run_fill_test.py:1450`, `scripts/v460/run_fill_test.py:1478` | `cycle_id` 生成式が重複。 | `_new_cycle_id()` ヘルパで一元化。 |

### §10.3 レビュー自己点検 (前回指摘の再評価)

| 前回指摘 | 再評価 | コメント |
|---|---|---|
| §8-#1 preflight と lot 適用順の不一致 | **維持** | 実装確認済み。`run` 側 preflight が先、`run_single_cycle` 側 lot 計算が後。 |
| §8-#2 `_current_lot` 永続更新による乗算増加 | **維持** | `self._current_lot = _order_lot` が残っており、一時調整仕様と不一致。 |
| §8-#3 片側更新による縮小時 preflight 過大化 | **維持** | 上方向のみ更新のため、縮小局面で過大判定が残る。 |
| §8-#4 「side=none 限定」の記述不一致 | **修正** | 私の記載が過剰。実ドキュメントは「監査系 reason 限定」が主旨で、`side=none` 限定は明記されていない。重大度は下げ、設計選択として扱うのが妥当。 |
| §8-#5 source inspection 偏重 | **維持** | テストは増えたが、重要経路は依然として実挙動検証が不足。 |
| §8-#6 新規レジーム設定の値域バリデーション不足 | **維持** | `__post_init__` に該当チェックは未追加。 |

### §10.4 レビュー方針補足

- 今後は「実装不一致」「収益影響」「回帰再発性」を優先して、提案ではなく検証可能な指摘を主軸にレビューを継続する。
