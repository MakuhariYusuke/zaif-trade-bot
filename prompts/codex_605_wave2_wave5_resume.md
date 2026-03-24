# Codex Task: 551# Wave 2-5 残課題の再開

> 対象: 037# セッション (Wave 2-5 実行計画)
> 前提: AGENTS.md の規約に従うこと。`git add .` 禁止、個別指定。コミットは `--no-verify`。
> 正本: 551# (Wave計画), 521# (母艦), 550# (maker_price設計), 557# (報酬系)
> 日付: 2026-03-25
> 背景: 602#-604# の緊急バグ修正が割り込み、037# セッション (SessionLog 037-564 + 3/24分) が中断。
>       3/24 最終作成物は analysis typing sweep (hindsight_filter, stopgap_daily_report の共通 loader 統一)。

---

## 実行順 (551# に従う)

**Wave 2 → Wave 3 → Wave 4 → Wave 5 の順で進める。**

優先規則:
1. 既存の shared type があるか
2. `Any` を増やさず `Protocol` / `TypeAlias` / `cast` で止められるか
3. focused pytest で守れるか
4. 並行差分に触れずに切り出せるか → 満たさない場合は後ろに回す

---

## Wave 2A: `maker_price.py` veto/cache ownership 最終整理

### 現状

- ~1900行、`__slots__` で65+属性管理
- `compute()` は stage 別メソッドへの分割済み:
  - `_apply_as_reservation_shift()`, `_apply_regime_boosts()`, `_apply_spread_adaptive()`
  - `_apply_kyle_lambda()`, `_apply_amihud_illiq()`, `_apply_volatility_guard()`
  - `_apply_cross_venue_lead_lag_guard()`, `_apply_imbalance_risk()`
  - `_apply_loss_boost()`, `_apply_ffd_boost()`
- pure helper は `ztb/trading/pricing/` に抽出済み (10+モジュール)
- 3 Mixin に責務分割済み: `maker_risk_guards.py`, `maker_microstructure.py`, `maker_regime_boost.py`
- stage seed / final serialize / preflight-cache-resolve helper化まで完了 (037-561〜564)

### 残タスク

1. **veto/telemetry/cache の update 点を local helper にさらに寄せる**
   - `compute()` 内に残る inline ownership (veto判定、telemetry 記録、cache更新) を特定し、helper 呼び出しに抽出
   - 対象: veto telemetry と cache ownership の update 点

2. **source-contract test を helper/stage 契約ベースへ寄せる**
   - 現在の direct 実装断片を見るテストを、helper/stage の契約を見る形に更新
   - optional stage (`kyle` / `amihud` / `imb_risk` / `buy_as_guard`) の helper 契約テスト追加

3. **Done 基準** (以下すべて満たす):
   - `compute()` が「preflight」「stage pipeline」「finalize」の3ブロックで読める
   - source-inspection test が helper/stage 契約を見る
   - optional stage の contract が明示

4. **止めどころ**: public shape や source-inspection 契約を崩し始めたら、そこで止める。state object化や大分割は Wave 2 の守備範囲外。

### 検証

```bash
# focused
python -m pytest tests/unit/v460/test_regime_detector.py tests/unit/v460/test_145_structural_fixes.py tests/unit/v460/test_239_feasible_quote.py tests/unit/v460/test_240_toxicity_budget.py -x --tb=short --no-cov

# targeted mypy
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py scripts/v460/lib/maker_price.py
```

---

## Wave 2B: `ab_judgment.py` result/report ownership 最終整理

### 現状

- ~950行
- `ABJudgmentResult` (272行目〜) を中心に統計検定・bootstrap・matched comparison 統合
- pure rule は `judgment_rules.py` に分離済み
- result 初期化 + insufficient early return + primary criteria append の local helper化まで完了

### 残タスク

1. **`ABJudgmentResult` orchestration の最終整理**
   - statistical comparison payload の ownership 見直し (helper 側 vs script 側)
   - result container / statistical payload / summary-report 文面 の3層をさらに固定

2. **reporting 文面は最後まで script ownership に残す**
   - `judgment_rules.py` は pure rule に留まり、script 側は result/report の責務だけ持つ

3. **Done 基準**:
   - pure rule / local orchestration / reporting の3層が混ざらない
   - `judgment_rules.py` は pure rule のみ

### 検証

```bash
python -m pytest tests/unit/v460/test_regime_detector.py -k "ABJudgment or ab_judgment" -x --tb=short --no-cov
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py scripts/v460/lib/ab_judgment.py
```

---

## Wave 3: telemetry/diagnostics payload 整列

### 前提

Wave 2 が「完了判定可能」になってから着手。理由: `maker_price` と `ab_judgment` が downstream test / telemetry の基準点。

### 対象モジュール

| モジュール | 焦点 |
|-----------|------|
| `ztb/training/sac/debug.py` | training stats canonical化 |
| `ztb/training/reward_stats.py` | training_stats payload & reward extraction |
| `scripts/v460/ml/sac_retrain_scheduler.py` | debug/history 統合 |
| `scripts/v460/ml/heavy_env.py` | `_sync_terminal_reward_outputs()` / `_append_reward_diagnostics_to_info()` 標準化 |
| `ztb/trading/environment/components/calculators/reward_calculator.py` | snapshot 契約, mutable payload alias 回避 |

### タスク

1. **payload shape 統一**: training stats / reward telemetry の field 符号・意味を module 間で揃える
2. **`RewardCalculator.get_last_reward_components()` を snapshot 契約へ**
   - mutable payload alias を避ける (現在は dict 参照を返している可能性)
   - `_last_reward_components` の ownership を一本化
3. **heavy_env**: `reward_components` と `info` の責務分離を横展開
4. **leak warning / rss warning / cache entry count** の観測を一貫化
5. **callback/reporting の `reward_components` 取得経路** を shared helper に寄せる

### Done 基準

- telemetry field の符号/意味が module ごとにぶれない
- memory diagnostics が `utils/training` helper から辿れる
- `RewardCalculator` snapshot 契約がテストで守られている

### 入口確認

```bash
# Wave 3 入口: 変更対象の targeted mypy
.venv/Scripts/python.exe scripts/quality/run_targeted_mypy.py ztb/training/sac/debug.py ztb/training/reward_stats.py scripts/v460/ml/sac_retrain_scheduler.py

# focused test
python -m pytest tests/unit/reward/ tests/unit/v460/test_558_reward_unification.py -x --tb=short --no-cov
```

### 正本

- 557# (報酬系詳細計画)

---

## Wave 4: broad 前の固定費削減 (残件)

### 完了済み (current suite)

- `TemporaryDirectory()` / `NamedTemporaryFile()` / `time.sleep()` の grep hit は current suite で解消済み
- broad wall time: `53.07s → 31.21s` (037-081 時点)

### 残タスク

| テストファイル | 現状 | 改善 |
|---------------|------|------|
| `tests/unit/environment/test_reward_function.py` | `_make_reward_env()` helper 使用、pytest fixture 未採用 | shared env fixture 化 |
| `tests/unit/environment/test_env_randomization_integration.py` | `unittest.TestCase.setUpClass` 使用 | pytest 化 + setup 共通化 |
| `tests/training/test_lagrange_integration.py` | `test_custom_ppo_lagrange_creation` ~4.5秒 | creation smoke 1本だけ実体, 残りは patched constructor |
| `tests/unit/training/test_unified_optimizer.py` | real Optuna path が `BayesianOptimizer` テストに残る | 「本当に Optuna 実行が必要な1本」に絞り込み |

### 追加確認事項

- `tests/unit/environment/test_reverse_as_close.py`: module-scope env fixtures (6個) で再利用済み → **完了扱い**
- real-data setup の fixture 再利用 / sample cap 見直しが含まれうる

### 検証

```bash
# focused
python -m pytest tests/unit/environment/test_reward_function.py tests/unit/environment/test_env_randomization_integration.py tests/training/test_lagrange_integration.py --durations=10 --tb=short --no-cov

# broad (Wave 4 完了確認用)
python -m pytest tests/training/ tests/integration/ tests/unit/environment/ tests/unit/analysis/ tests/unit/training/ -q --no-cov --tb=short --show-capture=no --maxfail=5 --durations=40
```

---

## Wave 5: broad 最終確認

### 前提

Wave 2-4 が概ね完了してから実行。

### タスク

1. **filtered broad 実行** (test_113/152/260 除外, test_306 YAML deselect)
   ```bash
   python -m pytest tests/unit/v460/ -q --no-cov --tb=short --show-capture=no --maxfail=10 --durations=20
   ```
2. **top durations 再抽出** — 残る支配的 hotspot があれば追加で触る
3. **v460 broad 完走確認** — 前回 `4762 passed, 2 skipped` だが `KeyboardInterrupt` で完走未確認
4. **区切り記載**: 521# / 037# / 551# に完了サマリーを残す

### Done 基準

- broad が安定して完走する
- 残課題が「future に送るもの」と「今やるべきもの」に明確に分かれる

---

## 037# セッションログ記載

各 Wave の作業完了時に `docs/v460/037_phg_rpt_refactoring_session_log.md` に以下を追記:

```markdown
## 2026-03-25 Wave N 作業タイトル
- 変更ファイル一覧
- focused test 結果
- targeted mypy 結果 (Wave 2/3)
- broad 結果 (Wave 5)
```

セッション番号は既存の最大値 + 1 をインクリメンタルに採番する。
037# ファイル内のセッション番号体系 (037-NNN と日付見出しが混在) に注意し、既存パターンに合わせること。

---

## 参照ドキュメント

| Doc # | 用途 |
|-------|------|
| 551# | Wave 計画の正本 |
| 521# | 全体母艦 (master architecture) |
| 550# | maker_price state/stage 境界 |
| 557# | 報酬系詳細 (Wave 3 正本) |
| 579# | Phase 5.5 アーキテクチャ負債 (参考) |
| 589# | targeted mypy 入口 |
| 037# | 実行記録ハブ |
