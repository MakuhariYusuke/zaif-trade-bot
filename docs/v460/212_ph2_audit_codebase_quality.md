# 212# コードベース品質監査レポート

> **目的**: Codex / Gemini 外部レビュー用の改善ポイント一覧
> **実施日**: 2026-03-02
> **対象**: v460 全体 (geopolitical event 対応は 211# §8 に記載済み、本書は技術面のみ)

---

## 0. エグゼクティブサマリー

| カテゴリ | 件数 | 最高重要度 |
|---|---|---|
| **例外処理の欠陥** | bare `except:` 22箇所, サイレント swallow 50+箇所 | **CRITICAL** |
| **メモリリーク / パフォーマンス** | `pop(0)` O(n) 6箇所, truncation 漏れ 2箇所 | HIGH |
| **Hot-reload カバレッジ不足** | 実運用で即変更したいフィールド 15+ 漏れ | HIGH |
| **テストカバレッジ空白** | ライブ取引 36.7%, 本番系 37.5%, 0テストファイル 18件 | HIGH |
| **巨大ファイル (SRP 違反)** | >1500行 14ファイル, >1000行 13ファイル追加 | MEDIUM |
| **Any 型蔓延** | コアモジュール 5+, scripts 30+ | MEDIUM |
| **マジックナンバー** | live_trader.py 7箇所, reward 系 3箇所 | MEDIUM |
| **未実装 TODO (本番 API)** | bridge.py / trading_api.py 計 8箇所 | LOW* |
| **typing 旧記法** | `Dict/List/Tuple` from typing (20+ファイル) | LOW |

> *bridge.py / trading_api.py の TODO は coincheck 直接実装のため現状不使用。リスクは低いが dead code 整理の対象。

---

## 1. 例外処理の欠陥 (CRITICAL)

### 1.1 裸の `except:` (30箇所) — ✅ 修正済

`KeyboardInterrupt` / `SystemExit` も飲み込む最危険パターン。
全 30 箇所を `except Exception:` に一括置換済み。

| ファイル | 行 | 用途 |
|---|---|---|
| `ztb/python/quick_backtest.py` | L183, 260, 266 | バックテスト |
| `ztb/python/sac_backtest.py` | L81 | SAC バックテスト |
| `ztb/python/backtest_sac_v434_2.py` | L132 | SAC バックテスト |
| `ztb/ops/reports/generate_weekly_report.py` | L87 | レポート生成 |
| `ztb/ops/reports/feature_report.py` | L55, 131 | Feature レポート |
| `ztb/features/models/sac/sac_v427_feature_engineering.py` | L2133 | Feature 工学 |
| `ztb/features/models/v437/engine/sac_v437_feature_engineering.py` | L264, 318 | Feature 工学 |
| `ztb/analysis/models/unified_backtest.py` | L304 | バックテスト分析 |
| `scripts/v457/train_v457_*.py` | 各1箇所 | 学習スクリプト |
| `scripts/v455/find_missing_data.py` | L38 | データツール |
| `scripts/v455/run_backtest.py` | L245 | バックテスト |
| `scripts/utils/check_env_attrs.py` | L28 | ツール |
| `scripts/analysis/*.py` | 3箇所 | 分析ツール |
| `scripts/data_processing/collect_all_config_parameters.py` | L166 | データ処理 |

**対策**: `except Exception as e:` + `logger.exception(e)` に一括置換。scripts 配下はリスク低だが統一すべき。

### 1.2 サイレント `except Exception:` (特に危険な箇所) — ✅ trainer.py 修正済 (`553895a0e`)

| ファイル | 該当数 | リスク | 状態 |
|---|---|---|---|
| `ztb/training/unified_trainer/trainer.py` | **12箇所** | **学習中のエラーが invisible** — 最優先 | ✅ `logger.debug` 追加 (`553895a0e`) |
| `ztb/utils/checkpoint.py` | 7箇所 | チェックポイント破損を検知不能 | 未着手 |
| `ztb/utils/seed_manager.py` | 9箇所 | seed 設定失敗が silent | 未着手 |
| `ztb/utils/env_metrics.py` | 10箇所 | メトリクス計算失敗が silent | 未着手 |
| `ztb/utils/run_metadata.py` | 5箇所 | メタデータ破損検知不能 | 未着手 |

**対策**: `trainer.py` は `except Exception as e: logger.debug(...)` に変更済 (12箇所)。
  残り3箇所 (import guard) はオプショナルインポートパターンのため変更不要。
`safety.py` (7箇所) は安全装置なので設計上許容。

---

## 2. メモリリーク・パフォーマンス (HIGH)

### 2.1 `list.pop(0)` → `deque(maxlen=N)` (O(n) → O(1)) — ✅ 修正済

| ファイル | 行 | 対象リスト |
|---|---|---|
| `ztb/trading/signal/signal_guidance_system.py` | L134, 141 | `price_trend` / `volume_trend` |
| `ztb/training/callbacks/shared/base/learning_callback.py` | L236, 960 | `_error_history` / `adaptation_history` |
| `ztb/realtime_optimization/adaptive_learning_system.py` | L179 | `learning_experiences` |
| `ztb/trading/environment/components/reward/base_reward_calculator.py` | L295 | `_recent_actions` |
| `ztb/analysis/adaptive_confidence_adjuster.py` | L373 | `threshold_history` |
| `ztb/analysis/regime/basic_regime_detector.py` | L81 | `price_history` |

### 2.2 truncation 漏れ (無限成長リスク) — ✅ 修正済

| ファイル | 行 | 対象 |
|---|---|---|
| `ztb/trading/v433_integrated_system.py` | L388 | `metrics_history: List[Dict]` — `decision_history`/`outcome_history` は制限済みだが `metrics_history` のみ漏れ |
| `ztb/utils/rate_limiter.py` | L92 | `SlidingWindowRateLimiter.requests: deque()` — `maxlen` 未設定 |
| `ztb/utils/rate_limiter.py` | L203 | `MultiRateLimiter.limiters: Dict` — 動的キーで無限増殖リスク |

---

## 3. Hot-Reload カバレッジ不足 (HIGH)

`FillTestConfig` 318 フィールド中、hot-reloadable は 85 フィールド。
**ランタイム変更の恩恵が高いのに漏れているフィールド**:

### 3.1 即時対応 (HIGH)

| フィールド群 | 理由 |
|---|---|
| `loss_cooldown_threshold_bps`, `loss_cooldown_interval_mult`, `loss_boost_offset_mult` | 大損後の防御パラメータ。本番で動的調整したい最頻ケース |
| `toxic_fill_veto_threshold_bps`, `toxic_fill_veto_cycles` | Toxic fill 防御は市況依存で即座に緩和/強化したい |
| `trending_sell_as_offset_enabled`, `trending_sell_offset_boost_factor` | 一部は入っているが on/off トグルが欠落 |
| `one_sided_consecutive_limit`, `one_sided_consecutive_interval_mult` | 片側枯渇制限は市場状況で即調整必要 |

### 3.2 短期対応 (MEDIUM)

| フィールド群 | 理由 |
|---|---|
| `soft_drawdown_interval_multiplier` | Soft DD 時の interval 乗数 |
| `velocity_skip_as_offset_enabled`, `velocity_offset_boost_factor` 等 | velocity 系メインは入っているが toggle 一部漏れ |
| `sell_velocity_skip_enabled/threshold`, `buy_velocity_skip_enabled/threshold` | velocity skip on/off |
| `balance_forced_rescue_enabled/offset_mult`, `skip_balance_forced`, `balance_forced_deadlock_limit` | Balance forced 救済系 |
| `skip_sell_trending`, `skip_buy_unknown_regime`, `skip_sell_unknown_regime` | ガード on/off — regex マッチ漏れの可能性 |

---

## 4. テストカバレッジ空白 (HIGH)

### 4.1 全体統計

| 指標 | 値 |
|---|---|
| テストファイル総数 | 536 |
| テスト関数総数 | 6,692 |
| テスト対応ありソース (名前マッチ) | 45.9% |
| 0テスト空ファイル | 18件 |
| placeholder 1テストファイル | 86件 |

### 4.2 ビジネスリスク上位 (テストカバレッジ低)

| カバレッジ | ディレクトリ | リスク |
|---|---|---|
| **36.7%** | `ztb/trading/live/` | **ライブトレード — 実金銭リスク最大** |
| **37.5%** | `ztb/trading/production/` | 障害時安全機構 (emergency_stop, recovery_system, rollback_manager) |
| **5.3%** | `ztb/training/binary_search/` | 19個のオプティマイザ、テスト1つ |
| **0.0%** | `ztb/ops/validation/` | go_nogo_check — リリースゲート未テスト |
| **0.0%** | `ztb/trading/environment/components/reward/` | 報酬計算 10 コンポーネント |
| **0.0%** | `ztb/analysis/backtest/` | backtester 群 13 ファイル |
| **0.0%** | `ztb/analysis/sessions/` | ログ分析 10 ファイル |

### 4.3 既知の問題テスト

| テスト | 問題 |
|---|---|
| `test_088_features.py` | brittle — ソース文字列直比較。`CR` 定数ベースへの更新が 145# で LOW 残存 |
| `test_113_resilience.py` | `run_single_cycle` 行数ガードが機能追加ごとに閾値更新 (400→600)。テストの意味が薄い |
| `test_pnl_invariants.py` | xfail — 環境バグ (ポジション変更直後 PnL=0)。`Issue #XXX` 番号未記入で放置 |
| `test_algorithm_trainer.py` | skip 3件 — `PPOAlgorithmTrainer` 旧 API 参照のまま放置 |
| `test_supervise_1m.py` / `test_watch_1m.py` | skip — モジュール未実装。placeholder のまま |

---

## 5. 巨大ファイル (SRP 違反) (MEDIUM)

### 5.1 >1500行 (分割が急務)

| ファイル | 行数 | 推奨分割 |
|---|---|---|
| `ztb/analysis/comparative/analyze_backtest.py` | 2715 | Report / Metrics / Visualization |
| `ztb/training/unified_optimizer.py` | 2291 | Optimizer / SearchStrategy / ResultAnalyzer |
| `ztb/training/unified_trainer/trainer.py` | 2222 | TrainingLoop / Evaluation / Checkpoint / Logging |
| `ztb/metrics/metrics.py` | 2036 | PerformanceMetrics / RiskMetrics / TradeMetrics |
| `ztb/trading/environment/components/calculators/reward_calculator.py` | 1902 | RewardCalc / SignalIntegrator / PenaltyCalc |
| `scripts/v460/ml/retrain_scheduler.py` | 1851 | Scheduler / Trainer / DataPipeline |
| `ztb/trading/strategies/action_signal_guide/action_signal_guide.py` | 1835 | SignalGuide / ActionSelector / IndicatorAnalyzer |
| `ztb/training/unified_trainer/algorithms/sac_trainer.py` | 1745 | SACTrainer / HyperparamTuner / EarlyStopper |
| `ztb/utils/talib_wrapper.py` | 1619 | TalibCache / IndicatorGroups / Validation |
| `ztb/trading/live_trader/live_trader.py` | 1606 | LiveTrader / OrderManager / PositionTracker |
| `ztb/trading/environment/heavy_env/core.py` | 1568 | EnvCore / Observation / StepLogic |
| `scripts/v460/lib/fill_loop_orchestrator.py` | 1494* | Orchestrator / KillSwitch / CycleSleepManager |
| `scripts/v460/lib/fill_config.py` | 1302* | Config / Validation / HotReload |

> *fill_loop_orchestrator / fill_config は v460 コアのため分割はリスク高。Mixin パターンで既に一部分割済み。

---

## 6. Any 型使用 (MEDIUM)

### 6.1 コアモジュール

| ファイル | 出現数 | 改善案 |
|---|---|---|
| `ztb/utils/analysis_errors.py` | ~13 | `TypeVar` + `@overload` |
| `ztb/utils/type_validation.py` | ~12 | `TypeVar` / `Protocol` |
| `ztb/utils/performance_utils.py` | ~12 | `ParamSpec` (PEP 612) |
| `ztb/utils/config_loader.py` | 全関数 | `TypedDict` / Pydantic model |
| `ztb/utils/observability.py` | 3 | `TypeVar` |
| `ztb/utils/system_utils.py` | 1 | `types.ModuleType` |

### 6.2 Scripts (30+ 箇所)

`scripts/v459/run_phase_c.py` (30+) が最大。`Protocol` 定義で解消可能だが優先度低。

---

## 7. マジックナンバー (MEDIUM)

### 7.1 本番取引に直結 (live_trader.py)

| 行 | 値 | 推奨 |
|---|---|---|
| L296 | `timeout=5` (ticker API) | `LiveTraderConfig.ticker_timeout` |
| L507 | `recovery_timeout=60.0` | `LiveTraderConfig.recovery_timeout` |
| L509 | `timeout=10.0` | `LiveTraderConfig.order_timeout` |
| L622, 716 | `time.sleep(60)` | `LiveTraderConfig.retry_interval` |
| L1291 | `timeout=10` | 統一 |
| L1432 | `time.sleep(2)` | `LiveTraderConfig.order_poll_interval` |

### 7.2 報酬計算 (reward_calculator.py)

| 行 | 値 | 推奨 |
|---|---|---|
| L359 | `percentile_threshold = 0.8` | `RewardConfig.percentile_threshold` |
| L797-818 | `ichimoku_weight=5.0`, `adx_weight=2.0` 等 | レジーム別 config |

---

## 8. グローバル状態 / スレッドセーフティ (LOW)

| ファイル | 行 | 問題 |
|---|---|---|
| `ztb/utils/seed_manager.py` | L185-191 | グローバル `_seed_manager` — Lock なし |
| `ztb/utils/notify/discord.py` | L244-273 | グローバル `_default_notifier` — 遅延初期化、Lock なし |
| `ztb/utils/fault_injection.py` | L114-120 | グローバル `_fault_injector` — 同上 |

> 現状シングルスレッド実行のため実害なし。将来のマルチスレッド化時にリスク。

---

## 9. 未実装 TODO (本番 API) (LOW)

| ファイル | 行 | 内容 |
|---|---|---|
| `ztb/trading/environment/bridge.py` | L627, 690, 722 | `TODO: Implement actual Zaif API call` ×3 |
| `ztb/live_trading/trading_api.py` | L80, 238, 262, 296 | `TODO: ccxtライブラリを使用したZaif API統合` ×4 |
| `ztb/trading/live/core/reconciliation.py` | L581-599 | Coincheck reconciliation 3メソッド未実装 |

> 現在 coincheck 直接統合のため不使用。dead code 整理対象。

---

## 10. HACK コメント (注意)

| ファイル | 行 | 内容 |
|---|---|---|
| `scripts/v457/dry_run.py` | L35-36, 63 | `hacked config.py` / filter bypass / dynamic attributes |

> v457 レガシー使用続行なら正式リファクタリング要。

---

## 11. Deprecated Python パターン (LOW)

| パターン | 該当数 | 対策 |
|---|---|---|
| `from typing import Dict, List, Tuple, Optional` | 20+ファイル (主に `ztb/utils/`) | Python 3.9+ の `dict`, `list`, `tuple`, `X \| None` に統一 |

> `fill_config.py` では既にモダン記法を使用。`ztb/utils/` 配下が旧記法。

---

## 12. 既存残課題 (他 issue からの引継ぎ)

| ID | 出典 | 内容 | 対応状況 |
|---|---|---|---|
| H4 | 210# §6 | SellDynamicKillManager rolling PnL window 非永続化 | ✅ 実装済 (`49e1253c2`) |
| spread staleness 60s | 204# | ハードコード → Config 外部化 | LOW |
| test_088 | 145# | brittle string 比較 → CR 定数ベース | LOW |
| test_113 | 継続 | 行数ガードテストの形骸化 | テスト設計見直し |
| test_pnl_invariants | 不明 | Issue #XXX 番号未記入 | 放置 |
| P0-A | 211# §8 | Operator Alert Flag (地政学イベント対応) | Codex/Gemini レビュー待ち |

---

## 13. 対応優先度ロードマップ

### P0: 即時対応 (収益・安全に直結)

1. **Hot-reload フィールド追加** — `loss_cooldown_*`, `toxic_fill_veto_*`, `one_sided_*` — 変更量: ~10行
2. **P0-A: Operator Alert Flag** — 211# §8 参照 — 変更量: ~50行

### P1: 短期対応 (1 週間以内)

3. **`pop(0)` → `deque(maxlen=)`** — 7箇所一括 — ✅ 実装済
4. **trainer.py サイレント swallow** — 15箇所の `except Exception` 精査 — 変更量: ~50行
5. **bare `except:` 一括修正** — 30箇所 — ✅ 実装済

### P2: 中期対応 (2-4 週間)

6. **ライブ取引テスト拡充** — `ztb/trading/live/` カバレッジ 36.7% → 60% 目標
7. **本番系テスト拡充** — `ztb/trading/production/` カバレッジ 37.5% → 60% 目標
8. **live_trader.py マジックナンバー config 化** — 7箇所

### P3: 長期改善

9. **巨大ファイル分割** — 2000行超 4ファイルから段階的に
10. **Any 型削減** — コアモジュール 5 ファイルから
11. **typing モダン化** — ztb/utils/ 20+ファイル
12. **dead code 整理** — bridge.py / trading_api.py の Zaif TODO
13. **0テスト / placeholder テスト 104 ファイルの整理** — 削除 or 実装

---

## 14. Codex / Gemini レビューへの依頼事項

上記の優先度判断・分割粒度・対策案について以下の観点でレビューを依頼:

1. **優先度の妥当性** — 収益最大化の大義に照らして順序は正しいか
2. **見落とし** — 上記以外に重大なリスクや改善点はあるか
3. **分割戦略** — 巨大ファイルの分割パターンは適切か (Mixin vs Composition vs Strategy)
4. **テスト戦略** — ライブ系のテスト拡充はモック vs integration のどちらを優先すべきか
5. **P0-A 設計** — Operator Alert Flag のファイルタッチ方式は適切か、より良い IPC があるか
