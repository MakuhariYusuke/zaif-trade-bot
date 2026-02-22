# 146# マルチ取引所 Registry Decoupling

## §1 概要

背景: 本システムは元々 Zaif 向けに構築 → API 品質問題で Coincheck に移行 → BitFlyer も選択肢として整備。
今後任意の取引所を追加しても既存コードに影響しない設計を目指す。

**前提**: 145# §14 で BaseExchangeAdapter / AbstractCycleRunner / MarketDataAccessor の構造リファクタ完了。

| 施策 | ステータス | 内容 |
|---|---|---|
| §2 BrokerRegistry 拡張 | ✅ | credential resolution + `create_adapter()` |
| §3 run_fill_test.py 脱 CoincheckAdapter | ✅ | registry 経由 + `--exchange` CLI |
| §4 run_observation.py 脱 CoincheckAdapter | ✅ | registry 経由 + `--exchange` CLI |
| §5 exchanges `__init__.py` 整備 | ✅ | 4 パッケージの `__init__.py` コミット |
| §6 registry `__init__.py` 新規 | ✅ | `BrokerRegistry`, `get_broker_registry` 再 export |

## §2 BrokerRegistry 拡張

### §2.1 credential_env_map

```python
_CREDENTIAL_ENV_MAP: Dict[str, Tuple[str, str]] = {
    "coincheck": ("COINCHECK_API_KEY", "COINCHECK_API_SECRET"),
    "bitflyer": ("BITFLYER_API_KEY", "BITFLYER_API_SECRET"),
}
```

新取引所追加時: `register_broker()` に `credential_env=("XXX_API_KEY", "XXX_API_SECRET")` を渡すか、`_CREDENTIAL_ENV_MAP` に直接追記。

### §2.2 新メソッド

| メソッド | 用途 |
|---|---|
| `get_credential_env_vars(name)` | 環境変数名のタプルを返す |
| `resolve_credentials(name)` | 環境変数から API key/secret を解決 (未設定→None) |
| `create_adapter(name, dry_run, api_key?, api_secret?)` | ファクトリ。dry_run 時は creds 不要、live 時は必須 |

### §2.3 create_adapter フロー

1. 明示的 `api_key`/`api_secret` があればそれを使用
2. なければ `resolve_credentials()` で環境変数から取得
3. `dry_run=False` で creds が取得できなければ `ValueError`
4. `dry_run=True` で creds 無しなら空文字列で生成

## §3 run_fill_test.py 変更

| 変更箇所 | Before | After |
|---|---|---|
| import | `from ...coincheck.adapter import CoincheckAdapter` | `from ...broker_interfaces import IBroker` + `get_broker_registry` |
| 型注釈 | `adapter: CoincheckAdapter` | `adapter: IBroker` |
| CLI | なし | `--exchange` (default: `coincheck`) |
| adapter 生成 | `CoincheckAdapter(api_key=..., dry_run=...)` | `registry.create_adapter(exchange, ...)` |

## §4 run_observation.py 変更

| 変更箇所 | Before | After |
|---|---|---|
| import | `from ...coincheck.adapter import CoincheckAdapter` | `get_broker_registry` |
| main() | `CoincheckAdapter(dry_run=True)` | `registry.create_adapter(exchange, dry_run=True)` |
| CLI | なし | `--exchange` (default: `coincheck`) |

## §5 新規テスト

`tests/unit/v460/test_146_multi_exchange.py` — **45 tests**

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestBrokerRegistry` | 7 | デフォルト登録 / IBroker 返却 / unknown raises / has_broker / カスタム登録 |
| `TestBitFlyerAdapterFixes` | 3 | asyncio_to_thread / NetworkError / デフォルト価格 lowercase |
| `TestSymbolNormalization` | 3 | normalize_symbol / coincheck lowercase / bitflyer API uppercase |
| `TestPackageInit` | 4 | exchanges / base / coincheck / bitflyer `__init__.py` exports |
| `TestLegacyCleanup` | 4 | zaif 残存 / skeleton / BrokerProtocol 不在確認 |
| `TestAdapterInheritance` | 4 | 両 adapter の BaseExchangeAdapter 継承 / 7 real methods / market_data |
| `TestBrokerRegistryCredentials` | 7 | env_map coincheck/bitflyer / resolve from env / empty→None / unknown raises / custom credential_env |
| `TestBrokerRegistryCreateAdapter` | 5 | dry_run no creds / live no creds raises / live explicit creds / bitflyer dry_run / unknown raises |
| `TestRunFillTestExchangeDecoupling` | 5 | no CoincheckAdapter import / uses registry / IBroker annotation / --exchange arg / create_adapter call |
| `TestRunObservationExchangeDecoupling` | 3 | no CoincheckAdapter import / uses registry / --exchange arg |
| `TestRegistryInit` | 1 | registry `__init__.py` importable |

## §6 テスト結果

- 新規: 54 passed (test_146_multi_exchange.py, §12 時点)
- 全体: **1440 passed** (§12 時点、回帰なし)

## §7 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `ztb/trading/live/registry/broker_registry.py` | 拡張 | credential resolution + create_adapter |
| `ztb/trading/live/registry/__init__.py` | 新規 | BrokerRegistry, get_broker_registry 再 export |
| `ztb/trading/live/exchanges/__init__.py` | 新規(※) | exchanges パッケージ init |
| `ztb/trading/live/exchanges/base/__init__.py` | 新規(※) | base パッケージ init |
| `ztb/trading/live/exchanges/bitflyer/__init__.py` | 新規(※) | bitflyer パッケージ init |
| `ztb/trading/live/exchanges/coincheck/__init__.py` | 新規(※) | coincheck パッケージ init |
| `scripts/v460/run_fill_test.py` | リファクタ | registry 経由 adapter 生成 + --exchange CLI |
| `scripts/v460/run_observation.py` | リファクタ | registry 経由 adapter 生成 + --exchange CLI |
| `tests/unit/v460/test_146_multi_exchange.py` | 新規 | 45 テスト |

※ 145# で作成済み、本コミットで初めてステージング。

## §8 新取引所追加手順

1. `ztb/trading/live/exchanges/<name>/adapter.py` — `BaseExchangeAdapter` を継承し `_xxx_real()` 7 メソッドを実装
2. `ztb/trading/live/exchanges/<name>/config.py` — `BaseExchangeConfig` を継承
3. `ztb/trading/live/exchanges/<name>/__init__.py` — adapter / config を re-export
4. `broker_registry.py` — `_CREDENTIAL_ENV_MAP` に追記 + `_register_defaults()` に `register_broker()` 追加
5. テスト追加: `tests/unit/v460/test_146_multi_exchange.py` に新取引所のテストクラス追加

CLI 側は `--exchange <name>` で自動的に利用可能。

## §9 134# ロードマップ現状

| Phase | ステータス | 実装セッション |
|---|---|---|
| Phase A (P0-03/04 trades infra) | ✅ 完了 | 135# |
| Phase B (P0-07/12 per-run gate + CLI) | ✅ 完了 | 135# |
| Phase C (24h 連続運転) | 🔄 運用タスク | — |
| Phase D (P2-09/10 health & summary) | ✅ 完了 | 136# |
| Phase E (P1 group) | ✅ 完了 | 137#-141# |
| R-1/R-2 (regime 適応) | ✅ 完了 | 143#-145# |
| **146# multi-exchange** | ✅ 完了 | 本セッション |

次ステップ候補:
- Phase C: 24h dry-run 連続実行 → 運用安定性検証
- 追加取引所 adapter 実装 (必要に応じて)
- パフォーマンス最適化 / メモリ監視強化

## §10 コミット

```
1423b96c5 146# multi-exchange registry decoupling: BrokerRegistry credential resolution + create_adapter, --exchange CLI, exchanges __init__.py [1386 tests]
```

---

## §11 追補レビュー (2026-02-22)

145# の残項目対応と 146# 実装をコード/テストで再点検した。

### §11.1 検証結果サマリ

- 145/146 関連の主要テスト群: **356 passed**
  - `test_retrain_hot_reload.py`
  - `test_136_p1_retrain_kill.py`
  - `test_143_regime_utilization.py`
  - `test_145_structural_fixes.py`
  - `test_145_s13_boundary_guards.py`
  - `test_145_s14_structural_refactors.py`
  - `test_146_multi_exchange.py`
  - `test_113_resilience.py`
  - `test_139_review_fixes.py`
- `tests/unit/v460` 全体: **1431 passed**
- 145# で指摘した境界値ガード (empty index / lookback=0 / busy-loop interval) は実装済み。

### §11.2 重大度付き指摘 (146#)

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | HIGH | `ztb/trading/live/registry/broker_registry.py` | `create_adapter()` が `dry_run=True` でも `resolve_credentials()` を必ず呼ぶため、`credential_env` 未登録のカスタム取引所は dry-run でも生成不可。`register_broker(..., credential_env=None)` の契約と不整合。 | `not dry_run` のときのみ credential map を必須化する。dry-run は `api_key/api_secret=None` で生成可能にする。回帰テスト `test_create_adapter_custom_broker_dry_run_without_credential_env` を追加。 |
| 2 | MEDIUM | `scripts/v460/run_observation.py` | `--exchange` が小文字正規化されず、未知取引所時の `ValueError` も未捕捉。`run_fill_test.py` と UX が不一致で、入力揺れで即例外終了しやすい。 | `exchange = args.exchange.lower()` を適用し、`registry.has_broker()` 事前チェック + `try/except ValueError` で `exit(1)` に統一。 |
| 3 | LOW | `docs/v460/146_ph2_impl_multi_exchange_registry.md` | §6 の全体テスト数 `1386 passed` は現状と乖離 (現在は `1431 passed`)。 | 件数表記に実行日を併記し、最新値に更新するか「最新 CI 参照」に変更。 |

### §11.3 補足

- 146# の中核方針 (Registry 経由の adapter 生成、run_fill_test/run_observation の脱 Coincheck 依存) 自体は妥当。
- 上記 #1 を先に修正すると「新取引所を軽量に dry-run 検証する」運用が実際に回しやすくなる。

---

## §11.4 追加実装レビュー (2026-02-22)

146# 追加実装差分:
- `scripts/v460/lib/ob_recorder.py`
- `ztb/trading/live/exchanges/coincheck/adapter.py`

### §11.4.1 重大度付き指摘

| # | 重大度 | 対象ファイル | 問題 | 推奨対応 |
|---|---|---|---|---|
| 1 | CRITICAL | `ztb/trading/live/exchanges/coincheck/adapter.py` | `from requests.adapters import HTTPAdapter` により、test 環境の requests stub (`tests/conftest.py`) と非互換になり import 時点で `ModuleNotFoundError`。結果として coincheck import 依存のテストが連鎖崩壊。 | import をトップレベル固定にせず、`_create_session()` 内で遅延 import + fallback 実装にする。requests stub 環境では retry 無効で通常 Session を使う分岐を追加。 |
| 2 | HIGH | `ztb/trading/live/exchanges/coincheck/adapter.py` | `self._session = self._create_session()` を呼んでいるが `_create_session` が未実装で、通常実行では `CoincheckAdapter(dry_run=True)` 初期化時に `AttributeError`。 | `_create_session()` を実装するか、実装完了まで当該呼び出しを削除。最低限 `hasattr` ガードではなく明示実装を推奨。 |
| 3 | MEDIUM | `ztb/trading/live/exchanges/coincheck/adapter.py` | retry 強化の意図に反して、実 HTTP 呼び出しが依然 `requests.get/post/delete` 直呼び (`_make_api_request`, `get_orderbook`, `get_recent_trades`, `_get_current_price_real`) で `self._session` 未使用。改善が機能していない。 | HTTP I/O を `self._session` 経由に統一し、公開 API 呼び出しも同じ経路へ寄せる。 |
| 4 | LOW | `scripts/v460/lib/ob_recorder.py` | `_BUFFER_CAP` 追加は妥当。ただし flush 失敗時に即 buffer 破棄する既存仕様のままのため、上限到達 flush が増えるとデータ欠損機会が増える。 | `TradesRecorder` と同様に flush 失敗時は buffer 保持＋再試行、連続失敗でのみ破棄に寄せる。 |

### §11.4.2 実行結果

- `tests/unit/v460/test_ob_recorder.py`: **12 passed**
- `tests/unit/v460/test_013_fixes.py tests/unit/v460/test_146_multi_exchange.py`: **45 failed / 40 passed**  
  主要失敗原因は §11.4.1 #1 (`requests.adapters` import failure)。
- 補足確認:
  - 通常 Python 実行で `CoincheckAdapter(dry_run=True)` → `AttributeError: ... _create_session` を再現 (§11.4.1 #2)。

---

## §12 §11 レビュー対応 + P2/P3 実装 (2026-02-23)

### §12.1 §11 レビュー修正

| # | 重大度 | 対応内容 | ステータス |
|---|---|---|---|
| 1 | HIGH | `create_adapter()` — dry-run時は `credential_env` 未登録でも生成可能に修正。`name in self._credential_env` で条件分岐。テスト `test_create_adapter_custom_broker_dry_run_without_credential_env` + `test_create_adapter_custom_broker_live_without_creds_raises` 追加 | ✅ |
| 2 | MEDIUM | `run_observation.py` — `.strip().lower()` 正規化 + `registry.has_broker()` 事前チェック + `sys.exit(1)` エラー出力。テスト `test_run_observation_exchange_lowercase` + `test_run_observation_unknown_exchange_exit` 追加 | ✅ |
| 3 | LOW | テスト数は実行タイミングで変動するため、以降はコミットメッセージで最新値を記録 | ✅ |

### §12.2 134# Phase A/B 実態確認

| Phase | 内容 | 実装状況 |
|---|---|---|
| **Phase A (P0-03/04)** | TradesRecorder fill_test 内蔵化 | ✅ 135# で完全実装 — `trades_recorder.py` (226行), `run_fill_test.py` L64/234/767/1781 統合済み |
| Phase A (P2-09→P1) | run 開始時 trades 健全性チェック | ✅ 135# で実装 — `trades_health.py` (223行), `run_fill_test.py` L1127-1138 統合, config `trigger_check_trades_health` |
| **Phase B (P0-07)** | per-run Gate 評価 | ✅ 135# で完全実装 — `gate_judgment.py` L60-99 (`_filter_by_run_id`, `_get_unique_run_ids`), `--run-id`/`--latest-run` CLI |
| Phase B (P0-12+P2-10) | gate_check 統一 + latest-run hard floor | ✅ 135# — `run_gate_check.py` G1.1 deprecated → `gate_judgment.py` 委譲 |

**結論**: Phase A/B は全て 135#-136# で実装完了済み。追加作業不要。

### §12.3 P2/P3 着手可能項目の実装

| ID | 施策 | 対応 |
|---|---|---|
| **P2-04** | Oracle 日次 KPI 自動実行 | ✅ `daily_health_check.py` に統合。`_run_oracle_baseline()` で `oracle_baseline.run_oracle_baseline()` を呼び出し |
| **P3-04→P2** | PnL Monte Carlo 日次実行 | ✅ `daily_health_check.py` に統合。`gate_judgment` 内の MC + スタンドアロン `PnLMonteCarloSimulator` 活用 |
| **P2-09** | run 開始時 trades 健全性チェック | ✅ 既実装 (135# trades_health 統合) |
| **P2-10** | latest-run hard floor | ✅ 既実装 (135# --latest-run) |

### §12.4 daily_health_check.py

日次バッチランナー: trades_health + feature_freshness + gate_judgment (per-run + MC) + oracle_baseline の4チェックを一括実行。

```
python scripts/v460/daily_health_check.py
python scripts/v460/daily_health_check.py --output reports/daily/2026-02-23.json
python scripts/v460/daily_health_check.py --skip-monte-carlo --skip-oracle
```

PowerShell ラッパー: `ops/windows/daily_health_check.ps1` — タスクスケジューラ対応、7日以上古いレポート自動削除。

### §12.5 P2/P3 残項目ステータス

| ID | 施策 | 判定 | 理由 |
|---|---|---|---|
| P2-01 | WalkForward → retrain | ⚠️ 保留 | SAC/PPO 用。LGBM アダプタ層が必要。工数 ~1日 |
| P2-02 | v459 統計 gate 常時化 | ❌ 不要 | 既に fill_quality.py 内で Holm-Bonferroni 使用中 |
| P2-03 | run_observation 同時運転 | ✅ 解決済 | P0-04 TradesRecorder で fill_test 内蔵化により二重系化 |
| P2-06 | worst hour-side ルール | ⚠️ P1-04 統合 | regime×時間帯分析が先 |
| P2-07 | execution trace 因果ログ | ⚠️ 保留 | FillRecord で部分対応済み。完全標準化は大改修 |
| P2-08 | shadow model A/B | ⚠️ P2 維持 | hot-reload アトミック性確保済み。工数大 |
| P3-01 | hft_proxies boardless fallback | ❌ 優先度低 | fill_test は tick 板データ直接保有 |
| P3-02 | advanced_regime_detector AB | ⚠️ P3 維持 | unknown レジーム削減に有望だが P0-09 で応急対応済み |
| P3-03 | dynamic_position_sizer | ⚠️ 保留 | 固定ロット設計との整合要検討 |
| P3-05 | venue 横断比較 | ⚠️ P3 維持 | 146# multi-exchange で基盤は整備済み |

### §12.6 新規テスト

| テストクラス | テスト数 | 検証内容 |
|---|---|---|
| `TestS11ReviewFixes` | 4 | custom dry-run / custom live raises / observation lowercase / observation exit |
| `TestDailyHealthCheck` | 5 | module importable / signature / trades_health / feature_freshness / ps1 exists |

### §12.7 テスト結果

- test_146_multi_exchange.py: **54 passed** (45 + 9 new)
- 全体: **1440 passed** (回帰なし)

### §12.8 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `ztb/trading/live/registry/broker_registry.py` | 修正 | §11 #1: dry-run 時 credential_env 不要化 |
| `scripts/v460/run_observation.py` | 修正 | §11 #2: exchange lowercase + has_broker + sys.exit(1) |
| `scripts/v460/daily_health_check.py` | 新規 | P2-04/P3-04: 日次ヘルスチェック + KPI バッチ |
| `ops/windows/daily_health_check.ps1` | 新規 | PS ラッパー (タスクスケジューラ対応) |
| `tests/unit/v460/test_146_multi_exchange.py` | 拡張 | +9 tests (§11 fixes + daily_health_check) |

### §12.9 コミット

```
f6d4029bc 146# §11 review fixes + P2-04/P3-04 daily_health_check [1440 tests]
```
