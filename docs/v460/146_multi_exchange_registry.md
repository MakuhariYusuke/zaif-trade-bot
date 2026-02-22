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

- 新規: 45 passed (test_146_multi_exchange.py)
- 全体: **1386 passed** (回帰なし)

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
