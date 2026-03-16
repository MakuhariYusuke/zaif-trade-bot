# 354# コード品質改善: 型安全性・DRY・サイレント例外・drift修正

> **種別**: fix (phg)  
> **フェーズ**: phg (フェーズ横断品質改善)  
> **前提**: 353# 実装後のコードレビュー, mypy エラー 12 件  
> **日付**: 2026-03-09  
> **コミット**: `819ec73b2`

---

## §1 概要

353# VPIN buy boost 実装後にテスト全件実行 → 2 件失敗 (test_183, test_236)。
加えて mypy / コードレビューで 6 カテゴリの品質問題を特定・修正。

---

## §2 修正一覧

### ERR-1: サイレント例外 (risk_manager.py)

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/risk_manager.py` |
| 問題 | `except Exception: pass` — drawdown 例外を完全に握りつぶし |
| リスク | DD 制御不動作に気付けない → 資金保護崩壊 |
| 修正 | `logger.warning("drawdown_controller ...", exc_info=True)` |

### TYPE-1: sell_dynamic_kill.py 型安全化 (mypy 9→0)

| 項目 | 内容 |
|---|---|
| ファイル | `ztb/risk/sell_dynamic_kill.py` |
| 問題 | `import_state()` で dict[str, Any] の値を直接 int/float として使用 |
| 修正 | `_get_int()` / `_get_float()` ヘルパー追加 (isinstance 型ガード) |

### TYPE-2: config_hot_reload.py 戻り値型 (mypy 2→0)

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/config_hot_reload.py` |
| 問題 | `_resolve_time_filter_cls()` の戻り値型が `type[object]` |
| 修正 | `type[TimeFilter]` に変更 + `# type: ignore[no-any-return]` |

### TYPE-3: maker_risk_guards.py Protocol stub (mypy 1→0)

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/maker_risk_guards.py` |
| 問題 | Protocol stub の空ボディに mypy empty-body エラー |
| 修正 | `# type: ignore[empty-body]` 付与 |

### DRY-1: fill_config_parser.py パーサー共通化 (~60行→22行)

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/fill_config_parser.py` |
| 問題 | sell/buy の dynamic_kill + inv_relaxation パーサーが完全重複 |
| 修正 | `_parse_dynamic_kill_block()` / `_parse_inv_relaxation_block()` 抽出 |

### DRIFT-2: config_hot_reload.py ホットリロード登録漏れ

| 項目 | 内容 |
|---|---|
| ファイル | `scripts/v460/lib/config_hot_reload.py` |
| 問題 | `sell_dynamic_kill_inv_relaxation_enabled/scale` が `_HOT_RELOADABLE_FIELDS` に未登録 |
| 修正 | 2 フィールドを whitelist に追加 |

---

## §3 テスト修正

| テスト | 問題 | 修正 |
|---|---|---|
| test_183 | 353# で `buy_velocity_skip_threshold_bps` を -6→-4 に変更、assertion 未追従 | 期待値を -4.0 に更新 |
| test_236 | `_build_state_snapshot` が `orchestrator_lifecycle.py` に移動済みだが参照パスが旧モジュール | `OrchestratorLifecycleMixin` に修正 |

---

## §4 横展開確認

- test_254 / test_226: `inspect.getsource(FillLoopOrchestratorMixin.METHOD)` — MRO 経由で正しく解決されるため修正不要
- mypy: 3 ファイル計 12 エラー → 全 0

---

## §5 検証

- v460 全 4194 テスト pass
- Bot 再起動 (PID 変更確認, Cycle 8919~, git_sha=819ec73b2081)
