# 180# — Watchdog 非表示化 + 179# Self-Review + from_yaml 堅牢化

> **分類**: ph2_impl  
> **前提**: 179# RegimePolicyConfig + CycleStrategy + Chase  
> **コミット**: 本作業

---

## §1 Watchdog cmd ポップアップ修正

### 問題

Windows Task Scheduler の `ZTB-Watchdog` タスクが 5 分間隔で
`powershell.exe` を直接起動し、コンソールウィンドウが一瞬表示されていた。

### 原因

`register_tasks.ps1` の `New-ScheduledTaskAction` が `-Execute "powershell.exe"` を
使用しており、PowerShell.exe は起動時に必ずコンソールウィンドウを生成する。
`-WindowStyle Hidden` だけでは起動直後の 1 フレームが表示される (Windows の既知動作)。

### 修正

VBScript ラッパー `ops/windows/run_hidden.vbs` を新規作成:
- `WScript.Shell.Run cmd, 0, True` の第 2 引数 `0` (SW_HIDE) で完全非表示
- `wscript.exe` は GUI プロセスのため、コンソールウィンドウを一切生成しない

`register_tasks.ps1` の Action を変更:
```
Before: powershell.exe -NoProfile -ExecutionPolicy Bypass -File "watchdog.ps1"
After:  wscript.exe "run_hidden.vbs" "watchdog.ps1" -Notify -AutoRestart
```

### 変更ファイル

| ファイル | 種別 |
|----------|------|
| `ops/windows/run_hidden.vbs` | 新規 |
| `ops/windows/register_tasks.ps1` | 修正 |

---

## §2 179# Self-Review

### レビュー結果サマリー

| 項目 | 評価 | 詳細 |
|------|------|------|
| 構造設計 | ✅ | SRP / Protocol 分離が適切 |
| 型安全 | ✅ | Any 不使用、型注釈完備 |
| エッジケース | ⚠️ | `from_yaml` のエラーハンドリング不足 → §3 で修正 |
| `_check_fallback()` 性能 | ✅ | time.time() 1 回 + bool 1 回 = 問題なし |
| `hasattr` ガード | ✅ | Mixin パターンでの正当な使用 |
| テスト網羅性 | ⚠️ | 不正入力テストなし → §3 で追加 |
| hot-reload | ✅ | try/except で失敗時は旧 strategy 維持 |

### 発見事項

1. **`from_yaml` の型変換エラー未処理**: `float()` / `int()` 変換で不正値が
   入った場合に `ValueError` / `TypeError` が伝播し、YAML reload がクラッシュする。
2. **`kwargs: dict` 型注釈不足**: `dict[str, object]` が適切。
3. **`yaml_cfg: dict` 引数型注釈不足**: `dict[str, object]` が適切。

---

## §3 from_yaml 堅牢化

### 修正内容

- 各セクション (dynamic_cycle/dynamic_wait/chase/stop_conditions) の
  型変換を `try/except (TypeError, ValueError)` で囲む
- 不正値はキー単位で warning ログ出力し、デフォルト値にフォールバック
- `regime_policy` が dict でない場合の早期リターン追加
- `kwargs` / `yaml_cfg` の型注釈を `dict[str, object]` に修正

### 追加テスト (7 件)

| テスト | 内容 |
|--------|------|
| `test_regime_policy_not_dict` | regime_policy が文字列 |
| `test_intervals_with_non_numeric` | intervals に変換不能値 |
| `test_waits_with_malformed_sides` | waits の side 値が文字列 |
| `test_chase_drift_bps_non_numeric` | chase.drift_bps が文字列 |
| `test_stop_conditions_non_numeric` | stop_conditions の値が文字列 |
| `test_none_regime_policy` | regime_policy が None |
| `test_empty_dynamic_cycle` | dynamic_cycle が空 dict |

テスト結果: **72/72 PASSED** (65 既存 + 7 新規)

---

## §4 178# 残課題棚卸し

179# で Phase 1–3 (S1/S2/S3 + C/D + Chase) は完了。残る項目:

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| Phase 4 | 条件付き IOC | ⏳ 未着手 | Coincheck API IOC 対応要確認 |
| Phase 5 | EV_weighted 評価窓 | ⏳ 未着手 | pnl120 パイプライン確認後 |
| §1.2 | Trend Mode 発動条件厳格化 | ⏳ 未着手 | confidence + velocity + spread AND |
| §1.5 | CircuitBreaker 統合 | ⏳ 未着手 | API 障害耐性の前提条件 |
| §2.4 | 在庫偏り regime 別緩和 | △ 要検討 | balance_forced_deadlock_limit の regime 分岐 |
| §6 | Mixin → 独立クラス化 | ⏳ 長期 | 破壊的変更、fill_test 稼働中は危険 |

### 直近の優先度

1. **まず C/D/Chase の有効化検証** — `enabled: false` のままなので、
   設定変更 → hot-reload で効果を確認
2. **EV_weighted (Phase 5)** — pnl120 パイプラインの信頼性確認
3. **CircuitBreaker** — C で cycle 短縮すると API 負荷が上がるため必要

