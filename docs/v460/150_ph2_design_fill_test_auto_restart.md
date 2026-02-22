# 150# fill_test 自動再起動設計 (P2-B)

**日時**: 2026-02-23  
**種別**: design (設計)  
**Phase**: ph2 (maker 執行可能性検証)  
**前提**: 147# P2-B, 148# §4 #1/#2, 149# §8

---

## §1 背景と目的

### 1.1 問題

Phase C 24h run 中に fill_test が異常停止した (147#)。停止原因は特定できず (148# 確度: 中高)、以下の事実が判明:

- main() の未捕捉例外でプロセスが落ちる経路が存在した → **148# P0 で修正済**
- 停止しても通知・検出手段がなかった → **P2-A watchdog で対応済**
- **停止後の自動復旧が存在しない** → **本設計の対象**

### 1.2 目的

fill_test プロセスが異常終了した際に、安全に自動再起動する仕組みを設計する。

### 1.3 制約

| 制約 | 理由 |
|------|------|
| Windows 環境限定 | 運用サーバが Windows |
| 既存 lock 機構との整合 | `fill_test.lock` による排他制御を尊重 |
| warm_start 非破壊 | 再起動後も既存データを resume |
| 二重起動禁止 | lock + heartbeat で防御済 (148# P0) |
| 連続再起動抑制 | crash loop 防止が必要 |

---

## §2 設計選択肢

### 2.1 案 A: watchdog 拡張 (推奨)

既存の `fill_test_watchdog.ps1` に `-AutoRestart` オプションを追加。

```
[タスクスケジューラ] → 5分間隔 → [watchdog.ps1 -AutoRestart]
                                    ├─ RUNNING → OK → exit 0
                                    └─ NOT_RUNNING → restart → 新プロセス起動
```

**メリット**:
- 既存コードの再利用 (DRY)
- watchdog のログ・通知基盤をそのまま活用
- crash loop 制御をスクリプト内で完結

**デメリット**:
- 5分間隔のため最大 5分の空白期間
- タスクスケジューラ依存

### 2.2 案 B: ラッパースクリプト (永続ループ)

`trading_service.bat` と同様のパターン。

```
[タスクスケジューラ] → 起動時 → [fill_test_service.ps1]
                                    └─ while(true) {
                                         python run_fill_test.py --hours 168
                                         if (crash) sleep 60; continue
                                       }
```

**メリット**:
- 空白期間が短い (~60s)
- 単一プロセス監視のため直感的

**デメリット**:
- ラッパー自身が死ぬと復旧不可
- 既存 watchdog と機能重複
- `--hours` 残り時間の管理が複雑

### 2.3 案 C: Windows サービス化 (NSSM)

NSSM (Non-Sucking Service Manager) で fill_test を Windows サービスとして登録。

**メリット**:
- OS レベルの再起動保証
- サービス管理 UI で可視化

**デメリット**:
- 外部ツール依存
- `--hours` ベースの有限 run と相性が悪い
- venv の Python パスとの整合が煩雑

### 2.4 選定

| 観点 | 案 A | 案 B | 案 C |
|------|------|------|------|
| 実装コスト | ◎ 低 | ○ 中 | △ 高 |
| 信頼性 | ○ | △ | ◎ |
| 既存整合 | ◎ | ○ | △ |
| 空白期間 | △ ~5分 | ◎ ~60s | ◎ ~30s |
| crash loop 制御 | ◎ | ○ | ○ |

**結論: 案 A (watchdog 拡張) を推奨**。理由:

1. 既存の watchdog 基盤を活用でき DRY
2. crash loop 制御が明示的に書ける
3. 148# P0 の lock + heartbeat + events.jsonl との整合が自然
4. 本番稼働前の検証期間では十分。信頼性要件が高まれば案 C へ移行可

---

## §3 案 A 詳細設計

### 3.1 watchdog 拡張仕様

```
fill_test_watchdog.ps1 -AutoRestart [-MaxRestarts 3] [-CooldownMinutes 10]
```

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `-AutoRestart` | `$false` | 有効時、NOT_RUNNING で自動再起動 |
| `-MaxRestarts` | `3` | `CooldownMinutes` 内の最大再起動回数 |
| `-CooldownMinutes` | `60` | crash loop 判定ウィンドウ (分) |
| `-Hours` | `168` | 再起動時の `--hours` パラメータ |
| `-Config` | `configs/v460/fill_test.yaml` | 再起動時の `--config` パラメータ |

### 3.2 再起動判定フロー

```
1. プロセス存在確認
   ├─ RUNNING → ログ出力 → exit 0
   └─ NOT_RUNNING → 2 へ

2. crash loop 判定
   - fill_test_events.jsonl の直近 CooldownMinutes 内の
     "watchdog_restart" イベント数をカウント
   ├─ >= MaxRestarts → アラート "crash loop detected" → exit 2
   └─ < MaxRestarts → 3 へ

3. lock ファイル確認
   ├─ lock あり & PID alive → 矛盾 (WMI 検出漏れ?) → exit 1
   ├─ lock あり & PID dead → stale lock → lock 削除 → 4 へ
   └─ lock なし → 4 へ

4. 再起動実行
   - fill_test_events.jsonl に "watchdog_restart" イベント記録
   - Start-Process で新プロセスをバックグラウンド起動
   - Discord 通知
   - exit 0
```

### 3.3 再起動コマンド

```powershell
$pythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
$arguments = @(
    "-m", "scripts.v460.run_fill_test",
    "--hours", $Hours,
    "--config", $Config
)
Start-Process -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $RootDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $ResultsDir "logs\fill_test_stdout.log") `
    -RedirectStandardError (Join-Path $ResultsDir "logs\fill_test_stderr.log")
```

### 3.4 crash loop 防止

`fill_test_events.jsonl` 内の `watchdog_restart` イベントのタイムスタンプを走査し、直近 `CooldownMinutes` 分以内に `MaxRestarts` 回以上の再起動が発生していた場合は再起動を抑止。

```
CooldownMinutes=60, MaxRestarts=3 の場合:
  04:00 restart → count=1 → OK
  04:15 restart → count=2 → OK
  04:25 restart → count=3 → BLOCK (crash loop detected)
  05:01 restart → count=2 (04:00 は 60分超で失効) → OK
```

### 3.5 warm_start との整合

`run_fill_test.py` は起動時に `resume_from_existing()` で既存 fill_records を読み込むため、再起動後も累積データは自動復元される。`--hours` は新プロセスの起動時点からのカウントになるため、元の残り時間とは独立する。

**注意**: 連続稼働時間の評価は `fill_test_events.jsonl` の start/stop/restart イベント列から計算する必要がある。

---

## §4 タスクスケジューラ統合

### 4.1 登録コマンド

```powershell
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File C:\Users\Admin\dev\zaif-trade-bot\ops\windows\fill_test_watchdog.ps1 -AutoRestart -MaxRestarts 3 -CooldownMinutes 60 -Hours 168"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5)
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew
Register-ScheduledTask `
    -TaskName "ZTB-FillTestWatchdog" `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "150# P2-B: fill_test watchdog + auto-restart"
```

### 4.2 タスク一覧

| タスク名 | 間隔 | スクリプト | 目的 |
|---------|------|-----------|------|
| ZTB-FillTestWatchdog | 5分 | `fill_test_watchdog.ps1 -AutoRestart` | 死活監視 + 自動再起動 |
| ZTB-DailyHealthCheck | 日次 09:00 | `daily_health_check.ps1` | ヘルスチェック + KPI |

---

## §5 テスト計画

| # | テストケース | 期待結果 | 手順 |
|---|-------------|---------|------|
| T1 | プロセス稼働中に watchdog 実行 | exit 0、再起動なし | watchdog 手動実行 |
| T2 | プロセス停止後に watchdog 実行 | 再起動実行、events.jsonl に restart 記録 | fill_test 停止 → watchdog |
| T3 | crash loop 検出 | 再起動抑止、アラート送出 | 4回連続停止 → watchdog |
| T4 | stale lock 残存で停止 | lock 削除 → 再起動 | kill -9 で lock 残し → watchdog |
| T5 | 再起動後の warm_start | 既存 fill_records を resume | T2 後にログ確認 |

---

## §6 リスクと緩和策

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| watchdog 自体の障害 | 再起動不可 | タスクスケジューラのエラー通知 + 手動復旧手順を明文化 |
| 再起動時の未決済ポジション | 損失拡大 | `_cancel_stale_orders()` が起動時に実行される (既存設計) |
| crash loop false negative | 復旧不可 | `MaxRestarts` を保守的に設定 (3回/60分) |
| 5分の復旧空白期間 | 取引機会の逸失 | 許容範囲 (fill_test は ~20s/cycle のため ~15 cycle 分) |
| `--hours` と実際の連続稼働乖離 | 評価誤り | events.jsonl で実効稼働時間を計算する運用 |

---

## §7 将来拡張

| 項目 | 条件 | 内容 |
|------|------|------|
| NSSM サービス化 (案 C) | 本番稼働で 99.9%+ 可用性が必要な場合 | watchdog → NSSM 移行 |
| Prometheus + Grafana | 監視基盤の本格化 | heartbeat → Prometheus exporter |
| Auto-scaling lot | 復旧直後のリスク軽減 | 再起動後 N サイクルは lot を縮小 |

---

## §8 Codex レビュー依頼事項

### 8.1 設計レビュー

- 案 A (watchdog 拡張) の選定は妥当か。信頼性観点で案 C を先行すべきか
- crash loop 防止ロジック (`MaxRestarts=3, CooldownMinutes=60`) の閾値は適切か
- warm_start との整合で見落としている状態遷移はないか

### 8.2 安全性レビュー

- 再起動時の未決済ポジション処理 (`_cancel_stale_orders`) の網羅性
- lock stale → 再起動の競合可能性 (watchdog と fill_test 本体の間の TOCTOU)
- `Start-Process` でのバックグラウンド起動時の stderr/stdout 保全

### 8.3 コードベース確認依頼

```
ops/windows/fill_test_watchdog.ps1:
  - 現在の実装。§3 の拡張追加先

scripts/v460/run_fill_test.py:
  - _acquire_lock() / _release_lock(): lock の排他ロジック
  - _cancel_stale_orders(): 起動時の滞留注文クリア
  - resume_from_existing(): warm_start の復元ロジック

scripts/v460/lib/fill_config.py:
  - lock_heartbeat_period_sec / lock_stale_heartbeat_sec: heartbeat 設定

ops/windows/trading_service.bat:
  - 既存の永続ループパターン (案 B 参考)
```

---

## 変更ファイル

| ファイル | 変更内容 |
|----------|----------|
| `docs/v460/150_ph2_design_fill_test_auto_restart.md` | NEW: 本ドキュメント |
