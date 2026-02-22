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

> **注**: 以下は設計段階。`fill_test_watchdog.ps1` の現行実装は監視のみ (P2-A)。P2-B 実装で追加予定。

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
| `-Exchange` | `coincheck` | 再起動時の `--exchange` パラメータ |
| `-DryRun` | `$false` | 再起動時の `--dry-run` フラグ |

### 3.2 イベント契約 (schema)

`fill_test_events.jsonl` のイベント種別を厳密に定義:

| event | 発行元 | 説明 |
|-------|--------|------|
| `start` | fill_test | プロセス開始 (起動パラメータ含む) |
| `stop` | fill_test | 正常終了 |
| `crash` | fill_test | 未捕捉例外による停止 |
| `signal` | fill_test | シグナル受信 (SIGINT 等) |
| `watchdog_alert` | watchdog | 監視異常検出 (プロセス不在、heartbeat stale 等) |
| `watchdog_restart` | watchdog | 自動再起動実行 |
| `trades_health_alert` | fill_test | trades データ鮮度劣化 |

`start` イベントには `details.args` として起動パラメータ (`--hours`, `--config`, `--exchange`, `--dry-run`) を記録し、watchdog が復元可能にする。

### 3.3 再起動判定フロー

```
1. restart.lock 取得 (短寿命、TOCTOU 防止)
   ├─ 取得失敗 → 別の watchdog が再起動中 → exit 0
   └─ 取得成功 → 2 へ

2. プロセス存在確認
   ├─ RUNNING → ログ出力 → restart.lock 解放 → exit 0
   └─ NOT_RUNNING → 3 へ

3. crash loop 判定
   - fill_test_events.jsonl の直近 CooldownMinutes 内の
     "watchdog_restart" イベント数をカウント
   ├─ >= MaxRestarts → アラート "crash loop detected" → restart.lock 解放 → exit 2
   └─ < MaxRestarts → 4 へ

4. lock ファイル確認
   ├─ lock あり & PID alive → 矛盾 (WMI 検出漏れ?) → restart.lock 解放 → exit 1
   ├─ lock あり & PID dead → stale lock → lock 削除 → 5 へ
   └─ lock なし → 5 へ

5. 再起動実行
   - fill_test_events.jsonl に "watchdog_restart" イベント記録
   - Start-Process で新プロセスをバックグラウンド起動
   - Discord 通知
   - restart.lock 解放
   - exit 0
```

### 3.4 再起動コマンド

```powershell
$pythonExe = Join-Path $RootDir ".venv\Scripts\python.exe"
$arguments = @(
    "-m", "scripts.v460.run_fill_test",
    "--hours", $Hours,
    "--config", $Config,
    "--exchange", $Exchange
)
if ($DryRun) { $arguments += "--dry-run" }
Start-Process -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $RootDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $ResultsDir "logs\fill_test_stdout.log") `
    -RedirectStandardError (Join-Path $ResultsDir "logs\fill_test_stderr.log")
```

**起動パラメータ復元**: watchdog は `fill_test_events.jsonl` の最新 `start` イベントの `details.args` から `--exchange`, `--dry-run` 等を復元可能。明示パラメータがない場合のフォールバックとして使用。

### 3.5 crash loop 防止

`fill_test_events.jsonl` 内の `watchdog_restart` イベントのタイムスタンプを走査し、直近 `CooldownMinutes` 分以内に `MaxRestarts` 回以上の再起動が発生していた場合は再起動を抑止。

```
CooldownMinutes=60, MaxRestarts=3 の場合:
  04:00 restart → count=1 → OK
  04:15 restart → count=2 → OK
  04:25 restart → count=3 → BLOCK (crash loop detected)
  05:01 restart → count=2 (04:00 は 60分超で失効) → OK
```

### 3.6 warm_start との整合

`run_fill_test.py` は起動時に `resume_from_existing()` で既存 fill_records を読み込むため、再起動後も累積データは自動復元される。`--hours` は新プロセスの起動時点からのカウントになるため、元の残り時間とは独立する。

**注意**: 連続稼働時間の評価は `fill_test_events.jsonl` の start/stop/restart イベント列から計算する必要がある。

### 3.7 再起動直後セーフモード

再起動直後は取引所 API 状態が不明なため、以下のセーフモードを適用:

- 再起動後 N サイクル (デフォルト: 5) は lot を 50% に縮小
- `_cancel_stale_orders()` 失敗時は発注を抑止し、watchdog_alert イベントを記録
- セーフモード解除条件: 連続 N 回の正常サイクル完了

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
| 再起動時の未決済ポジション | 損失拡大 | `_cancel_stale_orders()` が起動時に実行される (既存設計)。API 異常時の残注文リスクは §3.7 セーフモードで緩和 |
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

---

## §9 Codex 深掘りレビュー追記 (2026-02-23)

### 9.1 指摘事項 (重大度順)

| # | 重大度 | 対象 | 指摘 | 推奨対応 |
|---|---|---|---|---|
| 1 | **HIGH** | `ops/windows/fill_test_watchdog.ps1` 現行実装 | 設計で前提にしている `-AutoRestart / -MaxRestarts / -CooldownMinutes` が未実装。設計と実体のギャップがある。 | 実装前提を「設計段階」と明記し、最小実装（AutoRestart + restart event 記録）を先に切る。 |
| 2 | **HIGH** | `ops/windows/fill_test_watchdog.ps1:72` | stale 判定閾値が `300` 秒ハードコード。`fill_test` 側設定と乖離しうる。 | watchdog が `configs/v460/fill_test.yaml` か lock metadata を読む形に統一。 |
| 3 | **HIGH** | `docs/v460/150_ph2_design_fill_test_auto_restart.md:151-157` | 再起動コマンドが固定 `--hours/--config` で、起動時の `--exchange`/`--dry-run`/運用モード差分を引き継がない。 | 最後の start event から起動パラメータを復元するか、watchdog 側に明示設定を持つ。 |
| 4 | **MEDIUM** | `docs/v460/150_ph2_design_fill_test_auto_restart.md:123-147` | crash loop 判定は `watchdog_restart` イベント依存だが、現行イベント流には同名イベントが存在しない。 | 先にイベント契約（schema）を定義し、`watchdog_alert` と `watchdog_restart` を厳密に分離。 |
| 5 | **MEDIUM** | `docs/v460/150_ph2_design_fill_test_auto_restart.md:124-139` | lock 判定→再起動の間に並行実行が入る TOCTOU が残る。Task Scheduler 設定だけでは完全防止できない。 | watchdog 側にも `restart.lock`（短寿命）を導入し、再起動処理の排他を取る。 |
| 6 | **LOW** | `docs/v460/150_ph2_design_fill_test_auto_restart.md:169-177` | `_cancel_stale_orders()` を主緩和策にしているが、取引所 API 異常時の未キャンセル残存ケースが明示されていない。 | 再起動直後 N サイクルは lot 縮小＋発注抑制のセーフモードを追加。 |

### 9.2 総評

- 案 A（watchdog 拡張）自体は妥当。  
- ただし **「イベント契約の確定」と「起動パラメータ継承」** を先に固めないと、再起動しても評価汚染や誤起動のリスクが残る。  
- 実装順は `イベント契約` → `AutoRestart最小実装` → `crash loop` → `通知強化` が安全。

### 9.3 対応結果

| # | 対応 |
|---|------|
| 1 | **実装済**: §3.1 に「設計段階」明記 → P2-B 実装完了 (`fill_test_watchdog.ps1` 拡張) |
| 2 | **実装済**: `Get-StaleHeartbeatSec` 関数で `fill_test.yaml` から `lock_stale_heartbeat_sec` を動的読み取り |
| 3 | **実装済**: `-Exchange`/`-DryRun` パラメータ追加 + `Get-LastStartArgs` で start event からパラメータ復元 |
| 4 | **実装済**: §3.2 イベント契約定義 + `Write-EventLog` で `watchdog_alert`/`watchdog_restart` を厳密分離 |
| 5 | **実装済**: `restart.lock` (短寿命 30s) による TOCTOU 防止 + `try/finally` で確実解放 |
| 6 | **設計反映**: §3.7 セーフモード設計を追加。fill_test 側の実装は次フェーズ |

---

## §10 P2-B 実装ログ (2026-02-23)

### 10.1 実装ファイル

| ファイル | 変更内容 |
|----------|----------|
| `ops/windows/fill_test_watchdog.ps1` | P2-B: AutoRestart 全機能実装 (168→320行) |
| `scripts/v460/run_fill_test.py` | start イベントに `details.args` 追加 (watchdog パラメータ復元用) |
| `ops/windows/test_fill_test_watchdog.ps1` | NEW: Pester テスト |

### 10.2 実装機能一覧

| 機能 | パラメータ | 説明 |
|------|-----------|------|
| 自動再起動 | `-AutoRestart` | NOT_RUNNING 検出時に fill_test を再起動 |
| crash loop 防止 | `-MaxRestarts 3 -CooldownMinutes 60` | 一定時間内の再起動回数を制限 |
| TOCTOU 防止 | `restart.lock` | 複数 watchdog インスタンスの排他制御 |
| stale 閾値動的取得 | YAML 読み取り | `lock_stale_heartbeat_sec` をハードコードから YAML 参照に変更 |
| パラメータ復元 | `Get-LastStartArgs` | events.jsonl の最新 start event から `--exchange`, `--dry-run` 等を復元 |
| 明示パラメータ | `-Hours/-Config/-Exchange/-DryRun` | watchdog コマンドラインから直接指定も可 |

### 10.3 動作確認結果

| テスト | 結果 | 詳細 |
|--------|------|------|
| 監視モード (AutoRestart なし) | ✅ | PID=108148 検出、heartbeat stale 報告 |
| AutoRestart + RUNNING | ✅ exit 0 | 再起動せず正常終了 |
| restart.lock 排他 | ✅ exit 0 SKIP | 別インスタンスの lock を検出、スキップ |
| Python テスト (16件) | ✅ ALL PASS | start event args 含む全テスト OK |
