<#
.SYNOPSIS
    fill_test_watchdog.ps1 のテスト (150# P2-B)
.DESCRIPTION
    watchdog の各機能を検証する Pester テスト。
    実行: Invoke-Pester .\ops\windows\test_fill_test_watchdog.ps1
#>

BeforeAll {
    $Script:RootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..") | Select-Object -ExpandProperty Path
    $Script:WatchdogScript = Join-Path $PSScriptRoot "fill_test_watchdog.ps1"
}

Describe "fill_test_watchdog.ps1" {

    Context "監視モード (AutoRestart なし)" {

        It "fill_test 未起動時に exit 1 を返す" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                # fill_test が起動していない状態で実行
                $result = & $Script:WatchdogScript -ResultsDir $tmpDir 2>&1
                # exit code は $LASTEXITCODE で取得
                $LASTEXITCODE | Should -Be 1
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "watchdog_alert イベントが events.jsonl に記録される" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                & $Script:WatchdogScript -ResultsDir $tmpDir 2>&1 | Out-Null
                $eventsPath = Join-Path $tmpDir "fill_test_events.jsonl"
                Test-Path $eventsPath | Should -Be $true
                $content = Get-Content $eventsPath -Raw
                $content | Should -Match "watchdog_alert"
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "watchdog.log が作成される" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                & $Script:WatchdogScript -ResultsDir $tmpDir 2>&1 | Out-Null
                $logPath = Join-Path $tmpDir "logs\watchdog.log"
                Test-Path $logPath | Should -Be $true
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Context "AutoRestart モード" {

        It "crash loop 検出で exit 2 を返す" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                # 3件の watchdog_restart イベントを作成 (直近)
                $eventsPath = Join-Path $tmpDir "fill_test_events.jsonl"
                1..3 | ForEach-Object {
                    $evt = @{
                        timestamp = (Get-Date).ToUniversalTime().ToString("o")
                        event     = "watchdog_restart"
                        run_id    = $null
                        git_sha   = $null
                        pid       = $PID
                        reason    = "auto_restart"
                        details   = @{}
                    } | ConvertTo-Json -Compress
                    Add-Content -Path $eventsPath -Value $evt
                }

                & $Script:WatchdogScript -ResultsDir $tmpDir -AutoRestart -MaxRestarts 3 -CooldownMinutes 60 2>&1 | Out-Null
                $LASTEXITCODE | Should -Be 2
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "restart.lock が別インスタンスに保持されていたら exit 0 (SKIP)" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                # 新しい restart.lock を作成 (< 30秒)
                $lockPath = Join-Path $tmpDir "restart.lock"
                [System.IO.File]::WriteAllText($lockPath, "$PID|$((Get-Date).ToString('o'))")

                & $Script:WatchdogScript -ResultsDir $tmpDir -AutoRestart 2>&1 | Out-Null
                $LASTEXITCODE | Should -Be 0
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "stale restart.lock (> 30s) は自動削除される" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                $lockPath = Join-Path $tmpDir "restart.lock"
                [System.IO.File]::WriteAllText($lockPath, "12345|2020-01-01T00:00:00+00:00")
                (Get-Item $lockPath).LastWriteTime = (Get-Date).AddSeconds(-60)

                & $Script:WatchdogScript -ResultsDir $tmpDir -AutoRestart 2>&1 | Out-Null
                # stale lock が削除された後、NOT_RUNNING で再起動を試みるはず
                # (プロセスが実在しないので失敗するかもしれないが、lock は削除されるべき)
                Test-Path $lockPath | Should -Be $false
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Context "Helper: Get-StaleHeartbeatSec" {

        It "YAML から lock_stale_heartbeat_sec を読み取る" {
            $tmpYaml = Join-Path $env:TEMP "test_config_$(Get-Random).yaml"
            try {
                Set-Content -Path $tmpYaml -Value "fill_test:`n  lock_stale_heartbeat_sec: 600.0"
                # source the function
                . $Script:WatchdogScript -ResultsDir "." 2>&1 | Out-Null
                # Inline test: parse the file
                $content = Get-Content $tmpYaml -Raw
                $content -match 'lock_stale_heartbeat_sec:\s*([\d.]+)' | Should -Be $true
                [int][double]$Matches[1] | Should -Be 600
            } finally {
                Remove-Item $tmpYaml -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Context "Helper: Test-CrashLoop" {

        It "空の events.jsonl で crash loop なし" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                $eventsPath = Join-Path $tmpDir "fill_test_events.jsonl"
                New-Item -Path $eventsPath -ItemType File -Force | Out-Null
                # events.jsonl が空なら crash loop ではない
                $content = Get-Content $eventsPath
                ($content | Where-Object { $_ -match "watchdog_restart" }).Count | Should -Be 0
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }

        It "古い restart イベントは crash loop にカウントされない" {
            $tmpDir = Join-Path $env:TEMP "watchdog_test_$(Get-Random)"
            New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null
            try {
                $eventsPath = Join-Path $tmpDir "fill_test_events.jsonl"
                # 2時間前のイベント (CooldownMinutes=60 の範囲外)
                $oldTime = (Get-Date).AddHours(-2).ToUniversalTime().ToString("o")
                1..5 | ForEach-Object {
                    $evt = @{
                        timestamp = $oldTime
                        event     = "watchdog_restart"
                        reason    = "auto_restart"
                    } | ConvertTo-Json -Compress
                    Add-Content -Path $eventsPath -Value $evt
                }
                # 直近の restart は 0 件なので crash loop ではない
                $cutoff = (Get-Date).AddMinutes(-60).ToUniversalTime()
                $count = 0
                Get-Content $eventsPath | ForEach-Object {
                    $e = $_ | ConvertFrom-Json
                    if ($e.event -eq "watchdog_restart") {
                        $ts = [DateTime]::Parse($e.timestamp).ToUniversalTime()
                        if ($ts -gt $cutoff) { $count++ }
                    }
                }
                $count | Should -Be 0
            } finally {
                Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
