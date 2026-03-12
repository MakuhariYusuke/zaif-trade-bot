# v456 訓練進捗モニタリングスクリプト

param(
    [int]$RefreshInterval = 10,  # 秒単位でリフレッシュ
    [switch]$Continuous = $false
)

$LogFile = "training_log.txt"
$LastLineCount = 0

function Get-TrainingMetrics {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        return @{
            IsRunning = $false
            CurrentLine = "ログファイルなし"
            LastMilestone = "N/A"
            Status = "未開始"
        }
    }
    
    $content = Get-Content $FilePath -ErrorAction SilentlyContinue
    if (-not $content) {
        return @{
            IsRunning = $false
            CurrentLine = ""
            LastMilestone = "N/A"
            Status = "ログなし"
        }
    }
    
    $lines = @($content) | ForEach-Object { $_ }
    $lastLine = $lines[-1]
    
    # Milestone の検出
    $milestones = @($lines | Select-String "Milestone|Completed|Error" -ErrorAction SilentlyContinue)
    $lastMilestone = if ($milestones.Count -gt 0) { $milestones[-1].Line } else { "N/A" }
    
    # 完了判定
    $isCompleted = $lastLine -match "✅|Completed|完了"
    $isError = $lastLine -match "Error|エラー|Failed"
    
    $status = if ($isError) { "❌ エラー" } 
             elseif ($isCompleted) { "✅ 完了" }
             else { "🔄 実行中" }
    
    return @{
        IsRunning = -not $isCompleted -and -not $isError
        LineCount = $lines.Count
        CurrentLine = $lastLine
        LastMilestone = $lastMilestone
        Status = $status
        IsCompleted = $isCompleted
        IsError = $isError
    }
}

# メイン実行
Write-Host "v456 訓練進捗モニタリング開始" -ForegroundColor Cyan
Write-Host "ログファイル: $LogFile" -ForegroundColor Gray
Write-Host ""

if ($Continuous) {
    Write-Host "連続モニタリング (Ctrl+C で停止)" -ForegroundColor Yellow
    while ($true) {
        Clear-Host
        Write-Host "v456 訓練進捗モニタリング" -ForegroundColor Cyan
        Write-Host "最終更新: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
        Write-Host ""
        
        $metrics = Get-TrainingMetrics $LogFile
        
        Write-Host "ステータス: $($metrics.Status)" -ForegroundColor $(
            if ($metrics.IsError) { "Red" }
            elseif ($metrics.IsCompleted) { "Green" }
            else { "Yellow" }
        )
        
        Write-Host "ログ行数: $($metrics.LineCount)"
        Write-Host ""
        
        if ($metrics.LastMilestone -ne "N/A") {
            Write-Host "最後の Milestone:" -ForegroundColor Cyan
            Write-Host $metrics.LastMilestone
            Write-Host ""
        }
        
        Write-Host "最後の行:" -ForegroundColor Cyan
        Write-Host $metrics.CurrentLine -ForegroundColor Gray
        Write-Host ""
        
        if ($metrics.IsCompleted) {
            Write-Host "✅ 訓練完了しました！" -ForegroundColor Green
            break
        }
        elseif ($metrics.IsError) {
            Write-Host "❌ エラーが発生しました" -ForegroundColor Red
            break
        }
        
        Write-Host "次の更新: $RefreshInterval 秒後 (リフレッシュ中...)" -ForegroundColor Gray
        Start-Sleep -Seconds $RefreshInterval
    }
} else {
    # 単発実行
    $metrics = Get-TrainingMetrics $LogFile
    
    Write-Host "ステータス: $($metrics.Status)" -ForegroundColor $(
        if ($metrics.IsError) { "Red" }
        elseif ($metrics.IsCompleted) { "Green" }
        else { "Yellow" }
    )
    Write-Host "ログ行数: $($metrics.LineCount)"
    Write-Host ""
    
    if ($metrics.LastMilestone -ne "N/A") {
        Write-Host "最後の Milestone:" -ForegroundColor Cyan
        Write-Host $metrics.LastMilestone
        Write-Host ""
    }
    
    Write-Host "最後の行:" -ForegroundColor Cyan
    Write-Host $metrics.CurrentLine
}
