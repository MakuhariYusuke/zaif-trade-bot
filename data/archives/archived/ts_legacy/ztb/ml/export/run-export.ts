// RLデータエクスポートランナー
// RL学習用のデータをエクスポートするためのエントリーポイント

import { RLDataExporter, RLExportUtils } from '../export/rl';
import { traceLog } from '../../utils/logging/trace';

async function main() {
  try {
    traceLog('rl.export.start', { message: 'Starting RL data export' });

    // RLエクスポーターを作成
    const exporter = new RLDataExporter();

    // エクスポート実行
    const result = await exporter.exportToRL();

    if (result.success) {
      traceLog('rl.export.success', {
        total_files: result.totalFiles,
        total_steps: result.totalSteps,
        exported_files_count: result.exportedFiles.length,
        duration_ms: result.durationMs
      });

      console.log(`✅ RL data export completed successfully!`);
      console.log(`📁 Processed files: ${result.totalFiles}`);
      console.log(`🎯 Total RL steps: ${result.totalSteps}`);
      console.log(`💾 Exported files: ${result.exportedFiles.length}`);
      console.log(`⏱️ Duration: ${result.durationMs}ms`);

      // エクスポートされたファイル一覧を表示
      if (result.exportedFiles.length > 0) {
        console.log(`\n📋 Exported files:`);
        result.exportedFiles.forEach(file => console.log(`   - ${file}`));
      }

      // エクスポート統計を表示
      const stats = await exporter.getExportStats();
      console.log(`\n📊 Export statistics:`);
      console.log(`   Total files: ${stats.total_files}`);
      console.log(`   Total steps: ${stats.total_steps}`);
      console.log(`   Total size: ${stats.files.reduce((sum, f) => sum + f.size_mb, 0).toFixed(2)} MB`);

    } else {
      const errorMsg = result.error || 'Unknown error';
      traceLog('rl.export.failed', {
        error: errorMsg,
        duration_ms: result.durationMs
      });

      console.error(`❌ RL data export failed: ${errorMsg}`);
      throw new Error(errorMsg);
    }

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    traceLog('rl.export.error', { error: errorMsg });

    console.error(`💥 Unexpected error during RL data export:`, errorMsg);
    // 非同期ログやファイル書き込みの完了を待つ
    setTimeout(() => process.exit(1), 100); // 100ms待ってから終了
  }
}

// スクリプト実行
if (require.main === module) {
  main().catch(async error => {
    console.error('💥 Fatal error:', error);
    setTimeout(() => process.exit(1), 100);
  });
}
