// ML特徴量収集ランナー
// ETLパイプラインを実行するためのエントリーポイント

import { ETLPipeline, ETLPipelineUtils } from '../etl/pipeline';
import { traceLog } from '../../utils/logging/trace';

async function main() {
  try {
    traceLog('ml.collect.start', { message: 'Starting ML feature collection' });

    // 設定ファイルから設定を読み込み（オプション）
    const configPath = process.env.ETL_CONFIG_PATH;
    const config = configPath
      ? ETLPipelineUtils.loadConfigFromFile(configPath)
      : undefined;

    // ETLパイプラインを作成
    const pipeline = new ETLPipeline(config);

    // 単発実行
    const result = await pipeline.runOnce();

    if (result.success) {
      traceLog('ml.collect.success', {
        extracted_count: result.extractedCount,
        transformed_count: result.transformedCount,
        saved_count: result.savedCount,
        duration_ms: result.durationMs
      });

      console.log(`✅ ML feature collection completed successfully!`);
      console.log(`📊 Extracted: ${result.extractedCount} records`);
      console.log(`🔄 Transformed: ${result.transformedCount} features`);
      console.log(`💾 Saved: ${result.savedCount} features`);
      console.log(`⏱️ Duration: ${result.durationMs}ms`);
    } else {
      traceLog('ml.collect.failed', {
        errors: result.errors,
        duration_ms: result.durationMs
      });

      console.error(`❌ ML feature collection failed:`);
      result.errors.forEach(error => console.error(`   - ${error}`));
      process.exit(1);
    }

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    traceLog('ml.collect.error', { error: errorMsg });

    console.error(`💥 Unexpected error during ML feature collection:`, errorMsg);
    process.exit(1);
  }
}

// スクリプト実行
if (require.main === module) {
  main().catch(error => {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  });
}
