// ETL パイプライン オーケストレーター
// ML特徴量収集のメイン処理を管理

import { PriceExtractor } from './extract/price.js';
import { FeatureTransformer } from './transform/features.js';
import { ParquetLoader } from './load/parquet.js';
// import { logTrace } from '../../../utils/logger';

// トレースログ関数（フォールバック）
function traceLog(key: string, data: any): void {
  if (process.env.LOG_TRACE === '1') {
    console.log(`[TRACE:${key}]`, JSON.stringify(data, null, 2));
  }
}

export interface ETLConfig {
  batchIntervalMinutes: number;
  maxRetries: number;
  retryDelayMs: number;
  exchanges: string[];
  pairs: string[];
  lookbackHours: number;
}

export const DEFAULT_ETL_CONFIG: ETLConfig = {
  batchIntervalMinutes: 30,
  maxRetries: 3,
  retryDelayMs: 5000,
  exchanges: ['zaif'],
  pairs: ['btc_jpy', 'eth_jpy'],
  lookbackHours: 24
};

/**
 * ETLパイプライン実行結果
 */
export interface ETLResult {
  success: boolean;
  extractedCount: number;
  transformedCount: number;
  savedCount: number;
  durationMs: number;
  errors: string[];
}

/**
 * ETLパイプライン オーケストレーター
 */
export class ETLPipeline {
  private config: ETLConfig;
  private extractor: PriceExtractor;
  private transformer: FeatureTransformer;
  private loader: ParquetLoader;
  private isRunning: boolean = false;

  constructor(config: ETLConfig = DEFAULT_ETL_CONFIG) {
    this.config = config;
    this.extractor = new PriceExtractor();
    this.transformer = new FeatureTransformer();
    this.loader = new ParquetLoader();
  }

  /**
   * 単発のETL実行
   */
  async runOnce(): Promise<ETLResult> {
    const startTime = Date.now();
    const errors: string[] = [];

    try {
      traceLog('etl.pipeline.start', {
        exchanges: this.config.exchanges,
        pairs: this.config.pairs,
        lookback_hours: this.config.lookbackHours
      });

      // 1. データ抽出
      const rawData = await this.extractData();
      if (rawData.length === 0) {
        traceLog('etl.pipeline.no_data', { message: 'No data extracted' });
        return {
          success: true,
          extractedCount: 0,
          transformedCount: 0,
          savedCount: 0,
          durationMs: Date.now() - startTime,
          errors: []
        };
      }

      // 2. 特徴量変換
      const features = await this.transformData(rawData);

      // 3. データ保存
      await this.saveData(features);

      const result: ETLResult = {
        success: true,
        extractedCount: rawData.length,
        transformedCount: features.length,
        savedCount: features.length,
        durationMs: Date.now() - startTime,
        errors: []
      };

      traceLog('etl.pipeline.complete', {
        extracted_count: result.extractedCount,
        transformed_count: result.transformedCount,
        saved_count: result.savedCount,
        duration_ms: result.durationMs
      });

      return result;

    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      errors.push(errorMsg);

      traceLog('etl.pipeline.error', {
        error: errorMsg,
        duration_ms: Date.now() - startTime
      });

      return {
        success: false,
        extractedCount: 0,
        transformedCount: 0,
        savedCount: 0,
        durationMs: Date.now() - startTime,
        errors
      };
    }
  }

  /**
   * 定期的なETL実行を開始
   */
  async startScheduled(): Promise<void> {
    if (this.isRunning) {
      traceLog('etl.pipeline.already_running', { message: 'ETL pipeline already running' });
      return;
    }

    this.isRunning = true;
    const intervalMs = this.config.batchIntervalMinutes * 60 * 1000;

    traceLog('etl.pipeline.scheduled_start', {
      interval_minutes: this.config.batchIntervalMinutes,
      interval_ms: intervalMs
    });

    // 初回実行
    await this.runOnce();

    // 定期実行
    const intervalId = setInterval(async () => {
      if (!this.isRunning) {
        clearInterval(intervalId);
        return;
      }

      try {
        await this.runOnce();
      } catch (error) {
        traceLog('etl.pipeline.scheduled_error', {
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }, intervalMs);
  }

  /**
   * 定期実行を停止
   */
  stopScheduled(): void {
    this.isRunning = false;
    traceLog('etl.pipeline.scheduled_stop', { message: 'ETL scheduled execution stopped' });
  }

  /**
   * データ抽出フェーズ
   */
  private async extractData() {
    const allData: any[] = [];

    for (const exchange of this.config.exchanges) {
      for (const pair of this.config.pairs) {
        try {
          // 現在のAPIではexchangeを無視してpairのみを使用（PriceExtractorは'coincheck'固定）
          // 将来的に他取引所対応予定の場合は、extractorの型や実装を拡張してください
          const data = await this.extractor.extract(pair);
          allData.push(data);
        } catch (error) {
          traceLog('etl.pipeline.extract_error', {
            exchange,
            pair,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    return allData;
  }

  /**
   * 特徴量変換フェーズ
   */
  private async transformData(rawData: any[]) {
    try {
      const features: any[] = [];

      for (const data of rawData) {
        // RawMarketData形式に変換
        const marketData = {
          ts: new Date(data.ts).getTime(),
          exchange: data.exchange,
          pair: data.pair,
          price: data.price,
          volume: data.volume,
          spread: data.spread || null,
          depth_imbalance: null,
          order_flow: null
        };

        const feature = await this.transformer.transform(marketData);
        features.push(feature);
      }

      return features;
    } catch (error) {
      traceLog('etl.pipeline.transform_error', {
        error: error instanceof Error ? error.message : String(error),
        raw_data_count: rawData.length
      });
      throw error;
    }
  }

  /**
   * データ保存フェーズ
   */
  private async saveData(features: any[]) {
    try {
      await this.loader.load(features);
    } catch (error) {
      traceLog('etl.pipeline.save_error', {
        error: error instanceof Error ? error.message : String(error),
        features_count: features.length
      });
      throw error;
    }
  }

  /**
   * リトライ付き実行
   */
  private async executeWithRetry<T>(operation: () => Promise<T>): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        if (attempt < this.config.maxRetries) {
          traceLog('etl.pipeline.retry', {
            attempt,
            max_retries: this.config.maxRetries,
            delay_ms: this.config.retryDelayMs,
            error: lastError.message
          });

          await new Promise(resolve => setTimeout(resolve, this.config.retryDelayMs));
        }
      }
    }

    throw lastError;
  }

  /**
   * パイプラインの状態を取得
   */
  getStatus() {
    return {
      is_running: this.isRunning,
      config: this.config,
      loader_status: this.loader.getStatus()
    };
  }

  /**
   * リソースのクリーンアップ
   */
  async cleanup(): Promise<void> {
    this.stopScheduled();
    await this.loader.closeWriter();

    traceLog('etl.pipeline.cleanup', { message: 'ETL pipeline resources cleaned up' });
  }
}

/**
 * ユーティリティ関数
 */
export class ETLPipelineUtils {
  /**
   * 設定ファイルからETL設定を読み込み
   */
  static loadConfigFromFile(configPath: string = 'etl-config.json'): ETLConfig {
    try {
      const fs = require('fs');
      const configData = fs.readFileSync(configPath, 'utf8');
      const userConfig = JSON.parse(configData);
      return { ...DEFAULT_ETL_CONFIG, ...userConfig };
    } catch (error) {
      traceLog('etl.pipeline.config_load_error', {
        config_path: configPath,
        error: error instanceof Error ? error.message : String(error)
      });
      return DEFAULT_ETL_CONFIG;
    }
  }

  /**
   * ETL実行の統計情報を取得
   */
  static async getPipelineStats(baseDir: string = 'data/features') {
    const { ParquetUtils } = await import('./load/parquet.js');
    return ParquetUtils.getDataStats(baseDir);
  }
}