// RL データエクスポート
// 特徴量データをRL学習用の形式に変換・エクスポート

import * as fs from 'fs';
import * as path from 'path';
import { ParquetReader } from 'parquets';
import { FeatureRow, RLStep } from '../rl/schema';
// import { traceLog } from '../../../utils/logging/trace';

// トレースログ関数（フォールバック）
function traceLog(key: string, data: any): void {
  if (process.env.LOG_TRACE === '1') {
    console.log(`[TRACE:${key}]`, JSON.stringify(data, null, 2));
  }
}

export interface RLExportConfig {
  inputDir: string;
  outputDir: string;
  maxStepsPerFile: number;
  rewardFunction: 'pnl' | 'sharpe' | 'sortino';
  discountFactor: number;
  lookAheadSteps: number;
}

export const DEFAULT_RL_EXPORT_CONFIG: RLExportConfig = {
  inputDir: 'data/features',
  outputDir: 'data/rl',
  maxStepsPerFile: 10000,
  rewardFunction: 'pnl',
  discountFactor: 0.99,
  lookAheadSteps: 5
};

/**
 * RLデータエクスポートクラス
 */
export class RLDataExporter {
  private config: RLExportConfig;

  constructor(config: RLExportConfig = DEFAULT_RL_EXPORT_CONFIG) {
    this.config = config;
    this.ensureDirectories();
  }

  /**
   * 特徴量データをRLステップに変換してエクスポート
   */
  async exportToRL(): Promise<RLExportResult> {
    const startTime = Date.now();

    try {
      traceLog('rl.export.start', {
        input_dir: this.config.inputDir,
        output_dir: this.config.outputDir,
        reward_function: this.config.rewardFunction
      });

      // Parquetファイルの一覧を取得
      const parquetFiles = this.getParquetFiles();

      if (parquetFiles.length === 0) {
        traceLog('rl.export.no_files', { message: 'No Parquet files found' });
        return {
          success: true,
          totalFiles: 0,
          totalSteps: 0,
          exportedFiles: [],
          durationMs: Date.now() - startTime
        };
      }

      // 各ファイルを処理
      const allSteps: (RLStep & { pnl: number })[] = [];
      for (const file of parquetFiles) {
        const steps = await this.processParquetFile(file);
        allSteps.push(...steps);
      }

      // ステップを時系列でソート
      allSteps.sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());

      // リワードを計算
      this.calculateRewards(allSteps);

      // ファイルを分割して保存
      const exportedFiles = await this.saveRLSteps(allSteps);

      const result: RLExportResult = {
        success: true,
        totalFiles: parquetFiles.length,
        totalSteps: allSteps.length,
        exportedFiles,
        durationMs: Date.now() - startTime
      };

      traceLog('rl.export.complete', {
        total_files: result.totalFiles,
        total_steps: result.totalSteps,
        exported_files_count: result.exportedFiles.length,
        duration_ms: result.durationMs
      });

      return result;

    } catch (error) {
      traceLog('rl.export.error', {
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - startTime
      });

      return {
        success: false,
        totalFiles: 0,
        totalSteps: 0,
        exportedFiles: [],
        durationMs: Date.now() - startTime,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * ParquetファイルをRLステップに変換
   */
  private async processParquetFile(filePath: string): Promise<(RLStep & { pnl: number })[]> {
    const steps: (RLStep & { pnl: number })[] = [];

    // ファイルごとに一意なepisode_idを生成
    const episodeId = `ep_${path.basename(filePath, '.parquet')}_${Date.now()}`;

    try {
      const reader = await ParquetReader.openFile(filePath);
      const cursor = reader.getCursor();

      let record: FeatureRow | null = null;
      while (record = await cursor.next()) {
        const step = this.convertFeatureToRLStep(record, episodeId);
        steps.push(step);
      }

      await reader.close();

      traceLog('rl.export.file_processed', {
        file_path: filePath,
        steps_count: steps.length
      });

    } catch (error) {
      traceLog('rl.export.file_error', {
        file_path: filePath,
        error: error instanceof Error ? error.message : String(error)
      });
    }

    return steps;
  }

  /**
   * FeatureRowをRLStepに変換
   */
  private convertFeatureToRLStep(feature: FeatureRow, episode_id: string): RLStep & { pnl: number } {
    // 特徴量ベクトルを作成（数値フィールドのみ）
    const state = [
      feature.price,
      feature.volume,
      feature.spread || 0,
      feature.depth_imbalance || 0,
      feature.order_flow || 0,
      feature.sma_10 || 0,
      feature.sma_50 || 0,
      feature.rsi_14 || 50,
      feature.atr_14 || 0,
      feature.bb_width_20 || 0,
      feature.vol_ratio || 1,
      feature.liquidity_score || 0
    ];

    return {
      ts: feature.ts,
      pair: feature.pair,
      episode_id,
      state,
      action: 0 as -1 | 0 | 1, // デフォルト値、後で計算
      reward: 0, // 後で計算
      next_state: [], // 後で計算
      done: 0 as 0 | 1, // 後で計算
      pnl: feature.pnl || 0 // PnL情報を保持
    };
  }

  /**
   * リワードを計算
   */
  private calculateRewards(steps: (RLStep & { pnl: number })[]): void {
    for (let i = 0; i < steps.length; i++) {
      const step = steps[i];

      switch (this.config.rewardFunction) {
        case 'pnl':
          step.reward = step.pnl;
          break;

        case 'sharpe':
          step.reward = this.calculateSharpeRatio(steps, i);
          break;

        case 'sortino':
          step.reward = this.calculateSortinoRatio(steps, i);
          break;

        default:
          step.reward = step.pnl;
      }

      // 次の状態を設定
      if (i < steps.length - 1) {
        step.next_state = steps[i + 1].state;
        step.done = 0;
      } else {
        step.next_state = step.state; // 最後のステップ
        step.done = 1;
      }

      // アクションを決定（簡易版: 価格変化に基づく）
      step.action = this.determineAction(steps, i);
    }
  }

  /**
   * Sharpe Ratioを計算
   */
  private calculateSharpeRatio(steps: (RLStep & { pnl: number })[], currentIndex: number): number {
    const lookback = Math.min(20, currentIndex + 1);
    const recentPnls = steps.slice(currentIndex - lookback + 1, currentIndex + 1)
      .map(s => s.pnl);

    const avgReturn = recentPnls.reduce((a, b) => a + b, 0) / recentPnls.length;
    const variance = recentPnls.reduce((a, b) => a + Math.pow(b - avgReturn, 2), 0) / recentPnls.length;
    const stdDev = Math.sqrt(variance);

    return stdDev > 0 ? avgReturn / stdDev : 0;
  }

  /**
   * Sortino Ratioを計算
   */
  private calculateSortinoRatio(steps: (RLStep & { pnl: number })[], currentIndex: number): number {
    const lookback = Math.min(20, currentIndex + 1);
    const recentPnls = steps.slice(currentIndex - lookback + 1, currentIndex + 1)
      .map(s => s.pnl);

    const avgReturn = recentPnls.reduce((a, b) => a + b, 0) / recentPnls.length;
    const targetReturn = 0;
    const negativeReturns = recentPnls.filter(pnl => pnl < targetReturn);
    const downsideVariance = negativeReturns.length > 0
      ? negativeReturns.reduce((a, b) => a + Math.pow(b - targetReturn, 2), 0) / negativeReturns.length
      : 0;
    const downsideStdDev = Math.sqrt(downsideVariance);

    return downsideStdDev > 0 ? avgReturn / downsideStdDev : 0;
  }

  /**
   * アクションを決定（簡易版）
   */
  private determineAction(steps: (RLStep & { pnl: number })[], currentIndex: number): -1 | 0 | 1 {
    if (currentIndex === 0) return 0; // HOLD

    const currentPrice = steps[currentIndex].state[0];
    const prevPrice = steps[currentIndex - 1].state[0];

    const priceChange = (currentPrice - prevPrice) / prevPrice;

    if (priceChange > 0.001) return 1; // BUY
    if (priceChange < -0.001) return -1; // SELL
    return 0; // HOLD
  }

  /**
   * RLステップをファイルに保存
   */
  private async saveRLSteps(steps: RLStep[]): Promise<string[]> {
    const exportedFiles: string[] = [];
    const batches = this.chunkArray(steps, this.config.maxStepsPerFile);

    const timestamp = Date.now();
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i];
      const fileName = `rl-steps-${timestamp}-${i}.json`;
      const filePath = path.join(this.config.outputDir, fileName);

      const data = {
        metadata: {
          created_at: new Date().toISOString(),
          total_steps: batch.length,
          reward_function: this.config.rewardFunction,
          discount_factor: this.config.discountFactor,
          config: this.config
        },
        steps: batch
      };

      await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2));
      exportedFiles.push(filePath);

      traceLog('rl.export.file_saved', {
        file_path: filePath,
        steps_count: batch.length
      });
    }

    return exportedFiles;
  }

  /**
   * Parquetファイルの一覧を取得
   */
  private getParquetFiles(): string[] {
    const files: string[] = [];

    function scanDir(dir: string) {
      if (!fs.existsSync(dir)) return;

      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          scanDir(fullPath);
        } else if (item.endsWith('.parquet')) {
          files.push(fullPath);
        }
      }
    }

    scanDir(this.config.inputDir);
    return files.sort(); // 時系列順にソート
  }

  /**
   * 配列をチャンクに分割
   */
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  /**
   * ディレクトリが存在することを確認
   */
  private ensureDirectories(): void {
    if (!fs.existsSync(this.config.outputDir)) {
      fs.mkdirSync(this.config.outputDir, { recursive: true });
    }
  }

  /**
   * エクスポート統計を取得
   */
  async getExportStats(): Promise<RLExportStats> {
    const outputFiles = fs.readdirSync(this.config.outputDir)
      .filter(file => file.endsWith('.json'))
      .map(file => path.join(this.config.outputDir, file));

    let totalSteps = 0;
    const fileStats: Array<{ file: string; steps: number; size_mb: number }> = [];

    for (const file of outputFiles) {
      try {
        const content = fs.readFileSync(file, 'utf8');
        const data = JSON.parse(content);
        const steps = data.steps?.length || 0;
        const stat = fs.statSync(file);
        const sizeMB = stat.size / (1024 * 1024);

        totalSteps += steps;
        fileStats.push({
          file: path.basename(file),
          steps,
          size_mb: Math.round(sizeMB * 100) / 100
        });
      } catch (error) {
        traceLog('rl.export.stats_file_error', {
          file_path: file,
          error: error instanceof Error ? error.message : String(error)
        });
        // ファイル読み込みエラーはスキップ
      }
    }

    return {
      total_files: outputFiles.length,
      total_steps: totalSteps,
      files: fileStats
    };
  }
}

/**
 * エクスポート結果
 */
export interface RLExportResult {
  success: boolean;
  totalFiles: number;
  totalSteps: number;
  exportedFiles: string[];
  durationMs: number;
  error?: string;
}

/**
 * エクスポート統計
 */
export interface RLExportStats {
  total_files: number;
  total_steps: number;
  files: Array<{
    file: string;
    steps: number;
    size_mb: number;
  }>;
}

/**
 * ユーティリティ関数
 */
export class RLExportUtils {
  /**
   * RLデータをトレーニング/テストに分割
   */
  static splitTrainTest(
    steps: RLStep[],
    trainRatio: number = 0.8
  ): { train: RLStep[]; test: RLStep[] } {
    const splitIndex = Math.floor(steps.length * trainRatio);
    return {
      train: steps.slice(0, splitIndex),
      test: steps.slice(splitIndex)
    };
  }

  /**
   * RLデータを時系列クロスバリデーション用に分割
   */
  static splitTimeSeriesCV(
    steps: RLStep[],
    nSplits: number = 5
  ): Array<{ train: RLStep[]; test: RLStep[] }> {
    const splits: Array<{ train: RLStep[]; test: RLStep[] }> = [];
    const testSize = Math.floor(steps.length / nSplits);

    for (let i = 0; i < nSplits; i++) {
      const testStart = i * testSize;
      const testEnd = (i + 1) * testSize;

      splits.push({
        train: steps.slice(0, testStart),
        test: steps.slice(testStart, testEnd)
      });
    }

    return splits;
  }
}