// ETL Load - Parquet形式でのデータ保存
// 特徴量データをParquet形式で保存

import * as fs from 'fs';
import * as path from 'path';
import { ParquetWriter, ParquetSchema } from 'parquets';
import { FeatureRow } from '../../rl/schema';
import { traceLog } from '../../../utils/logging/trace';

export interface ParquetConfig {
  baseDir: string;
  maxRowsPerFile: number;
  maxFileSizeMB: number;
  compression?: 'SNAPPY' | 'GZIP' | 'LZO' | 'BROTLI' | 'LZ4' | 'ZSTD';
}

/**
 * Parquet保存設定
 */
export const DEFAULT_PARQUET_CONFIG: ParquetConfig = {
  baseDir: 'data/features',
  maxRowsPerFile: 10000,
  maxFileSizeMB: 64,
  compression: 'SNAPPY'
};

/**
 * Parquetライター
 */
export class ParquetLoader {
  private config: ParquetConfig;
  private currentFileRows: number = 0;
  private currentFilePath: string | null = null;
  private writer: ParquetWriter<FeatureRow> | null = null;

  constructor(config: ParquetConfig = DEFAULT_PARQUET_CONFIG) {
    this.config = config;
    this.ensureDirectoryExists();
  }

  /**
   * 特徴量データをParquet形式で保存
   */
  async load(features: FeatureRow[]): Promise<void> {
    const startTime = Date.now();

    try {
      if (features.length === 0) {
        traceLog('etl.load.parquet', { message: 'No features to save' });
        return;
      }

      // ファイルローテーション判定
      let filePath = this.currentFilePath;
      if (this.shouldRotateFile() || !this.writer) {
        filePath = this.getNextFilePath();
        await this.closeWriter();
        this.writer = await this.createWriter(filePath);
        this.currentFilePath = filePath;
        this.currentFileRows = 0;
      }

      // データを書き込み
      if (this.writer) {
        for (const feature of features) {
          await this.writer.appendRow(feature);
        }
      }

      this.currentFileRows += features.length;

      traceLog('etl.load.parquet', {
        file_path: filePath,
        rows_saved: features.length,
        total_rows_in_file: this.currentFileRows,
        duration_ms: Date.now() - startTime
      });

    } catch (error) {
      traceLog('etl.load.parquet.error', {
        error: error instanceof Error ? error.message : String(error),
        features_count: features.length,
        duration_ms: Date.now() - startTime
      });
      throw error;
    }
  }

  /**
   * 複数の特徴量データをバッチ保存
   */
  async loadBatch(featureBatches: FeatureRow[][]): Promise<void> {
    for (const batch of featureBatches) {
      await this.load(batch);
    }
  }

  /**
   * Writerを閉じる
   */
  async closeWriter(): Promise<void> {
    if (this.writer) {
      await this.writer.close();
      this.writer = null;
    }
  }

  /**
   * ParquetWriterを作成
   */
  private async createWriter(filePath: string): Promise<ParquetWriter<FeatureRow>> {
    // Parquetスキーマを定義
    const schema = new ParquetSchema({
      ts: { type: 'TIMESTAMP_MILLIS' },           // タイムスタンプ（ミリ秒）
      exchange: { type: 'UTF8' },                  // 取引所名
      pair: { type: 'UTF8' },                      // 通貨ペア
      price: { type: 'DOUBLE' },                   // 価格
      volume: { type: 'DOUBLE' },                  // 取引量
      spread: { type: 'DOUBLE', optional: true },  // スプレッド（買値-売値）
      depth_imbalance: { type: 'DOUBLE', optional: true }, // 板の不均衡度
      order_flow: { type: 'DOUBLE', optional: true },      // オーダーフロー指標
      sma_10: { type: 'DOUBLE', optional: true },          // 10期間単純移動平均
      sma_50: { type: 'DOUBLE', optional: true },          // 50期間単純移動平均
      rsi_14: { type: 'DOUBLE', optional: true },          // 14期間RSI
      atr_14: { type: 'DOUBLE', optional: true },          // 14期間ATR
      bb_width_20: { type: 'DOUBLE', optional: true },     // 20期間ボリンジャーバンド幅
      vol_ratio: { type: 'DOUBLE', optional: true },       // ボリューム比率
      liquidity_score: { type: 'DOUBLE', optional: true }, // 流動性スコア
      pnl: { type: 'DOUBLE', optional: true },             // 損益
      win: { type: 'BOOLEAN', optional: true }             // 勝敗フラグ
    });
    return await ParquetWriter.openFile(schema, filePath);
  }

  /**
   * 次のファイルパスを決定
   */
  private getNextFilePath(): string {
    const now = new Date();
    const year = now.getUTCFullYear();
    const month = String(now.getUTCMonth() + 1).padStart(2, '0');
    const day = String(now.getUTCDate()).padStart(2, '0');

    // 現在のファイルが容量オーバーかチェック
    const shouldRotate = this.shouldRotateFile();

    if (shouldRotate || !this.currentFilePath) {
      // 新しいファイルパスを生成
      const fileName = `features-${year}-${month}-${day}-${Date.now()}.parquet`;
      this.currentFilePath = path.join(this.config.baseDir, String(year), month, fileName);
      this.currentFileRows = 0;

      // ディレクトリが存在することを確認
      const dir = path.dirname(this.currentFilePath);
      fs.mkdirSync(dir, { recursive: true });
    }

    return this.currentFilePath;
  }

  /**
   * ファイルをローテーションすべきか判定
   */
  private shouldRotateFile(): boolean {
    if (!this.currentFilePath) return true;

    // 行数チェック
    if (this.currentFileRows >= this.config.maxRowsPerFile) {
      return true;
    }

    // ファイルサイズチェック
    // ファイルサイズチェック
    if (fs.existsSync(this.currentFilePath)) {
      try {
        const stats = fs.statSync(this.currentFilePath);
        const fileSizeMB = stats.size / (1024 * 1024);
        if (fileSizeMB >= this.config.maxFileSizeMB) {
          return true;
        }
      } catch (error) {
        // stat取得失敗時はローテーション
        return true;
      }
    }
    return false;
  }

  /**
   * ディレクトリが存在することを確認
   */
  private ensureDirectoryExists(): void {
    if (!fs.existsSync(this.config.baseDir)) {
      fs.mkdirSync(this.config.baseDir, { recursive: true });
    }
  }

  /**
   * 現在の状態を取得
   */
  getStatus() {
    return {
      current_file: this.currentFilePath,
      current_rows: this.currentFileRows,
      config: this.config,
      writer_open: this.writer !== null
    };
  }
}

/**
 * ユーティリティ関数
 */
export class ParquetUtils {
  /**
   * 保存されたParquetファイルの一覧を取得
   */
  static listParquetFiles(baseDir: string = 'data/features'): string[] {
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

    scanDir(baseDir);
    return files;
  }

  /**
   * データディレクトリの統計情報を取得
   */
  static getDataStats(baseDir: string = 'data/features') {
    const files = this.listParquetFiles(baseDir);
    let totalSize = 0;
    let totalFiles = files.length;

    for (const file of files) {
      try {
        const stat = fs.statSync(file);
        totalSize += stat.size;
      } catch (error) {
        // ファイルが読み込めない場合はスキップ
      }
    }

    return {
      total_files: totalFiles,
      // ファイルサイズ(B)をMB単位に変換し、小数第2位まで丸める
        total_size_mb: Math.round(totalSize / (1024 * 1024) * 100) / 100,
      files: files
    };
  }
}