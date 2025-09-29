// ETL Transform - 特徴量計算
// 取得した生データからテクニカル指標を計算

import { PriceData, OrderBookData, TradeData } from '../extract/price';
import { FeatureRow } from '../../rl/schema';
import { traceLog } from '../../../utils/logging/trace';
import { wilderSmooth } from '../../../utils/indicators/utils';

// 簡易テクニカル指標計算関数
function calculateSMA(values: number[], period: number): number | null {
  if (values.length < period) return null;
  const window = values.slice(-period);
  return window.reduce((sum, val) => sum + val, 0) / period;
}

function calculateRSI(values: number[], period: number): number | null {
  if (values.length < period + 1) return null;

  const gains: number[] = [];
  const losses: number[] = [];

  for (let i = 1; i < values.length; i++) {
    const change = values[i] - values[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }

  if (gains.length < period) return null;

  const avgGain = gains.slice(-period).reduce((sum, val) => sum + val, 0) / period;
  const avgLoss = losses.slice(-period).reduce((sum, val) => sum + val, 0) / period;

  if (avgLoss === 0) return 100;

  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

function calculateBBWidth(values: number[], period: number): number | null {
  if (values.length < period) return null;

  const window = values.slice(-period);
  const sma = window.reduce((sum, val) => sum + val, 0) / period;
  const variance = window.reduce((sum, val) => sum + Math.pow(val - sma, 2), 0) / period;
  const std = Math.sqrt(variance);

  const upperBand = sma + (std * 2);
  const lowerBand = sma - (std * 2);

  return (upperBand - lowerBand) / sma; // 幅をSMAで正規化
}

export interface RawMarketData {
  price: PriceData;
  orderbook?: OrderBookData;
  trades?: TradeData[];
}

/**
 * 特徴量変換クラス
 */
export class FeatureTransformer {
  private priceHistory: number[] = [];
  private volumeHistory: number[] = [];

  /**
   * 生データを特徴量に変換
   */
  async transform(rawData: RawMarketData): Promise<FeatureRow> {
    const startTime = Date.now();

    try {
      // 価格履歴を更新
      this.priceHistory.push(rawData.price.price);
      this.volumeHistory.push(rawData.price.volume);

      // 最大1000件に制限
      if (this.priceHistory.length > 1000) {
        this.priceHistory.shift();
        this.volumeHistory.shift();
      }

      // 基本特徴量
      const baseFeatures: Partial<FeatureRow> = {
        ts: rawData.price.ts,
        exchange: rawData.price.exchange,
        pair: rawData.price.pair,
        price: rawData.price.price,
        volume: rawData.price.volume,
      };

      // 板情報がある場合は追加
      if (rawData.orderbook) {
        baseFeatures.spread = rawData.orderbook.spread;
        baseFeatures.depth_imbalance = this.calculateDepthImbalance(rawData.orderbook);
        baseFeatures.order_flow = this.calculateOrderFlow(rawData.trades || []);
      }

      // テクニカル指標の計算
      const technicalFeatures = await this.calculateTechnicalIndicators();
      const atr14 = technicalFeatures.atr_14 ?? null;
      const riskFeatures = this.calculateRiskFeatures(atr14);

      const featureRow: FeatureRow = {
        ...baseFeatures,
        ...technicalFeatures,
        ...riskFeatures,
      } as FeatureRow;

      /**
       * traceLog: ETL特徴量変換の処理状況を記録
       * - ログキー: 'etl.transform.features'
       * - 出力内容: ペア名、価格、計算した指標数、処理時間(ms)
       * - 目的: ETL処理のパフォーマンス・進捗監視
       * - タイミング: 特徴量変換完了時
       */
      traceLog('etl.transform.features', {
        pair: rawData.price.pair,
        price: rawData.price.price,
        indicators_calculated: Object.keys(technicalFeatures).length,
        duration_ms: Date.now() - startTime
      });

      return featureRow;

    } catch (error) {
      /**
       * traceLog: ETL特徴量変換のエラーを記録
       * - ログキー: 'etl.transform.features.error'
       * - 出力内容: ペア名、エラーメッセージ、処理時間(ms)
       * - 目的: ETL処理の例外監視・デバッグ
       * - タイミング: 特徴量変換で例外発生時
       */
      traceLog('etl.transform.features.error', {
        pair: rawData.price.pair,
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - startTime
      });
      throw error;
    }
  }

  /**
   * テクニカル指標を計算
   */
  private async calculateTechnicalIndicators(): Promise<Partial<FeatureRow>> {
    const features: Partial<FeatureRow> = {};

    try {
      // SMA計算
      if (this.priceHistory.length >= 10) {
        features.sma_10 = calculateSMA(this.priceHistory, 10);
      }
      if (this.priceHistory.length >= 50) {
        features.sma_50 = calculateSMA(this.priceHistory, 50);
      }

      // RSI計算
      if (this.priceHistory.length >= 14) {
        features.rsi_14 = calculateRSI(this.priceHistory, 14);
      }

      // ATR計算（簡易版）
      if (this.priceHistory.length >= 14) {
        features.atr_14 = this.calculateSimpleAtr(14);
      }

      // Bollinger Band幅計算
      if (this.priceHistory.length >= 20) {
        features.bb_width_20 = calculateBBWidth(this.priceHistory, 20);
      }

    } catch (error) {
      /**
       * traceLog: テクニカル指標計算のエラーを記録
       * - ログキー: 'etl.transform.technical.error'
       * - 出力内容: エラーメッセージ、価格履歴長
       * - 目的: テクニカル指標計算の例外監視
       * - タイミング: 指標計算で例外発生時
       */
      traceLog('etl.transform.technical.error', {
        error: error instanceof Error ? error.message : String(error),
        price_history_length: this.priceHistory.length
      });
    }

    return features;
  }

  /**
   * リスク関連特徴量を計算
   * @param atr14 ATR値（テクニカル指標計算から受け取る）
   */
  private calculateRiskFeatures(atr14: number | null): Partial<FeatureRow> {
    const features: Partial<FeatureRow> = {};

    try {
      // ボラティリティ比 (ATR/価格)
      if (this.priceHistory.length > 0 && atr14 !== null && atr14 !== undefined) {
        const currentPrice = this.priceHistory[this.priceHistory.length - 1];
        features.vol_ratio = atr14 / currentPrice;
      }

      // 流動性スコア（出来高ベースの簡易計算）
      if (this.volumeHistory.length >= 10) {
        const avgVolume = this.volumeHistory.slice(-10).reduce((sum, vol) => sum + vol, 0) / 10;
        const currentVolume = this.volumeHistory[this.volumeHistory.length - 1];
        features.liquidity_score = Math.min(currentVolume / avgVolume, 2) * 0.5; // 0-1に正規化
      }

    } catch (error) {
      /**
       * traceLog: リスク特徴量計算のエラーを記録
       * - ログキー: 'etl.transform.risk.error'
       * - 出力内容: エラーメッセージ
       * - 目的: リスク指標計算の例外監視
       * - タイミング: リスク特徴量計算で例外発生時
       */
      traceLog('etl.transform.risk.error', {
        error: error instanceof Error ? error.message : String(error)
      });
    }

    return features;
  }

  /**
   * 板の不均衡度を計算
   */
  private calculateDepthImbalance(orderbook: OrderBookData): number {
    try {
      const bidVolume = orderbook.bids.slice(0, 5).reduce((sum, [_, amount]) => sum + amount, 0);
      const askVolume = orderbook.asks.slice(0, 5).reduce((sum, [_, amount]) => sum + amount, 0);
      const totalVolume = bidVolume + askVolume;

      if (totalVolume === 0) return 0;

      // -1 (買い圧力強) から 1 (売り圧力強) の範囲
      return (askVolume - bidVolume) / totalVolume;
    } catch (error) {
      return 0;
    }
  }

  /**
   * 注文フローを計算（売買比率ベース）
   */
  private calculateOrderFlow(trades: TradeData[]): number {
    try {
      if (trades.length === 0) return 0;

      const recentTrades = trades.slice(-10); // 直近10件
      const buyVolume = recentTrades
        .filter(t => t.side === 'buy')
        .reduce((sum, t) => sum + t.amount, 0);
      const sellVolume = recentTrades
        .filter(t => t.side === 'sell')
        .reduce((sum, t) => sum + t.amount, 0);
      const totalVolume = buyVolume + sellVolume;

      if (totalVolume === 0) return 0;

      // -1 (売り優勢) から 1 (買い優勢) の範囲
      return (buyVolume - sellVolume) / totalVolume;
    } catch (error) {
      return 0;
    }
  }

  /**
   * ATRの簡易計算（True Rangeベース）
   */
  private calculateSimpleAtr(period: number): number {
    try {
      if (this.priceHistory.length < period + 1) return 0;

      const trValues: number[] = [];
      for (let i = 1; i < Math.min(this.priceHistory.length, period + 1); i++) {
        const current = this.priceHistory[this.priceHistory.length - i];
        const previous = this.priceHistory[this.priceHistory.length - i - 1];
        const tr = Math.abs(current - previous); // 簡易版TR
        trValues.push(tr);
      }

      if (trValues.length === 0) return 0;

      // Wilderの平滑化
      let atr = trValues[0];
      for (let i = 1; i < trValues.length; i++) {
        atr = wilderSmooth(atr, trValues[i], period);
      }

      return atr;
    } catch (error) {
      return 0;
    }
  }

  /**
   * 履歴をクリア（テスト用）
   */
  clearHistory(): void {
    this.priceHistory = [];
    this.volumeHistory = [];
  }
}

/**
 * ラベル変換クラス（将来のRL報酬設計用）
 */
export class LabelTransformer {
  /**
   * 簡易ラベル生成（次の価格変動を予測）
   */
  transform(features: FeatureRow[]): FeatureRow[] {
    // 最後の要素は次の価格がないため、win/pnlをnullで埋めて一貫性を持たせる
    return features.map((feature, index) => {
      if (index < features.length - 1) {
        const nextPrice = features[index + 1].price;
        const currentPrice = feature.price;
        const pnl = (nextPrice - currentPrice) / currentPrice;

        return {
          ...feature,
          pnl,
          win: pnl > 0 ? 1 : 0
        };
      }
      return {
        ...feature,
        pnl: null,
        win: null
      };
    });
  }
}