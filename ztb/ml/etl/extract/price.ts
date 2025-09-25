// ETL Extract - 価格データ取得
// Coincheck APIから価格データを取得

import { traceLog } from '../../../utils/logging/trace';

export interface PriceData {
  ts: string;
  pair: string;
  price: number;
  volume: number;
  exchange: 'coincheck';
}

export interface OrderBookData {
  ts: string;
  pair: string;
  bids: Array<[number, number]>; // [price, amount][]
  asks: Array<[number, number]>; // [price, amount][]
  spread: number;
  exchange: 'coincheck';
}

export interface TradeData {
  ts: string;
  pair: string;
  side: 'buy' | 'sell';
  price: number;
  amount: number;
  exchange: 'coincheck';
}

/**
 * 価格データ取得（Ticker/最新価格）
 */
export class PriceExtractor {
  async extract(pair: string = 'btc_jpy'): Promise<PriceData> {
    const startTime = Date.now();

    try {
      // Coincheck APIから価格データを取得
      // 実際の実装ではAPIコールを行う
      const mockData: PriceData = {
        ts: new Date().toISOString(),
        pair,
        price: 5000000 + Math.random() * 100000, // モック価格
        volume: Math.random() * 100,
        exchange: 'coincheck'
      };

      traceLog('etl.extract.price', {
        pair,
        price: mockData.price,
        volume: mockData.volume,
        duration_ms: Date.now() - startTime
      });

      return mockData;
    } catch (error) {
      traceLog('etl.extract.price.error', {
        pair,
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - startTime
      });
      throw error;
    }
  }

  /**
   * 複数ペアの価格データを取得（同時実行数制限付き）
   */
  async extractMultiple(pairs: string[] = ['btc_jpy', 'eth_jpy'], concurrency: number = 2): Promise<PriceData[]> {
    const results: PriceData[] = [];
    let idx = 0;

    while (idx < pairs.length) {
      const batch = pairs.slice(idx, idx + concurrency);
      const batchResults = await Promise.all(batch.map(pair => this.extract(pair)));
      results.push(...batchResults);
      idx += concurrency;
    }

    return results;
  }
}

/**
 * 板情報取得
 */
export class OrderBookExtractor {
  async extract(pair: string = 'btc_jpy', depth: number = 10): Promise<OrderBookData> {
    const startTime = Date.now();

    try {
      // Coincheck APIから板情報を取得
      // 実際の実装ではAPIコールを行う
      const basePrice = 5000000;
      const mockBids: Array<[number, number]> = [];
      const mockAsks: Array<[number, number]> = [];

      // 買い板（bid）を生成
      for (let i = 0; i < depth; i++) {
        const price = basePrice - (i + 1) * 1000;
        const amount = Math.random() * 10;
        mockBids.push([price, amount]);
      }

      // 売り板（ask）を生成
      for (let i = 0; i < depth; i++) {
        const price = basePrice + (i + 1) * 1000;
        const amount = Math.random() * 10;
        mockAsks.push([price, amount]);
      }

      const spread = mockAsks[0][0] - mockBids[0][0];

      const mockData: OrderBookData = {
        ts: new Date().toISOString(),
        pair,
        bids: mockBids,
        asks: mockAsks,
        spread,
        exchange: 'coincheck'
      };

      traceLog('etl.extract.orderbook', {
        pair,
        depth,
        spread,
        bid_count: mockBids.length,
        ask_count: mockAsks.length,
        duration_ms: Date.now() - startTime
      });

      return mockData;
    } catch (error) {
      traceLog('etl.extract.orderbook.error', {
        pair,
        depth,
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - startTime
      });
      throw error;
    }
  }
}

/**
 * 取引履歴取得
 */
export class TradeExtractor {
  async extract(pair: string = 'btc_jpy', limit: number = 100): Promise<TradeData[]> {
    const startTime = Date.now();

    try {
      // Coincheck APIから取引履歴を取得
      // 実際の実装ではAPIコールを行う
      const mockTrades: TradeData[] = [];
      const basePrice = 5000000;

      for (let i = 0; i < limit; i++) {
        // 1分ごとに過去へ遡るタイムスタンプを生成
        const tradeTimestamp = new Date(Date.now() - i * 60000).toISOString();
        const trade: TradeData = {
          ts: tradeTimestamp,
          pair,
          side: Math.random() > 0.5 ? 'buy' : 'sell',
          price: basePrice + (Math.random() - 0.5) * 20000,
          amount: Math.random() * 5,
          exchange: 'coincheck'
        };
        mockTrades.push(trade);
      }

      traceLog('etl.extract.trades', {
        pair,
        limit,
        trade_count: mockTrades.length,
        duration_ms: Date.now() - startTime
      });

      return mockTrades;
    } catch (error) {
      traceLog('etl.extract.trades.error', {
        pair,
        limit,
        error: error instanceof Error ? error.message : String(error),
        duration_ms: Date.now() - startTime
      });
      throw error;
    }
  }
}