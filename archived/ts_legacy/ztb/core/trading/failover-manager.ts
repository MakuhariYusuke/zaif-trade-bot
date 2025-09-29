// Failover Manager - 形だけの実装
// 将来の拡張用にインターフェースを定義

import { OrderRequest } from '../safety/safety-manager';

export interface ExchangeAdapter {
  executeTrade(order: OrderRequest): Promise<any>;
  /**
   * Returns the current health status of the exchange.
   * - 'healthy': Exchange is fully operational.
   * - 'degraded': Exchange is partially operational or experiencing issues.
   * - 'down': Exchange is not operational.
   * Status may transition between these states based on exchange connectivity and error rates.
   */
  getStatus(): Promise<'healthy' | 'degraded' | 'down'>;
}

export class FailoverManager {
  constructor(
    private primaryExchange: ExchangeAdapter,
    private backupExchange?: ExchangeAdapter
  ) {}

  /**
   * Executes a trade using the primary exchange.
   *
   * 現在はプライマリのみ使用。
   * 将来的には、プライマリが 'down' または 'degraded' の場合にバックアップへフェイルオーバーする設計を想定。
   * フェイルオーバー時は、エラー処理や状態遷移のロジックを追加する予定。
   */
  async executeTrade(order: OrderRequest): Promise<any> {
    // 現在はプライマリのみ使用
    return this.primaryExchange.executeTrade(order);
  }

  async getStatus(): Promise<{
    primary: 'healthy' | 'degraded' | 'down';
    backup?: 'healthy' | 'degraded' | 'down';
  }> {
    const primaryStatus = await this.primaryExchange.getStatus();
    const backupStatus = this.backupExchange
      ? await this.backupExchange.getStatus()
      : undefined;

    return {
      primary: primaryStatus,
      backup: backupStatus
    };
  }
}