// Safety Manager - 縮小モード実装
// 完全停止ではなく、資金をゼロにしない安全弁として機能

import { traceLog } from '../../utils/logging/trace';

// 注文リクエスト型定義
export interface OrderRequest {
  pair: string;
  side: 'bid' | 'ask';
  size: number;
  price?: number;
  type?: 'market' | 'limit';
}

export interface SafetyConfig {
  // 基本設定
  maxOrdersPerHour: number;    // 1時間あたりの最大注文数
  minSize: number;             // 最小注文サイズ
  maxDailyLossPct: number;     // 当日最大損失率 (%)

  // 縮小係数設定
  drawdownScaleFactor: number; // ドローダウンによる縮小係数
  volatilityScaleFactor: number; // ボラティリティによる縮小係数
  frequencyScaleFactor: number; // 頻度超過による縮小係数

  // 閾値設定
  highDrawdownThreshold: number; // 高ドローダウン閾値 (%)
  highVolatilityThreshold: number; // 高ボラティリティ閾値
  circuitBreakerThreshold: number; // サーキットブレーカー閾値 (%)
}

export interface MetricsPort {
  getRollingDrawdown(period: string): number; // 期間指定のドローダウン取得
  getVolatility(indicator: string): number;   // ボラティリティ指標取得
  getOrderCount(period: string): number;      // 期間指定の注文数取得
  getDailyPnL(): number;                      // 当日の損益取得
}

// デフォルト設定
export const DEFAULT_SAFETY_CONFIG: SafetyConfig = {
  maxOrdersPerHour: 60,
  minSize: 0.001,
  maxDailyLossPct: 5.0,

  drawdownScaleFactor: 0.7,    // ドローダウンで70%に縮小
  volatilityScaleFactor: 0.8,  // ボラティリティで80%に縮小
  frequencyScaleFactor: 0.5,   // 頻度超過で50%に縮小

  highDrawdownThreshold: 2.0,   // 2%以上のドローダウンで縮小開始
  highVolatilityThreshold: 1000, // ATR 1000以上の場合高ボラ
  circuitBreakerThreshold: 5.0   // 5%損失で最小サイズに
};

export class SafetyManager {
  constructor(
    private cfg: SafetyConfig = DEFAULT_SAFETY_CONFIG,
    private metrics: MetricsPort
  ) {}

  /**
   * 注文リクエストを安全基準で調整
   */
  adjust(order: OrderRequest): OrderRequest {
    const originalSize = order.size;
    let adjustedSize = originalSize;

    traceLog('safety.adjust.start', {
      original_size: originalSize,
      pair: order.pair,
      side: order.side
    });

    // 1. ドローダウンによる縮小
    const drawdown = this.metrics.getRollingDrawdown('24h');
    adjustedSize *= this.scaleByDrawdown(drawdown);

    // 2. ボラティリティによる縮小
    const volatility = this.metrics.getVolatility('ATR14');
    adjustedSize *= this.scaleByVolatility(volatility);

    // 3. 頻度制限による縮小
    const orderCount = this.metrics.getOrderCount('1h');
    adjustedSize *= this.scaleByFrequency(orderCount);

    // 4. サーキットブレーカー適用
    const dailyPnL = this.metrics.getDailyPnL();
    if (Math.abs(dailyPnL) > this.cfg.circuitBreakerThreshold) {
      adjustedSize = this.cfg.minSize;
      traceLog('safety.circuit_breaker', {
        daily_pnl: dailyPnL,
        threshold: this.cfg.circuitBreakerThreshold,
        adjusted_size: adjustedSize
      });
    }

    // 最小サイズ保証
    adjustedSize = Math.max(adjustedSize, this.cfg.minSize);

    const adjustment = {
      original_size: originalSize,
      adjusted_size: adjustedSize,
      scale_factor: adjustedSize / originalSize,
      drawdown,
      volatility,
      order_count: orderCount,
      daily_pnl: dailyPnL
    };

    traceLog('safety.adjust.complete', adjustment);

    return {
      ...order,
      size: adjustedSize
    };
  }

  /**
   * ドローダウンによるサイズ縮小
   */
  private scaleByDrawdown(drawdownPct: number): number {
    if (drawdownPct <= 0) return 1.0; // 利益時は縮小なし

    const lossPct = Math.abs(drawdownPct);
    if (lossPct < this.cfg.highDrawdownThreshold) return 1.0;

    // 線形縮小: threshold以上でscaleFactorまで縮小
    const excess = lossPct - this.cfg.highDrawdownThreshold;
    const maxExcess = this.cfg.maxDailyLossPct - this.cfg.highDrawdownThreshold;
    const scale = 1.0 - (excess / maxExcess) * (1.0 - this.cfg.drawdownScaleFactor);

    return Math.max(scale, this.cfg.drawdownScaleFactor);
  }

  /**
   * ボラティリティによるサイズ縮小
   */
  private scaleByVolatility(volatility: number): number {
    if (volatility < this.cfg.highVolatilityThreshold) return 1.0;

    // ボラティリティが高いほど強く縮小
    const excess = volatility - this.cfg.highVolatilityThreshold;
    const scale = Math.pow(this.cfg.volatilityScaleFactor, excess / 1000);

    return Math.max(scale, 0.1); // 最低10%まで
  }

  /**
   * 注文頻度によるサイズ縮小
   */
  private scaleByFrequency(orderCount: number): number {
    if (orderCount <= this.cfg.maxOrdersPerHour) return 1.0;

    // 超過分に応じて線形縮小
    const excess = orderCount - this.cfg.maxOrdersPerHour;
    const scale = 1.0 - (excess / this.cfg.maxOrdersPerHour) * (1.0 - this.cfg.frequencyScaleFactor);

    return Math.max(scale, this.cfg.frequencyScaleFactor);
  }

  /**
   * 設定更新
   */
  updateConfig(newConfig: Partial<SafetyConfig>): void {
    const merged = { ...this.cfg, ...newConfig };
    // 必須項目チェック
    const requiredKeys: (keyof SafetyConfig)[] = [
      'maxOrdersPerHour',
      'minSize',
      'maxDailyLossPct',
      'drawdownScaleFactor',
      'volatilityScaleFactor',
      'frequencyScaleFactor',
      'highDrawdownThreshold',
      'highVolatilityThreshold',
      'circuitBreakerThreshold'
    ];
    for (const key of requiredKeys) {
      if (merged[key] === undefined || merged[key] === null) {
        throw new Error(`SafetyConfigの必須項目が欠落しています: ${key}`);
      }
    }
    this.cfg = merged;
    traceLog('safety.config.updated', { config: this.cfg });
  }

  /**
   * 現在の安全状態を取得
   */
  getSafetyStatus(): {
    drawdown: number;
    volatility: number;
    orderCount: number;
    dailyPnL: number;
    isCircuitBreakerActive: boolean;
  } {
    const drawdown = this.metrics.getRollingDrawdown('24h');
    const volatility = this.metrics.getVolatility('ATR14');
    const orderCount = this.metrics.getOrderCount('1h');
    const dailyPnL = this.metrics.getDailyPnL();

    return {
      drawdown,
      volatility,
      orderCount,
      dailyPnL,
      isCircuitBreakerActive: Math.abs(dailyPnL) > this.cfg.circuitBreakerThreshold
    };
  }
}

// 簡易メトリクス実装（本番では適切な実装に置き換え）
export class SimpleMetrics implements MetricsPort {
  private orders: number[] = [];
  private pnlHistory: number[] = [];

  getRollingDrawdown(period: string): number {
    // 簡易実装: 直近の損益合計
    const recentPnL = this.pnlHistory.slice(-10).reduce((sum, pnl) => sum + pnl, 0);
    return recentPnL < 0 ? Math.abs(recentPnL) : 0;
  }

  getVolatility(indicator: string): number {
    // 簡易実装: 固定値
    return 500; // ATRの適当な値
  }

  getOrderCount(period: string): number {
    // period例: '1h', '24h', '30m'
    let ms = 60 * 60 * 1000; // デフォルト1時間
    const match = period.match(/^(\d+)([hm])$/);
    if (match) {
      const value = parseInt(match[1], 10);
      const unit = match[2];
      if (unit === 'h') ms = value * 60 * 60 * 1000;
      if (unit === 'm') ms = value * 60 * 1000;
    }
    const since = Date.now() - ms;
    return this.orders.filter(ts => ts > since).length;
  }

  getDailyPnL(): number {
    // 簡易実装: 当日の損益合計
    return this.pnlHistory.slice(-24).reduce((sum, pnl) => sum + pnl, 0);
  }

  // テスト用メソッド
  recordOrder(): void {
    this.orders.push(Date.now());
  }

  recordPnL(pnl: number): void {
    this.pnlHistory.push(pnl);
  }
}