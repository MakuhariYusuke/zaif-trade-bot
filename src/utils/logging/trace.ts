// Traceログユーティリティ
// LOG_TRACE=1 で有効化される詳細ログ出力
// ETL・指標計算・発注フローで使用

import { logTrace as originalLogTrace } from '../../utils/logger';

/**
 * 階層キー付きTraceログ出力
 * @param key 階層キー (例: 'etl.extract.price', 'trading.order.submit')
 * @param data ログデータ
 * @param meta 追加メタデータ（traceDataにフラットに展開されるため、keyの衝突に注意）
 * 注意: metaの内容はtraceDataにフラットに展開されるため、keyの衝突に注意してください。
 */
export function traceLog(key: string, data: any, meta?: Record<string, any>): void {
  if (process.env.LOG_TRACE !== '1') return;

  const traceData = {
    key,
    data,
    ...(meta ?? {}),
    timestamp: new Date().toISOString(),
    pid: process.pid,
    memory: process.memoryUsage?.()
  };

  originalLogTrace(`[TRACE:${key}]`, traceData);
}

/**
 * ETL関連Traceログ
 */
export const etlTrace = {
  extract: {
    price: (data: any) => traceLog('etl.extract.price', data),
    orderbook: (data: any) => traceLog('etl.extract.orderbook', data),
    trades: (data: any) => traceLog('etl.extract.trades', data),
  },
  transform: {
    features: (data: any) => traceLog('etl.transform.features', data),
    labels: (data: any) => traceLog('etl.transform.labels', data),
  },
  load: {
    parquet: (data: any) => traceLog('etl.load.parquet', data),
  }
};

/**
 * 指標計算関連Traceログ
 */
export const indicatorTrace = {
  calculate: (indicator: string, params: any, result: any) =>
    traceLog(`indicator.${indicator}.calculate`, { params, result }),

  memoize: (indicator: string, key: string, hit: boolean) =>
    traceLog(`indicator.${indicator}.memoize`, { key, hit }),

  error: (indicator: string, error: any, params: any) =>
    traceLog(`indicator.${indicator}.error`, { error: error instanceof Error ? error.message : String(error), params })
};

/**
 * 取引関連Traceログ
 */
export const tradingTrace = {
  order: {
    submit: (order: any) => traceLog('trading.order.submit', order),
    fill: (fill: any) => traceLog('trading.order.fill', fill),
    cancel: (cancel: any) => traceLog('trading.order.cancel', cancel),
  },
  risk: {
    check: (check: any) => traceLog('trading.risk.check', check),
    adjust: (adjustment: any) => traceLog('trading.risk.adjust', adjustment),
  },
  safety: {
    scale: (scaling: any) => traceLog('trading.safety.scale', scaling),
    limit: (limit: any) => traceLog('trading.safety.limit', limit),
  }
};

/**
 * パフォーマンス計測付きTraceログ
 */
export function withPerformanceTrace<T>(
  key: string,
  operation: () => T,
  meta?: Record<string, any>
): T {
  const NS_PER_MS = 1_000_000;
  const start = process.hrtime.bigint();
  try {
    const result = operation();
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / NS_PER_MS;

    traceLog(`${key}.performance`, {
      duration_ms: durationMs,
      success: true
    }, meta);

    return result;
  } catch (error) {
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / NS_PER_MS;

    traceLog(`${key}.performance`, {
      duration_ms: durationMs,
      success: false,
      error: error instanceof Error ? error.message : String(error)
    }, meta);

    throw error;
  }
}

/**
 * メモリ使用量Traceログ
 */
export function memoryTrace(key: string, meta?: Record<string, any>): void {
  if (process.env.LOG_TRACE !== '1') return;

  const memUsage = process.memoryUsage();
  const MB = 1024 * 1024;
  traceLog(`${key}.memory`, {
    rss_mb: Math.round(memUsage.rss / MB),
    heap_used_mb: Math.round(memUsage.heapUsed / MB),
    heap_total_mb: Math.round(memUsage.heapTotal / MB),
    external_mb: Math.round(memUsage.external / MB)
  }, meta);
}