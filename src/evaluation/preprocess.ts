import * as pd from 'pandas-js';

export class SmartPreprocessor {
  private cache: Map<string, any> = new Map();

  preprocess(data: any, required: string[]): any {
    const cacheKey = JSON.stringify(required);
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    let result = data.copy();

    required.forEach(req => {
      if (req.startsWith('ema:')) {
        const period = parseInt(req.split(':')[1]);
        result = this.calculateEMA(result, period);
      } else if (req.startsWith('rolling:')) {
        const parts = req.split(':');
        const window = parseInt(parts[1]);
        const func = parts[2];
        result = this.calculateRolling(result, window, func);
      }
    });

    this.cache.set(cacheKey, result);
    return result;
  }

  private calculateEMA(data: any, period: number): any {
    const colName = `ema_${period}`;
    // Simple EMA calculation (placeholder; in real implementation, use proper EMA formula)
    data[colName] = data['close'].rolling(window=period).mean(); // pandas-js approximation
    return data;
  }

  private calculateRolling(data: any, window: number, func: string): any {
    const colName = `rolling_${func}_${window}`;
    if (func === 'mean') {
      data[colName] = data['close'].rolling(window=window).mean();
    } else if (func === 'std') {
      data[colName] = data['close'].rolling(window=window).std();
    }
    return data;
  }
}
