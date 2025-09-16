import { describe, it, expect, beforeEach, vi } from 'vitest';
import { logger, setLoggerContext, addLoggerRedactFields } from '../../../src/utils/logger';

describe('logger JSON required meta and redaction', () => {
  beforeEach(() => {
    process.env.LOG_JSON = '1';
    process.env.TEST_MODE = '1';
    process.env.LOG_LEVEL = 'INFO';
  });
  it('emits JSON with redacted fields', () => {
    const spy = vi.spyOn(console, 'log').mockImplementation(()=>undefined as any);
    try {
      addLoggerRedactFields(['apiKey','secret']);
      logger.log?.('INFO','API','request',{ requestId: 'abc', pair: 'btc_jpy', side: 'buy', amount: 0.1, price: 100, retries: 1, headers: { Authorization: 'token' }, apiKey: 'xxx', secret: 'yyy' });
      expect(spy).toHaveBeenCalled();
      const arg = spy.mock.calls[0][0];
      const obj = JSON.parse(arg);
      expect(obj.category).toBe('API');
      expect(obj.data[0].requestId).toBe('abc');
      expect(obj.data[0].headers.Authorization).toBe('***');
      expect(obj.data[0].apiKey).toBe('***');
      expect(obj.data[0].secret).toBe('***');
    } finally { spy.mockRestore(); }
  });
});
