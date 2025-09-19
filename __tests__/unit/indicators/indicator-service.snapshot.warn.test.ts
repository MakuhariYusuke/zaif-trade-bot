import { describe, it, expect, beforeEach } from 'vitest';
import { IndicatorService } from '../../../src/adapters/indicator-service';

describe('indicator-service WARN once and aliases', () => {
  const logs: { level: string; cat: string; msg: string; data?: any }[] = [];
  const raw: string[] = [];
  const origConsole = { log: console.log, error: console.error, warn: console.warn };
  beforeEach(()=>{
    // capture logs by monkey-patching BaseService.clog via console
    logs.length = 0;
  });

  it('emits missing-volume WARN once across multiple updates and sets aliases', () => {
    process.env.LOG_LEVEL = 'INFO';
    process.env.TEST_MODE = '0';
    // monkey patch console used by BaseService.clog
    (console as any).log = (...args: any[]) => {
      raw.push(args.map(a=> typeof a==='string'?a:JSON.stringify(a)).join(' '));
      const first = args[0]; const meta = args[1];
      if (typeof first === 'string' && first.startsWith('[WARN][IND] missing') && meta && typeof meta==='object') {
        logs.push({ level: 'WARN', cat: 'IND', msg: 'missing', data: meta });
      }
    };
    (console as any).warn = (...args: any[]) => {
      raw.push(args.map(a=> typeof a==='string'?a:JSON.stringify(a)).join(' '));
      const first = args[0]; const meta = args[1];
      if (typeof first === 'string' && first.startsWith('[WARN][IND] missing') && meta && typeof meta==='object') {
        logs.push({ level: 'WARN', cat: 'IND', msg: 'missing', data: meta });
      }
    };
    try {
      const svc = new IndicatorService({ obvEnabled: true });
      const now = Date.now();
      for (let i = 0; i < 5; i++) svc.update(now + i*1000, 100 + i);
      // WARN should contain both MFI and OBV once each
      const mfiWarns = logs.filter(l => l.msg === 'missing' && l.data?.indicator === 'mfi');
      const obvWarns = logs.filter(l => l.msg === 'missing' && l.data?.indicator === 'obv');
      if (mfiWarns.length !== 1 || obvWarns.length !== 1) {
        console.error('[DIAG][indicator-service] raw logs=', raw);
      }
      expect(mfiWarns.length).toBe(1);
      expect(obvWarns.length).toBe(1);
      // last snapshot has aliases
      const snap = svc.update(now + 6000, 106);
      expect(snap).toHaveProperty('rsi14');
      expect(snap).toHaveProperty('atr14');
    } finally {
      console.log = origConsole.log; console.error = origConsole.error; console.warn = origConsole.warn;
      // reset once flags for isolation
      const g: any = global as any; delete g.__ind_warn_mfi_missing; delete g.__ind_warn_obv_missing;
    }
  });
});

function normalize(args: any[]): [string,string,string,any] { return ['INFO','GEN','', undefined]; }
