import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { logInfo, logWarn, logError, setLoggerContext, clearLoggerContext } from '../../../src/utils/logger';

describe('utils/logger', () => {
  const envBk = { ...process.env };
  const orig = { log: console.log, warn: console.warn, error: console.error };
  let logs: any[] = []; let warns: any[] = []; let errs: any[] = [];
  beforeEach(()=>{
    process.env = { ...envBk };
    logs = []; warns = []; errs = [];
  console.log = ((...a: any[]) => { logs.push(a.join(' ')); }) as any;
  console.warn = ((...a: any[]) => { warns.push(a.join(' ')); }) as any;
  console.error = ((...a: any[]) => { errs.push(a.join(' ')); }) as any;
    clearLoggerContext();
  });
  afterEach(()=>{ console.log = orig.log; console.warn = orig.warn; console.error = orig.error; process.env = { ...envBk }; });

  it('respects LOG_LEVEL threshold', ()=>{
    process.env.LOG_LEVEL = 'WARN';
    logInfo('nope');
    logWarn('warn');
    expect(logs.length).toBe(0);
    expect(warns.length).toBeGreaterThan(0);
  });

  it('outputs JSON when LOG_JSON=1 and includes context', ()=>{
    process.env.LOG_LEVEL = 'INFO';
    process.env.LOG_JSON = '1';
    setLoggerContext({ reqId: 'abc' });
    logInfo('hello', { x: 1 });
    const line = errs[0] || warns[0] || logs[0];
    const obj = JSON.parse(line);
    expect(obj.level).toBe('INFO');
    expect(obj.message).toBe('hello');
    expect(obj.reqId).toBe('abc');
  });

  it('suppresses non-error logs in TEST_MODE', ()=>{
    process.env.LOG_LEVEL = 'DEBUG';
    process.env.TEST_MODE = '1';
    logInfo('info');
    logWarn('warn');
    logError('err');
    expect(logs.length).toBe(0);
    expect(warns.length).toBe(0);
    expect(errs.length).toBeGreaterThan(0);
  });
});
