import { vi, expect } from 'vitest';

export function setupJsonLogs() {
  process.env.TEST_MODE = '0';
  process.env.LOG_JSON = '1';
  process.env.LOG_LEVEL = 'DEBUG';
}

export function captureLogs() {
  const logs: { level: 'WARN'|'ERROR'|'INFO'|'DEBUG'; category?: string; message: string; data?: any }[] = [];
  const origLog = console.log;
  const origWarn = console.warn;
  const origErr = console.error;
  vi.spyOn(console, 'log').mockImplementation((line: any) => {
    try { const e = JSON.parse(String(line)); if (e && e.level && e.message) logs.push({ level: e.level, category: e.category, message: e.message, data: e.data?.[0] }); } catch {}
    origLog(line);
  });
  vi.spyOn(console, 'warn').mockImplementation((line: any) => {
    try { const e = JSON.parse(String(line)); if (e && e.level && e.message) logs.push({ level: e.level, category: e.category, message: e.message, data: e.data?.[0] }); } catch {}
    origWarn(line);
  });
  vi.spyOn(console, 'error').mockImplementation((line: any) => {
    try { const e = JSON.parse(String(line)); if (e && e.level && e.message) logs.push({ level: e.level, category: e.category, message: e.message, data: e.data?.[0] }); } catch {}
    origErr(line);
  });
  return logs;
}

export function expectJsonLog(logs: any[], category: string, level: 'WARN'|'ERROR'|'INFO'|'DEBUG', msgIncludes?: string, metaKeys?: string[]) {
  const hit = logs.find(l => l.category === category && l.level === level && (!msgIncludes || String(l.message).includes(msgIncludes)));
  expect(hit, `expected log ${category}/${level} containing '${msgIncludes}'`).toBeTruthy();
  if (hit && metaKeys && metaKeys.length) {
    for (const k of metaKeys) expect(hit.data, `missing meta ${k}`).toHaveProperty(k);
  }
  return hit;
}
