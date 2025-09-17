import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { logTrace, logDebug, logInfo, logWarn, logError, logFatal } from '../../../src/utils/logger';

type LogLevel = 'TRACE' | 'DEBUG' | 'INFO' | 'WARN' | 'ERROR' | 'FATAL';

describe('utils/logger', () => {
    const envBk = { ...process.env };
    let logs: string[] = [];
    let warns: string[] = [];
    let errs: string[] = [];

    beforeEach(() => {
        process.env = { ...envBk };
        process.env.TEST_MODE = '0';
        logs = [];
        warns = [];
        errs = [];
        vi.spyOn(console, 'log').mockImplementation((...a: any[]) => { logs.push(a.join(' ')); });
        vi.spyOn(console, 'warn').mockImplementation((...a: any[]) => { warns.push(a.join(' ')); });
        vi.spyOn(console, 'error').mockImplementation((...a: any[]) => { errs.push(a.join(' ')); });
    // no-op: logger context APIs are internal now
    });

    afterEach(() => {
        vi.restoreAllMocks();
        process.env = { ...envBk };
    });

    const levels: LogLevel[] = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'];
    const logFunctions = {
        TRACE: logTrace,
        DEBUG: logDebug,
        INFO: logInfo,
        WARN: logWarn,
        ERROR: logError,
        FATAL: logFatal,
    };
    const outputCounters = {
        TRACE: () => logs.length,
        DEBUG: () => logs.length,
        INFO: () => logs.length,
        WARN: () => warns.length,
        ERROR: () => errs.length,
        FATAL: () => errs.length,
    };

    levels.forEach((level, levelIndex) => {
        it(`respects LOG_LEVEL='${level}' threshold`, () => {
            process.env.LOG_LEVEL = level;
            // For each candidate level, emit only that level and verify counters
            levels.forEach((l, i) => {
                // reset spies output per sub-check
                logs = []; warns = []; errs = [];
                logFunctions[l](l.toLowerCase());
                const count = outputCounters[l]();
                if (i >= levelIndex) {
                    expect(count, `${l} should be logged when LOG_LEVEL is ${level}`).toBeGreaterThan(0);
                } else {
                    expect(count, `${l} should NOT be logged when LOG_LEVEL is ${level}`).toBe(0);
                }
            });
        });
    });


    it('outputs JSON when LOG_JSON=1 and includes meta', () => {
        process.env.LOG_LEVEL = 'INFO';
        process.env.LOG_JSON = '1';
        logInfo('hello', { x: 1 });
        // Expect logInfo to output to logs array
        expect(logs.length).toBeGreaterThan(0);
        const obj = JSON.parse(logs[0]);
        expect(obj.level).toBe('INFO');
        expect(obj.message).toBe('hello');
        expect(Array.isArray(obj.data)).toBe(true);
        expect(obj.data[0].x).toBe(1);
    });

    it('suppresses non-error logs in TEST_MODE', () => {
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
