import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

const TMP = path.resolve(process.cwd(), 'tmp-test-features');

function today(){ return new Date().toISOString().slice(0,10); }

describe('features-logger interval flush', () => {
  const pair = 'btc_jpy';
  const date = today();
  const root = path.join(TMP, 'logs');
  beforeEach(()=>{
    vi.resetModules();
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
  });

  it('flushes by interval when enabled', async () => {
    // enable interval and not test mode
    process.env.TEST_MODE = '0';
    delete process.env.VITEST_WORKER_ID;
    process.env.FEATURES_FLUSH_INTERVAL_MS = '50';
    process.env.FEATURES_LOG_DIR = root;
    const mod = await import('../../../src/utils/features-logger');
    mod.logFeatureSample({ ts: Date.now(), pair, side:'bid', price:1, qty:1 } as any);
    await new Promise(res=>setTimeout(res, 120));
    const csvPath = path.join(root, 'features', pair, `features-${date}.csv`);
    // header is not written until first flush when not in test mode
    expect(fs.existsSync(csvPath)).toBe(true);
  });
});
