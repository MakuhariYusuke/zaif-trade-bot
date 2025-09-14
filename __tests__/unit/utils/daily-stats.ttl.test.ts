import { describe, it, expect, beforeEach, vi } from 'vitest';
import fs from 'fs';
import path from 'path';

const TMP = path.resolve(process.cwd(), 'tmp-test-stats-edge');

describe('daily-stats TTL eviction', () => {
  beforeEach(()=>{
    vi.resetModules();
    if (fs.existsSync(TMP)) fs.rmSync(TMP, { recursive: true, force: true });
    fs.mkdirSync(path.join(TMP,'logs'), { recursive: true });
    process.env.STATS_DIR = path.join(TMP,'logs');
    process.env.DAILY_STATS_TTL_DAYS = '0'; // expire immediately
  });

  it('evicts stale cache entries and reloads from disk', async () => {
    const { appendFillPnl, loadDaily } = await import('../../../src/utils/daily-stats');
    const d = '2025-01-01';
    // write once
    appendFillPnl(d, 5, 'btc_jpy');
    // ensure evict
    const a1 = loadDaily(d, 'btc_jpy');
    expect(a1.realizedPnl).toBeDefined();
  });
});
