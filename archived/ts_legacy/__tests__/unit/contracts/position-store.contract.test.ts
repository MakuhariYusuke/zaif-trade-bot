import { describe, test, expect, beforeEach, afterEach } from 'vitest';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import resetTestState, { registerTempDir } from '../../../ztb/utils/test-reset';
import { CorePositionStore } from '../../../ztb/core/position-store';

import type { PositionStore, PositionState } from '@contracts';

function uniqTmpDir(suffix: string){
  const d = path.resolve(process.cwd(), 'tmp-test-position-store', `${Date.now()}-${process.pid}-${randomUUID()}-${suffix}`);
  fs.mkdirSync(d, { recursive: true });
  registerTempDir(d);
  return d;
}

function withEnv<T>(env: Record<string,string>, fn: ()=>T): T {
  const backup: Record<string, string|undefined> = {};
  for (const k of Object.keys(env)) { backup[k] = process.env[k]; (process.env as any)[k] = env[k]; }
  try { return fn(); } finally { for (const [k,v] of Object.entries(backup)) { if (v === undefined) delete (process.env as any)[k]; else (process.env as any)[k] = v; } }
}

function makeCoreFactory(): () => PositionStore { return () => new CorePositionStore(); }

describe('PositionStore contract: core & services-adapter', () => {
  const cases: [string, () => PositionStore][] = [ ['core', makeCoreFactory()] ];

  beforeEach(() => {
    resetTestState({ envSnapshot: process.env, restoreEnv: true });
  });

  afterEach(() => {
    resetTestState({ envSnapshot: process.env, restoreEnv: true });
  });

  test.each(cases)('%s: CRUD & isolation', async (_name, make) => {
    const dir = uniqTmpDir(_name);
    const file = path.join(dir, 'pos.store.json');
    withEnv({ POSITION_STORE_DIR: dir, POSITION_STORE_FILE: file }, () => {});
    const store = make();

    const pair = 'btc_jpy';
    // load → default
    const init = await store.load(pair);
    expect(init).toBeTruthy();
    expect(init.pair).toBe(pair);
    expect(init.qty).toBe(0);

    // save → reload
    const s1: PositionState = { ...init, qty: 0.1, avgPrice: 7000000, dcaCount: 0, openOrderIds: [] };
    await store.save(pair, s1);
    const got1 = await store.load(pair);
    expect(got1.qty).toBeCloseTo(0.1);
    expect(got1.avgPrice).toBe(7000000);

    // update partial
    const got2 = await store.update(pair, { qty: 0.2 });
    expect(got2.qty).toBeCloseTo(0.2);
    const got3 = await store.load(pair);
    expect(got3.qty).toBeCloseTo(0.2);

    // clear
    await store.clear?.(pair);
    const got4 = await store.load(pair);
    expect(got4.qty).toBe(0);
  });
});
