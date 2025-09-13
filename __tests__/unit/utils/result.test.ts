import { describe, it, expect } from 'vitest';
import { ok, err } from '../../../src/utils/result';

describe('utils/result', () => {
  it('ok returns expected shape', ()=>{
    const r = ok(123);
    expect(r.ok).toBe(true);
    expect(r.value).toBe(123);
  });
  it('err returns expected shape', ()=>{
    const e = err('E_TEST', 'failed', { cause: 1 });
    expect(e.ok).toBe(false);
    expect(e.error.code).toBe('E_TEST');
    expect(e.error.message).toBe('failed');
  });
});
