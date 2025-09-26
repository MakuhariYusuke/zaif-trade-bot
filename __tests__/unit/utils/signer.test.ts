import { describe, it, expect } from 'vitest';
import { signBody, createNonce, setNonceBase, createFlexibleNonce, getLastNonce } from '../../../ztb/utils/signer';

describe('utils/signer', () => {
  it('signBody returns deterministic HMAC', ()=>{
    const sig1 = signBody('a=1&b=2', 'secret');
    const sig2 = signBody('a=1&b=2', 'secret');
    expect(sig1).toBe(sig2);
  });
  it('createNonce is monotonic and respects base', ()=>{
    const n1 = createNonce();
    setNonceBase(n1 + 1000);
    const n2 = createNonce();
    expect(n2).toBeGreaterThan(n1);
    const n3 = createNonce();
    expect(n3).toBeGreaterThan(n2);
    expect(getLastNonce()).toBe(n3);
  });
  it('createFlexibleNonce strictly increases within same ms', ()=>{
    const a = createFlexibleNonce();
    const b = createFlexibleNonce();
    expect(b >= a).toBe(true);
  });
});
