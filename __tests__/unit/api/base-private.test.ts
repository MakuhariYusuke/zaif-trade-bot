import { describe, expect, test } from 'vitest';
import { BaseExchangePrivate } from '../../../src/api/base-private';

class TestPriv extends BaseExchangePrivate {
  constructor(public key: string, public secret: string){ super(); }
  async callSample(url: string, body: string) {
    const nonce = this.nextNonce();
    const sig = this.hmacSha256(nonce + url + body, this.secret);
    return { nonce, sig };
  }
}

describe('BaseExchangePrivate', () => {
  test('nonce increases and signature changes', async () => {
    const c = new TestPriv('k','s');
    const r1 = await c.callSample('https://example.com','a=1');
    const r2 = await c.callSample('https://example.com','a=1');
    expect(Number(r2.nonce)).toBeGreaterThan(Number(r1.nonce));
    expect(r1.sig).not.toBe(r2.sig);
  });
});
