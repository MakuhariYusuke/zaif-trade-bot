import { describe, it, expect } from 'vitest';
import { buildSlackSummary } from '../../../src/tools/slack-summary';

describe('slack-summary build', () => {
  it('includes coverage, commit sha, phase counts when available', () => {
    process.env.GITHUB_SHA = 'abcdef1234567890';
    const { payload, text } = buildSlackSummary();
    expect(payload.text).toContain('Trade Summary:');
    // commit sha (shortened)
    expect(payload.meta.commit).toBe('abcdef1');
    // metrics object present
    expect(typeof payload.meta.metrics).toBe('object');
    // text contains executed / fails
    expect(text).toMatch(/executed:/);
  });
});
