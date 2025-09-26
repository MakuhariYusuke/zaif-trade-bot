/**
 * Test helpers for numerical comparisons with configurable epsilon
 */

export const DEFAULT_EPS = 1e-8;

/**
 * Asserts that two numbers are close within epsilon
 */
export function expectClose(actual: number, expected: number, eps: number = DEFAULT_EPS): void {
  const diff = Math.abs(actual - expected);
  if (diff > eps) {
    throw new Error(`Expected ${actual} to be close to ${expected} (diff: ${diff}, eps: ${eps})`);
  }
}

/**
 * Asserts that two arrays are element-wise close within epsilon
 */
export function expectCloseArray(actual: number[], expected: number[], eps: number = DEFAULT_EPS): void {
  if (actual.length !== expected.length) {
    throw new Error(`Array lengths differ: ${actual.length} vs ${expected.length}`);
  }

  for (let i = 0; i < actual.length; i++) {
    expectClose(actual[i], expected[i], eps);
  }
}

/**
 * Asserts that a number is close to zero within epsilon
 */
export function expectNearZero(actual: number, eps: number = DEFAULT_EPS): void {
  expectClose(actual, 0, eps);
}

/**
 * Asserts that two numbers are close as a percentage of the expected value
 */
export function expectClosePercent(actual: number, expected: number, percent: number = 0.01): void {
  const eps = Math.abs(expected) * percent;
  expectClose(actual, expected, eps);
}
