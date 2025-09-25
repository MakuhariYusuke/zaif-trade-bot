import { logInfo } from '../utils/logger';

export type CircuitState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

export interface CircuitBreakerOptions {
  windowSize?: number;
  failureThreshold?: number; // 0..1
  maxConsecutiveFailures?: number;
  latencyThreshold?: number; // ms
  halfOpenTrial?: number;
  cooldownMs?: number; // ms to move OPEN -> HALF_OPEN
}

interface Sample { ok: boolean; latency: number; ts: number }

/**
 * A Circuit Breaker implementation to wrap calls to external services.
 * States:
 * - CLOSED: normal operation, all calls allowed. Monitors success/failure rate and latency.
 * - OPEN: all calls blocked. After cooldown period, transitions to HALF_OPEN.
 * - HALF_OPEN: allows limited trial calls. If enough successes, transitions to CLOSED. If any failure, transitions to OPEN.
 * Transitions:
 * - CLOSED -> OPEN: if failure rate > threshold, or consecutive failures > max, or latency > threshold
 * - OPEN -> HALF_OPEN: after cooldown period
 * - HALF_OPEN -> CLOSED: if enough successes in trials
 * - HALF_OPEN -> OPEN: if any failure during trials
 * Usage:
 * const cb = new CircuitBreaker({ failureThreshold: 0.5, maxConsecutiveFailures: 5, latencyThreshold: 30000 });
 * if (cb.allowRequest()) {
 *   try {
 *     const start = Date.now();
 *    await callExternalService();
 *   cb.recordSuccess(Date.now() - start);
 *  } catch (e) {
 *    cb.recordFailure(e);
 *  }
 * } else {
 *   // short-circuit
 * }
 * 
 */
export class CircuitBreaker {
  private state: CircuitState = 'CLOSED';
  private samples: Sample[] = [];
  private consecutiveFailures = 0;
  private lastStateChange = Date.now();
  private halfOpenAttempts = 0;
  private halfOpenSuccesses = 0;

  private readonly windowSize: number;
  private readonly failureThreshold: number;
  private readonly maxConsecutiveFailures: number;
  private readonly latencyThreshold: number;
  private readonly halfOpenTrial: number;
  private readonly cooldownMs: number;

  constructor(opts: CircuitBreakerOptions = {}) {
    this.windowSize = Math.max(10, opts.windowSize ?? Number(process.env.CB_WINDOW_SIZE ?? 50));
    this.failureThreshold = opts.failureThreshold ?? Number(process.env.CB_FAILURE_THRESHOLD ?? 0.5);
    this.maxConsecutiveFailures = opts.maxConsecutiveFailures ?? Number(process.env.CB_MAX_CONSEC_FAIL ?? 5);
    this.latencyThreshold = opts.latencyThreshold ?? Number(process.env.CB_LATENCY_THRESHOLD_MS ?? 30000);
    this.halfOpenTrial = opts.halfOpenTrial ?? Number(process.env.CB_HALF_OPEN_TRIAL ?? 5);
    this.cooldownMs = opts.cooldownMs ?? Number(process.env.CB_COOLDOWN_MS ?? 60000);
  }

  /**
   * Get the current circuit state.
   * @returns Current circuit state (CLOSED, OPEN, HALF_OPEN)
   */
  getState(): CircuitState { return this.state; }

  /**
   * Check if a request is allowed under the current circuit state.
   * @returns Whether a request is allowed under the current circuit state.
   */
  allowRequest(): boolean {
    if (this.state === 'OPEN') {
      const now = Date.now();
      if (now - this.lastStateChange >= this.cooldownMs) {
        this.transition('HALF_OPEN', 'cooldown');
        return true; // allow limited trials
      }
      return false;
    }
    if (this.state === 'HALF_OPEN') {
      // allow up to halfOpenTrial requests
      return this.halfOpenAttempts < this.halfOpenTrial;
    }
    return true; // CLOSED
  }

  /**
   * Record a successful call.
   * @param {number} latencyMs Latency of the successful call in milliseconds.
   */
  recordSuccess(latencyMs: number) {
    this.pushSample({ ok: true, latency: latencyMs, ts: Date.now() });
    this.consecutiveFailures = 0;
    if (this.state === 'HALF_OPEN') {
      this.halfOpenAttempts++;
      this.halfOpenSuccesses++;
      const okRate = this.halfOpenSuccesses / this.halfOpenAttempts;
      if (this.halfOpenAttempts >= this.halfOpenTrial && okRate >= (1 - this.failureThreshold)) {
        this.transition('CLOSED', 'half_open_success');
        this.halfOpenAttempts = 0;
        this.halfOpenSuccesses = 0;
      }
    } else if (this.state === 'OPEN') {
      // no-op; will be gated by allowRequest which can move to HALF_OPEN
    } else {
      // CLOSED: evaluate latency-triggered open
      if (latencyMs > this.latencyThreshold) {
        this.transition('OPEN', 'latency');
      }
    }
    this.trimWindow();
    this.evaluateClosedWindow();
  }

  /**
   * Record a failed call.
   * @param {any} cause Optional error or cause of the failure.
   */
  recordFailure(cause?: any) {
    this.pushSample({ ok: false, latency: 0, ts: Date.now() });
    this.consecutiveFailures++;
    if (this.state === 'HALF_OPEN') {
      // immediate open if failure during trial
      this.transition('OPEN', 'half_open_failure');
      logInfo('CIRCUIT_BREAKER', `Circuit opened due to failure in HALF_OPEN state. Cause: ${cause?.message || cause || 'unknown'}`);
      this.halfOpenAttempts = 0;
    } else {
      // CLOSED
      const shouldOpen = this.consecutiveFailures > this.maxConsecutiveFailures || this.failureRate() > this.failureThreshold;
      if (shouldOpen) {
        const reason = this.consecutiveFailures > this.maxConsecutiveFailures ? 'consecutive_fail' : 'failure_rate';
        this.transition('OPEN', reason);
        logInfo('CIRCUIT_BREAKER', `Circuit opened due to ${reason}. Consecutive failures: ${this.consecutiveFailures}, Failure rate: ${(this.failureRate() * 100).toFixed(1)}%. Cause: ${cause?.message || cause || 'unknown'}`);
      }
    }
    this.trimWindow();
  }

  /**
   * Record a failed call.
   * @param {any} cause Optional error or cause of the failure.
   */
  private evaluateClosedWindow() {
    if (this.state !== 'CLOSED') return;
    // open on window stats
    if (this.failureRate() > this.failureThreshold) {
      this.transition('OPEN', 'failure_rate');
    } else if (this.medianLatency() > this.latencyThreshold) {
      this.transition('OPEN', 'latency');
    } else if (this.consecutiveFailures > this.maxConsecutiveFailures) {
      this.transition('OPEN', 'consecutive_fail');
    }
  }

  /**
   * Transition to a new circuit state.
   * @param {CircuitState} next The next circuit state to transition to.
   * @param {string} reason The reason for the state transition.
   * @returns {void}
   */
  private transition(next: CircuitState, reason: string) {
    const prev = this.state;
    if (prev === next) return;
    logInfo('CIRCUIT_BREAKER', `State transition: ${prev} -> ${next} (reason: ${reason})`);
    this.state = next;
    this.lastStateChange = Date.now();
    if (next === 'HALF_OPEN') {
      this.halfOpenAttempts = 0;
      this.halfOpenSuccesses = 0;
    } else {
      this.halfOpenAttempts = 0;
      this.halfOpenSuccesses = 0;
    }
    if (prev === 'HALF_OPEN' && next === 'CLOSED') {
      // reset error memory after successful trial
      this.samples = [];
      this.consecutiveFailures = 0;
    }
    logInfo('CB/INFO', { from: prev, to: next, reason });
  }

  private pushSample(s: Sample) {
    this.samples.push(s);
    this.trimWindow();
  }

  private trimWindow() {
    if (this.samples.length > this.windowSize) this.samples.splice(0, this.samples.length - this.windowSize);
  }

  private successRate(): number {
    if (this.samples.length === 0) return 1;
    const ok = this.samples.reduce((s, x) => s + (x.ok ? 1 : 0), 0);
    return ok / this.samples.length;
  }

  private failureRate(): number { return 1 - this.successRate(); }

  private medianLatency(): number {
    const lat = this.samples.filter(s => s.ok).map(s => s.latency).sort((a,b) => a-b);
    if (lat.length === 0) return 0;
    const mid = Math.floor(lat.length / 2);
    return lat.length % 2 === 0 ? (lat[mid-1] + lat[mid]) / 2 : lat[mid];
  }
}

// Provide a global singleton for easy wiring
let globalCb: CircuitBreaker | null = null;
export function getCircuitBreaker(): CircuitBreaker {
  if (!globalCb) globalCb = new CircuitBreaker();
  return globalCb;
}
export function setCircuitBreaker(cb: CircuitBreaker | null) { globalCb = cb; }
