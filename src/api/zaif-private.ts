// Renamed from private.ts
import axios from "axios";
import * as crypto from "crypto";
import { signBody, setNonceBase, createNonce, getLastNonce } from "../utils/signer";
import { ZaifApiConfig, PrivateApi, GetInfo2Response, ActiveOrder } from "../types/private";
import { loadAppConfig, restoreNonce, persistNonce } from "../utils/config";
import { logWarn, logError, logDebug } from "../utils/logger";
import { ok, err, Result } from "../utils/result";
import { BaseExchangePrivate } from "./base-private";
export function getNonceRetryTotal() { return nonceRetryTotal; }
let lastRequestNonceRetries = 0;
export function getAndResetLastRequestNonceRetries() { const v = lastRequestNonceRetries; lastRequestNonceRetries = 0; return v; }
const PRIVATE_URL = "https://api.zaif.jp/tapi";
let nonceRetryTotal = 0;
/** Builds a URL-encoded form string from the given parameters. */
function buildForm(params: Record<string, any>): string { return Object.entries(params).map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`).join("&"); }
/** Computes the SHA-256 hash of the given data. */
function sha256(data: string) { return crypto.createHash("sha256").update(data).digest("hex"); }

/**
 * Masks sensitive information in the request body for logging.
 * @param {string} body The raw request body.
 * @return {string} The masked body.
 */
function maskBody(body: string): string {
	// Mask digits
	let masked = body.replace(/[0-9]/g, "x");
	// Mask API key and secret if present (simple pattern, adjust as needed)
	masked = masked.replace(/(key=)[^&]+/gi, "$1[REDACTED]");
	masked = masked.replace(/(secret=)[^&]+/gi, "$1[REDACTED]");
	return masked;
}
/**
 * Normalizes permission-related error messages.
 * @param {string} msg Message to normalize
 * @returns {string} Normalized message
 */
function normalizePermMsg(msg: string): string {
	const lower = msg.toLowerCase();
	if (/(permission|whitelist|ip address|ip whitelist|two-?factor)/i.test(lower)) return msg + ' | Check API key rights (info/trade) and IP whitelist settings.';
	return msg;
}
/**
 * ZaifRealPrivateApi
 *
 * Client implementation for Zaif's private API endpoints. Handles signing requests,
 * nonce management, retries, error classification and basic response normalization.
 *
 * Usage notes:
 * - Construct with a ZaifApiConfig containing at minimum `key` and `secret`. Optional
 *   config fields used: `timeoutMs` (request timeout in ms) and `nonceStorePath`
 *   (path used by external nonce persistence helpers).
 * - All network requests are performed via axios POST to the PRIVATE_URL using
 *   "application/x-www-form-urlencoded" bodies that are signed with the configured secret.
 *
 * Key behaviors:
 * - Nonce handling and retries:
 *   - The internal request loop will retry on nonce-related errors up to MAX_NONCE_RETRIES
 *     (env MAX_NONCE_RETRIES, default 5) with a backoff between attempts configured by
 *     RETRY_BACKOFF_MS (env RETRY_BACKOFF_MS, default 300 ms).
 *   - On server-provided nonce suggestions/limits the implementation will attempt to
 *     advance the locally-stored nonce (via setNonceBase / restoreNonce / getLastNonce
 *     helpers if available).
 *   - The implementation increments/records per-request nonce retry counters and may
 *     restore persisted nonce state when a nonce issue is detected.
 *
 * - Signature and permission errors are detected from API messages and logged distinctly.
 * - Server time:
 *   - When the server responds with a server_time field it is stored as lastResponseServerTime
 *     (ms since epoch). If CLOCK_SKEW_TOLERANCE_MS (env) is set, large client/server clock
 *     skew will emit a warning.
 *
 * Error handling:
 * - The internal `call<T>(...)` method throws when retries are exhausted or on non-retryable
 *   errors. It classifies common API error messages (nonce, signature, permission) and logs
 *   contextual debug info (masked body, body hash, headers).
 * - `healthCheck()` returns a Result<GetInfo2Response> which encodes success or well-known
 *   normalized error codes: 'NONCE', 'SIGNATURE', 'API_ERROR', or 'NETWORK' for transport errors.
 *
 * Public methods:
 * - constructor(config: ZaifApiConfig)
 *     Create a client with credentials and optional settings.
 *
 * - async healthCheck(): Promise<Result<GetInfo2Response>>
 *     Performs a minimal signed call to 'get_info2' and returns a Result wrapper describing
 *     success or a normalized error class for easy health checks.
 *
 * - async testGetInfo2(): Promise<Result<any>>
 *     Convenience wrapper that calls `call('get_info2')` and returns a Result capturing errors
 *     instead of throwing.
 *
 * - async get_info2(): Promise<GetInfo2Response>
 *     Raw call to the 'get_info2' endpoint (throws on failure).
 *
 * - async active_orders(params?: any): Promise<ActiveOrder[]>
 *     Retrieves active orders and normalizes the API response into an array of ActiveOrder
 *     objects with consistent fields: order_id, pair, side, price, amount, timestamp.
 *
 * - async trade_history(params?: any)
 *     Retrieves and normalizes historic trades to objects with tid, order_id, side, price,
 *     amount and timestamp.
 *
 * - async trade(params: any)
 *     Executes a trade request and returns a normalized success object containing
 *     return.order_id as a string.
 *
 * - async cancel_order(params: any)
 *     Cancels an order and returns a normalized success object containing
 *     return.order_id as a string (falls back to the requested order_id when needed).
 *
 * - Convenience aliases:
 *     - getBalance() -> get_info2()
 *     - activeOrders(currency_pair?) -> active_orders(...)
 *     - cancelOrder(params) -> cancel_order(params)
 *     - tradeHistory(params) -> trade_history(params)
 *
 * Side effects and logging:
 * - Emits console warnings/errors for signature/nonce/permission issues and clock skew.
 * - Uses external helpers (setNonceBase, restoreNonce, getLastNonce, createFlexibleNonce,
 *   buildForm, signBody, sha256, maskBody) for nonce/signing, persistence and debugging.
 *
 * Environment variables:
 * - MAX_NONCE_RETRIES (default 5)
 * - RETRY_BACKOFF_MS (default 300)
 * - CLOCK_SKEW_TOLERANCE_MS (default 0 - disabled)
 *
 * Notes:
 * - The implementation assumes certain global helper functions and counters exist in scope
 *   (e.g. setNonceBase, restoreNonce, getLastNonce, nonceRetryTotal, lastRequestNonceRetries).
 * - Consumers should ensure secure handling of `config.secret` and any persisted nonce store.
 */
class ZaifRealPrivateApi extends BaseExchangePrivate implements PrivateApi {
	constructor(private config: ZaifApiConfig) { super(); }
	private lastResponseServerTime?: number;
	private async call<T>(method: string, extra: Record<string, any> = {}): Promise<T> {
		const maxRetries = Number(process.env.MAX_NONCE_RETRIES || 5);
		const baseBackoff = Number(process.env.RETRY_BACKOFF_MS || 300);
		const backoffFactor = Number(process.env.RETRY_BACKOFF_FACTOR || 1.5);
		const maxBackoff = Number(process.env.RETRY_MAX_BACKOFF_MS || 3000);
		const jitterMs = Number(process.env.RETRY_JITTER_MS || 100);
		const persistEnabled = (process.env.NONCE_PERSIST ?? '1') !== '0';
		const restoreOnError = (process.env.NONCE_RESTORE_ON_ERROR ?? '1') !== '0';
		let attempt = 0;
		let lastError: any;
		lastRequestNonceRetries = 0;
		while (attempt < maxRetries) {
			attempt++;
			const nonce = createNonce();
			const bodyObj = { method, nonce, ...extra };
			const body = buildForm(bodyObj);
			const sign = signBody(body, this.config.secret);
			const headers = { key: this.config.key, sign, "content-type": "application/x-www-form-urlencoded" } as Record<string, string>;
			try {
				const timeout = this.config.timeoutMs ?? 10000;
				const startClient = Date.now();
				const res = await axios.post(PRIVATE_URL, body, { headers, timeout });
				const data = res.data as any;
				if (data?.return?.server_time) {
					this.lastResponseServerTime = Number(data.return.server_time) * 1000;
					const tol = Number(process.env.CLOCK_SKEW_TOLERANCE_MS || 0);
					if (tol > 0) {
						const skew = Math.abs(this.lastResponseServerTime - startClient);
						if (skew > tol) logWarn(`Clock skew ~${skew}ms (>${tol})`);
					}
				}
				if (data && data.success === 0 && typeof data.error === 'string' && data.error.toLowerCase().includes('nonce')) {
					const nums = data.error.match(/[0-9]{6,}/g);
					if (nums?.length) {
						const max = Math.max(...nums.map(Number));
						setNonceBase(max + 1);
						continue;
					} else {
						// Fallback: server didn't return a concrete nonce; bump above last seen / now and retry
						const fallback = Math.max(getLastNonce() + 1, Date.now());
						setNonceBase(fallback);
						continue;
					}
				}
				// success path: persist last used nonce if configured
				if (this.config.nonceStorePath) {
					try { persistNonce(this.config.nonceStorePath, getLastNonce()); } catch {}
				}
				return data as T;
			} catch (e: any) {
				const msg = e?.response?.data?.error || e.message || 'unknown error';
				const status = e?.response?.status;
				const bodyDump = e?.response?.data;
				const debugHash = sha256(body);
				const headerNames = Object.keys(headers);
				const lowerOk = headerNames.every(h => h === h.toLowerCase());
				const logBase = { method, status, success: bodyDump?.success, return: bodyDump?.return, msg, bodyHash: debugHash, headers: headerNames, lowercaseHeaders: lowerOk };
				if (/signature/i.test(msg)) { logError('[PrivateAPI SignatureMismatch]', { ...logBase, rawBodyMasked: maskBody(body) }); lastError = e; break; }
				if (/nonce/i.test(msg)) {
					logWarn('[PrivateAPI NonceIssue]', { ...logBase, lastNonce: getLastNonce(), attemptedNonce: bodyObj.nonce });
					nonceRetryTotal++;
					lastRequestNonceRetries++;
					if (this.config.nonceStorePath && restoreOnError) {
						try {
							restoreNonce(this.config.nonceStorePath);
						} catch (restoreErr) {
							logWarn('[PrivateAPI restoreNonce failed]', { error: restoreErr, nonceStorePath: this.config.nonceStorePath });
						}
					}
					const nums = msg.match(/[0-9]{6,}/g);
					if (nums?.length) {
						const max = Math.max(...nums.map(Number));
						setNonceBase(Math.max(max + 1, Date.now()));
						if (this.config.nonceStorePath && persistEnabled) { try { persistNonce(this.config.nonceStorePath, getLastNonce()); } catch {} }
					} else {
						setNonceBase(Math.max(getLastNonce() + 1, Date.now()));
						if (this.config.nonceStorePath && persistEnabled) { try { persistNonce(this.config.nonceStorePath, getLastNonce()); } catch {} }
					}
					if (attempt < maxRetries) {
						const exp = Math.min(maxBackoff, Math.floor(baseBackoff * Math.pow(backoffFactor, attempt - 1)));
						const delay = exp + Math.floor(Math.random() * Math.max(0, jitterMs));
						logDebug(`[NONCE RETRY] attempt=${attempt} delay=${delay}ms`);
						await new Promise(r => setTimeout(r, delay));
						continue;
					}
					else { logError('[NONCE] retries exhausted'); }
				}
				if (/permission/i.test(msg)) logWarn('[PrivateAPI Permission]', { ...logBase }); else logError('[PrivateAPI Error]', logBase); lastError = e; break;
			}
		}
		throw lastError || new Error('Private API call failed');
	}
	async healthCheck(): Promise<Result<GetInfo2Response>> {
		const method = 'get_info2';
	const nonce = createNonce();
		const bodyObj = { method, nonce } as any;
		const body = buildForm(bodyObj);
		const sign = signBody(body, this.config.secret);
		const headers = { key: this.config.key, sign, "content-type": "application/x-www-form-urlencoded" } as Record<string, string>;
		try {
			const res = await axios.post(PRIVATE_URL, body, { headers, timeout: this.config.timeoutMs ?? 10000 });
			const data: GetInfo2Response = res.data;
			if (data?.success === 1) return ok(data);
			const msg = data?.error || 'unknown';
			return err(/nonce/i.test(msg) ? 'NONCE' : /signature/i.test(msg) ? 'SIGNATURE' : 'API_ERROR', normalizePermMsg(msg));
		} catch (e: any) { return err('NETWORK', e?.message || 'error', e); }
	}
	async testGetInfo2(): Promise<Result<GetInfo2Response>> {
		 try {
			 const r = await this.call<GetInfo2Response>('get_info2');
			 return ok(r);
		 } catch (e: any) {
			 return err('TEST_GET_INFO2_FAIL', e?.message || 'error', e);
		 }
	}
	async get_info2() { return this.call<GetInfo2Response>('get_info2'); }
	async active_orders(params?: any): Promise<ActiveOrder[]> {
		const raw: any = await this.call('active_orders', params || {});
		const arr: ActiveOrder[] = [];
		if (raw?.return) {
			for (const [oid, o] of Object.entries<any>(raw.return))
				arr.push({ order_id: oid, pair: o.currency_pair, side: o.action, price: o.price, amount: o.amount, timestamp: o.timestamp });
		}
		return arr;
	}
	async trade_history(params?: any): Promise<{ tid: any; order_id: string; side: any; price: any; amount: any; timestamp: any; }[]> {
		const raw: any = await this.call('trade_history', params || {});
		return (raw?.return || []).map((t: any) => ({ tid: t.tid, order_id: t.order_id?.toString(), side: t.trade_type, price: t.price, amount: t.amount, timestamp: t.date }));
	}
	async trade(params: any) {
		const r: any = await this.call('trade', params);
		const id = r?.return?.order_id ?? r?.order_id ?? 0;
		return { success: 1 as const, return: { order_id: String(id) } };
	}
	async cancel_order(params: any) {
		const r: any = await this.call('cancel_order', params);
		const id = r?.return?.order_id ?? params.order_id;
		return { success: 1 as const, return: { order_id: String(id) } };
	}
	async getBalance() { return this.get_info2(); }
	async activeOrders(currency_pair?: string) { return this.active_orders(currency_pair ? { currency_pair } : {}) }
	async cancelOrder(params: any) { return this.cancel_order(params); }
	async tradeHistory(params: any) { return this.trade_history(params); }
}
export function createPrivateReal(): PrivateApi {
	const key = process.env.ZAIF_API_KEY || '';
	const secret = process.env.ZAIF_API_SECRET || '';
	const appCfg = loadAppConfig();
	return new ZaifRealPrivateApi({ key, secret, nonceStorePath: appCfg.nonceStorePath });
}
