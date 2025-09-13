import axios from "axios";
import { signBody } from "../utils/signer";
import { setNonceBase, createFlexibleNonce, getLastNonce } from "../utils/signer";
import { ZaifApiConfig, PrivateApi, GetInfo2Response, ActiveOrder, TradeHistoryRecord, TradeResult, CancelResult } from "../types/private";
import { createPrivateMock } from "./private-mock";
import { restoreNonce } from "../utils/config";
import { ok, err, Result } from "../utils/result";
import { logWarn, logError } from "../utils/logger";

export function getNonceRetryTotal() { return nonceRetryTotal; }
let lastRequestNonceRetries = 0;
export function getAndResetLastRequestNonceRetries() { const v = lastRequestNonceRetries; lastRequestNonceRetries = 0; return v; }

const PRIVATE_URL = "https://api.zaif.jp/tapi"; // POST endpoint
let nonceRetryTotal = 0;

function buildForm(params: Record<string, any>): string {
  return Object.entries(params)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
}

function sha256(data: string) {
  const crypto = require("crypto");
  return crypto.createHash("sha256").update(data).digest("hex");
}

function maskBody(body: string) {
  // mask numbers except decimal dots for correlation while hiding raw nonce magnitude
  return body.replace(/[0-9]/g, "x");
}

function normalizePermMsg(msg: string) {
  const lower = msg.toLowerCase();
  if (/(permission|whitelist|ip address|ip whitelist|two-?factor)/i.test(lower)) {
    return msg + ' | Check API key rights (info/trade) and IP whitelist settings on the exchange dashboard.';
  }
  return msg;
}

class ZaifRealPrivateApi implements PrivateApi {
  constructor(private config: ZaifApiConfig) { }

  private lastResponseServerTime?: number;
  private async call<T>(method: string, extraParams: Record<string, any> = {}): Promise<T> {
    const maxRetries = Number(process.env.MAX_NONCE_RETRIES || 5);
    const baseBackoff = Number(process.env.RETRY_BACKOFF_MS || 300);
    const backoffFactor = Number(process.env.RETRY_BACKOFF_FACTOR || 1.5);
    const maxBackoff = Number(process.env.RETRY_MAX_BACKOFF_MS || 3000);
    const jitterMs = Number(process.env.RETRY_JITTER_MS || 100);
    const restoreOnError = (process.env.NONCE_RESTORE_ON_ERROR ?? '1') !== '0';
    let attempt = 0;
    let lastError: any;
    lastRequestNonceRetries = 0;
    while (attempt < maxRetries) {
      attempt++;
      const nonce = createFlexibleNonce();
      const bodyObj = { method, nonce, ...extraParams };
      const body = buildForm(bodyObj);
      const sign = signBody(body, this.config.secret);

      const headers: Record<string, string> = {
        "key": this.config.key,
        "sign": sign,
        "content-type": "application/x-www-form-urlencoded",
      };

      try {
        const timeout = this.config.timeoutMs ?? 10_000;
        const startClient = Date.now();
        const res = await axios.post(PRIVATE_URL, body, { headers, timeout });
        const data = res.data as any;
        // capture server_time if returned
        if (data?.return?.server_time) {
          this.lastResponseServerTime = Number(data.return.server_time) * 1000;
          const tolerance = Number(process.env.CLOCK_SKEW_TOLERANCE_MS || 0);
          if (tolerance > 0) {
            const skew = Math.abs(this.lastResponseServerTime - startClient);
            if (skew > tolerance) { logWarn(`Clock skew detected ~${skew}ms (> ${tolerance})`); }
          }
        }
        if (data && data.success === 0 && typeof data.error === "string" && data.error.toLowerCase().includes("nonce")) {
          // Try to extract a suggested nonce number from error text
          const nums = data.error.match(/[0-9]{6,}/g);
          if (nums && nums.length) {
            const max = Math.max(...nums.map(Number));
            setNonceBase(max + 1);
            continue; // retry once
          }
        }
        return data as T;
      } catch (e: any) {
        const msg = e?.response?.data?.error || e.message || "unknown error";
        const status = e?.response?.status;
        const bodyDump = e?.response?.data;
        const debugHash = sha256(body);
        const headerNames = Object.keys(headers);
        const headerPresence: Record<string, boolean> = {};
        headerNames.forEach(h => headerPresence[h] = true);
        const lowerOk = headerNames.every(h => h === h.toLowerCase());
        const logBase = { method, status, success: bodyDump?.success, return: bodyDump?.return, msg, bodyHash: debugHash, headers: headerNames, lowercaseHeaders: lowerOk };
        if (/signature/i.test(msg)) {
          logError("[PrivateAPI SignatureMismatch]", { ...logBase, rawBodyMasked: maskBody(body), contentType: headers["content-type"] });
          lastError = e; // do not retry signature mismatch blindly
          break;
        }
        if (/nonce/i.test(msg)) {
          logWarn("[PrivateAPI NonceIssue]", { ...logBase, lastNonce: getLastNonce(), attemptedNonce: bodyObj.nonce });
          nonceRetryTotal++;
          lastRequestNonceRetries++;
          // reload persisted nonce if path provided
          if (this.config.nonceStorePath && restoreOnError) restoreNonce(this.config.nonceStorePath);
          const nums = msg.match(/[0-9]{6,}/g);
          if (nums && nums.length) {
            const max = Math.max(...nums.map(Number));
            setNonceBase(Math.max(max + 1, Date.now()));
          } else {
            setNonceBase(Math.max(getLastNonce() + 1, Date.now()));
          }
          if (attempt < maxRetries) {
            const exp = Math.min(maxBackoff, Math.floor(baseBackoff * Math.pow(backoffFactor, attempt - 1)));
            const delay = exp + Math.floor(Math.random() * Math.max(0, jitterMs));
            await new Promise(r => setTimeout(r, delay));
            continue;
          } else { logError("[NONCE] out of range after retries; consider setting ZAIF_STARTING_NONCE >= server last."); }
        }
        if (/permission/i.test(msg)) { logWarn("[PrivateAPI Permission]", { ...logBase, hint: "Check API key Info/Trade permissions and IP whitelist." }); }
        else { logError("[PrivateAPI Error]", logBase); }
        lastError = e;
        break;
      }
    }
    throw lastError || new Error("Private API call failed");
  }

  /** Simple health check using get_info2 (Result) */
  async healthCheck(): Promise<Result<GetInfo2Response>> {
    const method = "get_info2";
    const nonce = createFlexibleNonce();
    const bodyObj = { method, nonce } as any;
    const body = buildForm(bodyObj);
    const sign = signBody(body, this.config.secret);
    const headers: Record<string, string> = {
      "key": this.config.key,
      "sign": sign,
      "content-type": "application/x-www-form-urlencoded",
    };
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
      const r = await this.call<GetInfo2Response>("get_info2");
      return ok(r);
    } catch (e: any) {
      return err('TEST_GET_INFO2_FAIL', e?.message || 'error', e);
    }
  }

  // Interface compatibility wrappers
  async get_info2(): Promise<GetInfo2Response> { return this.call<GetInfo2Response>("get_info2"); }
  async active_orders(params?: any): Promise<ActiveOrder[]> {
    const raw: any = await this.call("active_orders", params || {});
    const arr: ActiveOrder[] = [];
    if (raw?.return) {
      for (const [oid, o] of Object.entries<any>(raw.return)) {
        arr.push({ order_id: oid, pair: o.currency_pair, side: o.action, price: o.price, amount: o.amount, timestamp: o.timestamp });
      }
    }
    return arr;
  }
  async trade_history(params?: any): Promise<TradeHistoryRecord[]> {
    const raw: any = await this.call("trade_history", params || {});
    return (raw?.return || []).map((t: any) => ({ tid: t.tid, order_id: t.order_id?.toString(), side: t.trade_type, price: t.price, amount: t.amount, timestamp: t.date }));
  }
  async trade(params: any): Promise<TradeResult> {
    const r: any = await this.call("trade", params);
    const id = r?.return?.order_id ?? r?.order_id ?? 0;
    return { success: 1, return: { order_id: String(id) } };
  }
  async cancel_order(params: any): Promise<CancelResult> {
    const r: any = await this.call("cancel_order", params);
    const id = r?.return?.order_id ?? params.order_id;
    return { success: 1, return: { order_id: String(id) } };
  }

  // Compatibility wrappers (camelCase legacy)
  async getBalance() { return this.get_info2(); }
  async activeOrders(currency_pair?: string) { return this.active_orders(currency_pair ? { currency_pair } : {}); }
  async cancelOrder(params: any) { return this.cancel_order(params); }
  async tradeHistory(params: any) { return this.trade_history(params); }
}
// Remove legacy-specific wrappers (legacy methods deleted)

// Simple functional wrappers (optional)
export function createPrivateApi(): PrivateApi {
  if (process.env.USE_PRIVATE_MOCK === '1') { console.log('[MOCK] Private API adapter enabled'); return createPrivateMock(); }
  const key = process.env.ZAIF_API_KEY || '';
  const secret = process.env.ZAIF_API_SECRET || '';
  return new ZaifRealPrivateApi({ key, secret });
}