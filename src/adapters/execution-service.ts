import { listActiveOrders, fetchTradeHistory, placeLimitOrder, cancelOrder } from "./market-service";
import { PrivateApi } from "../types/private";
import { sleep } from "../utils/toolkit";
let privExec: PrivateApi | undefined;
export function init(privateApi: PrivateApi) { privExec = privateApi; }
import { logExecution, logTradeError, logSignal } from "../utils/trade-logger";
import { logInfo } from "../utils/logger";
import BaseService from "./base-service";
import type { Logger } from "../utils/logger";
import { loadPosition, savePosition } from "./position-store";
import { appendFillPnl } from "../utils/daily-stats";
import { getAndResetLastRequestNonceRetries } from "../api/private";
import { OrderLifecycleSummary } from "../types/domain";
import { getQtyEpsilon } from "./risk-config";
import { getEventBus } from "../application/events/bus";
import type { OrderSide } from "../application/events/types";

const RETRY_TIMEOUT_MS = Number(process.env.RETRY_TIMEOUT_MS ?? 15000);
const RETRY_PRICE_OFFSET_PCT = Number(process.env.RETRY_PRICE_OFFSET_PCT ?? 0.002); // 0.2%
const SLIPPAGE_TIME_WINDOW_MS = Number(process.env.SLIPPAGE_TIME_WINDOW_MS ?? 5000);
const TOL_QTY_PCT = Number(process.env.TOL_QTY_PCT ?? 0.005);
const TOL_PRICE_PCT = Number(process.env.TOL_PRICE_PCT ?? 0.01);
const ABS_QTY_TOL = Number(process.env.ABS_QTY_TOL ?? 1e-8);

// Local BaseService instance for retry + category logging without changing public API
const execSvc = new BaseService();
export function setExecutionLogger(lg: Logger) { execSvc.setLogger(lg); }

export interface OrderBookLevel { price: number; amount: number; }

export interface OrderSnapshot {
    side: "bid" | "ask";
    intendedPrice: number; // price aimed
    amount: number;
    orderId?: number;
    submittedAt?: number; // ms
    filledAmount?: number;
    avgFillPrice?: number;
    status?: "NEW" | "PARTIAL" | "FILLED" | "CANCELLED" | "EXPIRED";
    requestId?: string;
    snapshotLevels?: { bids: OrderBookLevel[]; asks: OrderBookLevel[] };
    fills?: Array<{ price: number; amount: number; ts: number }>;
    originalAmount?: number; // keep track for partial calc
    retries?: number; // total retry attempts
}

export interface SubmitParams {
    currency_pair: string;
    side: "bid" | "ask";
    limitPrice: number;
    amount: number;
    timeoutMs: number; // time to wait before cancel/adjust
    retry?: boolean;
    orderBook?: { bids: [number, number][]; asks: [number, number][] }; // pre snapshot
}

function generateRequestId() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }

/**
 * Poll the fill state of an order.
 * 
 * Updates the following fields of the provided OrderSnapshot:
 * - filledAmount
 * - avgFillPrice
 * - status
 * - retries
 * - fills
 * 
 * @param pair The currency pair.
 * @param snap The order snapshot.
 * @param maxWaitMs The maximum wait time in milliseconds.
 * @param pollIntervalMs The polling interval in milliseconds.
 * @returns The updated order snapshot.
 */
export async function pollFillState(pair: string, snap: OrderSnapshot, maxWaitMs: number, pollIntervalMs = 3000): Promise<OrderSnapshot> {
    if (!snap.orderId) return snap; // already filled
    const start = Date.now();
    let pollAttempts = 0;
    while (Date.now() - start < maxWaitMs) {
        try {
            const active: any = await listActiveOrders(pair) as any;
            // tests may mock as object map or array of orders; support both
            let still: any = null;
            if (Array.isArray(active)) {
                still = active.find((o: any) => o.order_id === snap.orderId);
            } else if (active && typeof active === 'object') {
                still = active[String(snap.orderId)] || null;
            }
            if (!still) {
                const hist = await fetchTradeHistory(pair, { count: 200 });
                const windowStart = (snap.submittedAt || 0) - SLIPPAGE_TIME_WINDOW_MS;
                const windowEnd = Date.now() + SLIPPAGE_TIME_WINDOW_MS;
                const sideChar = snap.side === "bid" ? "bid" : "ask";
                let filledAmt = snap.filledAmount || 0; let value = (snap.avgFillPrice || 0) * filledAmt;
                let usedOrderId = false;
                const matchedTids = new Set<number>();
                let matchedQtySoFar = snap.filledAmount || 0;
                for (const f of hist as any[]) {
                    const tsMs = (f.timestamp || 0) * 1000;
                    // If order_id exists and matches snapshot orderId -> prefer direct match
                    if (snap.orderId && f.order_id && Number(f.order_id) === snap.orderId) {
                        usedOrderId = true;
                        if (filledAmt < snap.amount) {
                            const add = Math.min(f.amount, snap.amount - filledAmt);
                            filledAmt += add;
                            value += add * f.price;
                            if (!snap.fills) snap.fills = [];
                            snap.fills.push({ price: f.price, amount: add, ts: tsMs });
                            const estFillPrice = snap.intendedPrice; // 改善余地: 板 or trade history
                            if (snap.side === 'ask') onExitFill(pair, estFillPrice, add);
                        }
                    } else if (!usedOrderId && tsMs >= windowStart && tsMs <= windowEnd && f.side === sideChar) {
                        if (filledAmt < snap.amount && !matchedTids.has(f.tid)) {
                            const refPrice = snap.intendedPrice;
                            const priceOk = Math.abs(f.price - refPrice) / refPrice <= TOL_PRICE_PCT;
                            const targetRemaining = snap.amount - matchedQtySoFar;
                            if (targetRemaining <= 0) break;
                            const qtyTol = Math.min(targetRemaining * TOL_QTY_PCT, ABS_QTY_TOL);
                            const qtyOk = f.amount <= targetRemaining + qtyTol; // accept plausible partial up to remaining + tol
                            if (priceOk && qtyOk) {
                                const addRaw = Math.min(f.amount, targetRemaining);
                                const add = Math.max(0, addRaw);
                                if (add > 0) {
                                    matchedQtySoFar += add;
                                    filledAmt += add;
                                    value += add * f.price;
                                    if (!snap.fills) snap.fills = [];
                                    snap.fills.push({ price: f.price, amount: add, ts: tsMs });
                                    matchedTids.add(f.tid);
                                    const estFillPrice = snap.intendedPrice; // 改善余地: 板 or trade history
                                    if (snap.side === 'ask') onExitFill(pair, estFillPrice, add);
                                }
                                if (matchedQtySoFar >= snap.amount) break;
                            }
                        }
                    }
                }
                if (usedOrderId) logSignal(`fill_match_method=order_id requestId=${snap.requestId}`); else logSignal(`fill_match_method=heuristic requestId=${snap.requestId}`);
                if (filledAmt > 0) {
                    snap.avgFillPrice = value / filledAmt;
                    snap.filledAmount = filledAmt;
                }
                if (filledAmt >= snap.amount * 0.999) {
                    snap.status = "FILLED";
                    logExecution("Order filled", { requestId: snap.requestId, orderId: snap.orderId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', amount: snap.filledAmount, price: snap.avgFillPrice, retries: snap.retries || 0 });
                    try { if (snap.orderId && snap.filledAmount && snap.avgFillPrice) { getEventBus().publish({ type: 'ORDER_FILLED', orderId: String(snap.orderId), requestId: snap.requestId || generateRequestId(), pair, side: (snap.side === 'bid' ? 'buy' : 'sell') as OrderSide, amount: snap.filledAmount, price: snap.avgFillPrice, filled: snap.filledAmount, avgPrice: snap.avgFillPrice }); } } catch {}
                    // PositionStore updates are handled via event subscribers
                } else {
                    snap.status = "CANCELLED";
                    logTradeError("Order missing from active list; marking CANCELLED", { requestId: snap.requestId, orderId: snap.orderId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', amount: filledAmt, cause: { code: 'ORDER_NOT_FOUND' } });
                    try { if (snap.orderId) { getEventBus().publish({ type: 'ORDER_CANCELED', orderId: String(snap.orderId), requestId: snap.requestId || generateRequestId(), pair, side: (snap.side === 'bid' ? 'buy' : 'sell') as OrderSide, amount: filledAmt, price: snap.intendedPrice }); } } catch {}
                    // PositionStore updates are handled via event subscribers
                }
                return snap;
            }
            // active_orders の still.amount (残量) があれば部分約定計算
            if (still) {
                const remaining = still.amount; // Zaif active_orders: amount = remaining
                if (remaining != null && snap.originalAmount) {
                    const filled = snap.originalAmount - remaining;
                    if (filled > 0) {
                        if (!snap.filledAmount || filled > snap.filledAmount) {
                            if (!snap.fills) snap.fills = [];
                            const increment = filled - (snap.filledAmount || 0);
                            const estFillPrice = snap.intendedPrice; // 改善余地: 板 or trade history
                            snap.fills.push({ price: estFillPrice, amount: increment, ts: Date.now() });
                            const totalValue = (snap.avgFillPrice || 0) * (snap.filledAmount || 0) + increment * estFillPrice;
                            snap.filledAmount = filled;
                            snap.avgFillPrice = totalValue / snap.filledAmount;
                            if (snap.side === 'ask') onExitFill(pair, estFillPrice, increment);
                        }
                        if (snap.filledAmount && snap.filledAmount < snap.originalAmount) snap.status = "PARTIAL";
                        pollAttempts++;
                    }
                }
            }
        } catch (e: any) {
            logTradeError("pollFillState error", { requestId: snap.requestId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', cause: { code: e?.code || 'POLL_ERROR', message: e?.message } });
        }
        await sleep(pollIntervalMs);
    }
    snap.retries = (snap.retries || 0) + pollAttempts;
    // Final check before timeout: if order disappeared, try matching fills by order_id
    try {
    const active: any = await listActiveOrders(pair) as any;
        let still: any = null;
        if (Array.isArray(active)) still = active.find((o: any) => o.order_id === snap.orderId);
        else if (active && typeof active === 'object') still = active[String(snap.orderId)] || null;
        if (!still) {
            const hist = await fetchTradeHistory(pair, { count: 200 });
            const byIdQty = (hist as any[]).filter((f: any) => Number(f.order_id) === snap.orderId).reduce((s, f) => s + Number(f.amount), 0);
            if (byIdQty >= (snap.amount || 0)) {
                snap.filledAmount = byIdQty;
                snap.avgFillPrice = (hist as any[]).filter((f: any) => Number(f.order_id) === snap.orderId)
                    .reduce((s, f) => s + Number(f.amount) * Number(f.price), 0) / byIdQty;
                snap.status = "FILLED";
                try { if (snap.orderId && snap.filledAmount && snap.avgFillPrice) { getEventBus().publish({ type: 'ORDER_FILLED', orderId: String(snap.orderId), requestId: snap.requestId || generateRequestId(), pair, side: (snap.side === 'bid' ? 'buy' : 'sell') as OrderSide, amount: snap.filledAmount, price: snap.avgFillPrice, filled: snap.filledAmount, avgPrice: snap.avgFillPrice }); } } catch {}
                return snap;
            }
        }
    } catch {}
    // Timeout
    snap.status = "EXPIRED";
    try {
        if (snap.orderId) { try { await execSvc.withRetry(() => cancelOrder({ order_id: snap.orderId as any }), 'cancelOrder', 3, 100, { category: 'EXEC', requestId: snap.requestId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', amount: snap.amount, price: snap.intendedPrice }); } catch {/* ignore */ } }
    } catch {/* ignore */ }
    try { if (snap.orderId) { getEventBus().publish({ type: 'ORDER_EXPIRED', orderId: String(snap.orderId), requestId: snap.requestId || generateRequestId(), pair, side: (snap.side === 'bid' ? 'buy' : 'sell') as OrderSide, amount: snap.amount || 0, price: snap.intendedPrice }); } } catch {}
    // Threshold-based EXEC warnings
    const check = shouldWarnPollingSlow(start, Date.now(), 30000);
    if (check.warn) execSvc.clog('EXEC','WARN','polling slow',{ requestId: snap.requestId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', amount: snap.amount, price: snap.intendedPrice, elapsedMs: check.elapsed });
    logTradeError("Order expired", { requestId: snap.requestId, pair, side: snap.side === 'bid' ? 'buy' : 'sell', amount: snap.amount, price: snap.intendedPrice, retries: snap.retries || 0 });
    return snap;
}

export function shouldWarnPollingSlow(startMs: number, nowMs: number, thresholdMs = 30000): { warn: boolean; elapsed: number } {
    const elapsed = nowMs - startMs;
    return { warn: elapsed > thresholdMs, elapsed };
}

/**
 * Compute the slippage between the intended and average fill prices.
 * @param intendedPrice The price at which the order was intended to be filled.
 * @param avgFillPrice The average price at which the order was actually filled.
 * @returns The slippage as a percentage.
 */
export function computeSlippage(intendedPrice: number, avgFillPrice?: number) {
    if (!avgFillPrice) return 0;
    return (avgFillPrice - intendedPrice) / intendedPrice;
}

export interface SubmitRetryParams extends Omit<SubmitParams, 'timeoutMs'> {
    primaryTimeoutMs?: number;
    retryTimeoutMs?: number;
    improvePricePct?: number; // override
}

/**
 * Submit a limit order with retry logic, adjusting the price if not filled within specified timeouts.
 * @param {SubmitRetryParams} p The parameters for submitting the order with retry logic.
 * @returns {Promise<OrderLifecycleSummary>} A summary of the order lifecycle.
 */
export async function submitWithRetry(p: SubmitRetryParams): Promise<OrderLifecycleSummary> {
    const actionSide = p.side === 'bid' ? 'BUY' : 'SELL';
    const requestId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL_MS || 300);
    const POLL_MIN_CYCLES = Number(process.env.POLL_MIN_CYCLES || 3);
    const primaryTimeout = p.primaryTimeoutMs ?? RETRY_TIMEOUT_MS;
    const retryTimeout = p.retryTimeoutMs ?? RETRY_TIMEOUT_MS;
    const improvePct = p.improvePricePct ?? RETRY_PRICE_OFFSET_PCT;
    const CANCEL_MAX = Number(process.env.CANCEL_MAX_RETRIES || 1);
    const BACKOFF = Number(process.env.RETRY_BACKOFF_MS || 300);


    const dryRun = process.env.DRY_RUN === '1';
    async function submit(price: number) {
        if (dryRun) return 'DRYRUN';
        const t0 = Date.now();
        try {
            const r = await placeLimitOrder(p.currency_pair, actionSide, price, p.amount);
            const dt = Date.now() - t0;
            const acceptedQty = Number((r as any)?.submitted_amount ?? (r as any)?.amount ?? p.amount);
            const eps = getQtyEpsilon(p.currency_pair);
            const qtyRounded = Math.abs(acceptedQty - p.amount) > eps;
            const evSide = (p.side === 'bid' ? 'buy' : 'sell') as OrderSide;
            if (dt > 800) execSvc.clog('ORDER','WARN','slow accept',{ requestId, pair: p.currency_pair, side: evSide, amount: p.amount, price, retries: 0, elapsedMs: dt });
            if (qtyRounded) execSvc.clog('ORDER','WARN','quantity rounded',{ requestId, pair: p.currency_pair, side: evSide, amount: p.amount, price, retries: 0, acceptedAmount: acceptedQty });
            try { getEventBus().publish({ type: 'ORDER_SUBMITTED', orderId: String(r.order_id||''), requestId, pair: p.currency_pair, side: evSide, amount: p.amount, price }); } catch {}
            return String(r.order_id || '');
        } catch (e: any) {
            execSvc.clog('ORDER','ERROR','submit failed',{ requestId, pair: p.currency_pair, side: p.side === 'bid' ? 'buy' : 'sell', amount: p.amount, price, retries: 0, cause: { code: e?.code ?? e?.cause?.code ?? 'SUBMIT_ERROR', message: e?.message } });
            throw e;
        }
    }

    async function pollLoop(snapshot: { orderId: string; originalAmount: number; assumedFilled: number; createdAt: number; expectedPx: number; actionSide: 'BUY' | 'SELL' }, timeoutMs: number) {
        let filledQty: number = 0; 
        let pollAttempts: number = 0;
        let status: 'PENDING' | 'FILLED' = 'PENDING';
        const startTs = Date.now();
        let avgFillPrice: number = 0;
        let repriceAttempts: number = 0;
        let filledCount: number = 0;
        for (let i = 1; ; i++) {
            const open: any[] = (privExec as any).active_orders ? await (privExec as any).active_orders({ currency_pair: p.currency_pair }) : [];
            const found = open.find((o: any) => o.order_id === snapshot.orderId);
            if (found) {
                const remaining = Math.max(0, Number(found.amount));
                const newAssumed = Math.max(0, snapshot.originalAmount - remaining);
                if (newAssumed > snapshot.assumedFilled) {
                    const delta = newAssumed - snapshot.assumedFilled;
                    snapshot.assumedFilled = newAssumed; filledQty += delta;
                    avgFillPrice = snapshot.expectedPx; // crude
                    filledCount++;
                }
                if (remaining === 0) { status = 'FILLED'; }
            } else {
                const hist: any[] = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 200 }) : [];
                const qtyById = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qtyById >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                    filledCount++;
                    avgFillPrice = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qtyById || snapshot.expectedPx;
                } else if (qtyById > snapshot.assumedFilled) {
                    const delta = qtyById - snapshot.assumedFilled; snapshot.assumedFilled = qtyById; filledQty += delta;
                    const priceAvg = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qtyById || snapshot.expectedPx;
                    avgFillPrice = priceAvg;
                    filledCount++;
                }
            }
            // first poll always cross-check trade history (already done for missing case; do for found too)
            if (i === 1) {
                const hist0: any[] = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 100 }) : [];
                const qty0 = hist0.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qty0 > snapshot.assumedFilled) {
                    const delta0 = qty0 - snapshot.assumedFilled; snapshot.assumedFilled = qty0; filledQty += delta0;
                }
                if (qty0 >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                    avgFillPrice = hist0.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qty0 || snapshot.expectedPx;
                }
                logSignal('[POLL0]', { orderId: snapshot.orderId, openCount: open.length, openAmtById: found ? Number(found.amount) : null, histQtyById: qty0 });
            }
            // Extra trade_history every 2 polls (and always first already done above)
            if (i > 1 && i % 2 === 0 && status !== 'FILLED') {
                const hist2: any[] = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 100 }) : [];
                const qty2 = hist2.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qty2 >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                    avgFillPrice = hist2.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qty2 || snapshot.expectedPx;
                } else if (qty2 > snapshot.assumedFilled) {
                    const delta = qty2 - snapshot.assumedFilled;
                    snapshot.assumedFilled = qty2;
                    filledQty += delta;
                    avgFillPrice = hist2.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qty2 || snapshot.expectedPx;
                }
            }
            pollAttempts++;
            // Slippage control & reprice (skip if dryRun or already filled)
            if (!dryRun && status !== 'FILLED') {
                const maxSlip = Number(process.env.RISK_MAX_SLIPPAGE_PCT || '0.005');
                const off = Number(process.env.REPRICE_OFFSET_PCT || '0.001');
                const maxReprice = Number(process.env.REPRICE_MAX_ATTEMPTS || '2');
                // fetch simple best prices (fallback to expectedPx)
                const bestAsk = snapshot.expectedPx * (actionSide === 'BUY' ? 1 : 1.001);
                const bestBid = snapshot.expectedPx * (actionSide === 'BUY' ? 0.999 : 1);
                const refPx = avgFillPrice > 0 ? avgFillPrice : (actionSide === 'BUY' ? bestAsk : bestBid);
                const slip = Math.abs((refPx - snapshot.expectedPx) / snapshot.expectedPx);
                if (slip > maxSlip) {
                    if (repriceAttempts < maxReprice) {
                        // cancel current
                        if (snapshot.orderId !== 'DRYRUN') {
                            try {
                                await execSvc.withRetry(() => cancelOrder({ order_id: snapshot.orderId as any }), 'cancelOrder', 3, 100);
                            } catch { }
                        }
                        const prevLimit = (actionSide === 'BUY' ? Math.max(refPx, snapshot.expectedPx) : Math.min(refPx, snapshot.expectedPx));
                        let newLimit = prevLimit;
                        if (actionSide === 'BUY') newLimit = Math.min(bestAsk, prevLimit * (1 + off));
                        else newLimit = Math.max(bestBid, prevLimit * (1 - off));
                        const newId = await submit(newLimit);
                        execSvc.clog('EXEC', 'INFO', '[SLIPPAGE] reprice', { requestId: snapshot.orderId, pair: p.currency_pair, side: actionSide, amount: snapshot.originalAmount, price: newLimit, repriceAttempts: repriceAttempts + 1, slip, newLimit });
                        try { getEventBus().publish({ type: 'SLIPPAGE_REPRICED', orderId: String(snapshot.orderId), requestId, pair: p.currency_pair, side: (p.side==='bid'?'buy':'sell') as OrderSide, amount: snapshot.originalAmount, price: newLimit, attempts: repriceAttempts + 1 }); } catch {}
                        snapshot.orderId = newId;
                        repriceAttempts++;
                        continue; // next poll cycle
                    } else {
                        status = 'FILLED'; // give up and treat as filled or could mark canceled
                        execSvc.clog('EXEC', 'ERROR', '[SLIPPAGE] max reprice reached', { requestId: snapshot.orderId, pair: p.currency_pair, side: actionSide, amount: snapshot.originalAmount, price: snapshot.expectedPx, repriceAttempts, maxReprice });
                    }
                }
            }
            if (filledQty >= snapshot.originalAmount) status = 'FILLED';
            if (filledQty > 0 && (Date.now() - startTs) >= 500) {
                status = 'FILLED';
                try { getEventBus().publish({ type: 'ORDER_FILLED', orderId: String(snapshot.orderId), requestId, pair: p.currency_pair, side: (p.side==='bid'?'buy':'sell') as OrderSide, amount: filledQty, price: avgFillPrice || snapshot.expectedPx, filled: filledQty, avgPrice: avgFillPrice || snapshot.expectedPx }); } catch {}
                return { filledQty, pollAttempts, status, repriceAttempts, filledCount, avgFillPrice };
            }
            if (status === 'FILLED') { try { getEventBus().publish({ type: 'ORDER_FILLED', orderId: String(snapshot.orderId), requestId, pair: p.currency_pair, side: (p.side==='bid'?'buy':'sell') as OrderSide, amount: filledQty, price: avgFillPrice || snapshot.expectedPx, filled: filledQty, avgPrice: avgFillPrice || snapshot.expectedPx }); } catch {} return { filledQty, pollAttempts, status, repriceAttempts, filledCount, avgFillPrice }; }
            const elapsed = Date.now() - startTs;
            if (elapsed > 30000 && status !== ("FILLED" as typeof status)) {
                execSvc.clog('EXEC','WARN','polling slow',{ requestId: snapshot.orderId, pair: p.currency_pair, side: actionSide, amount: snapshot.originalAmount, price: snapshot.expectedPx, elapsedMs: elapsed });
            }
            if (i >= POLL_MIN_CYCLES && elapsed >= timeoutMs) return { filledQty, pollAttempts, status, repriceAttempts, filledCount, avgFillPrice };
            await sleep(POLL_INTERVAL_MS);
        }
    }

    // First submission
    if (dryRun) {
    execSvc.clog('EXEC', 'INFO', '[DRYRUN] submitWithRetry simulate fill', { requestId, pair: p.currency_pair, side: actionSide, amount: p.amount, price: p.limitPrice });
        return { requestId, side: actionSide as any, intendedQty: p.amount, filledQty: p.amount, avgExpectedPrice: p.limitPrice, avgFillPrice: p.limitPrice, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 1, repriceAttempts: 0 } as unknown as OrderLifecycleSummary;
    }
    const submitOrderId = await submit(p.limitPrice);
    if (!submitOrderId) {
        execSvc.clog('ORDER','ERROR','missing orderId',{ requestId, pair: p.currency_pair, side: p.side === 'bid' ? 'buy' : 'sell', amount: p.amount, price: p.limitPrice, retries: 0, cause: { code: 'ORDER_ID_MISSING' } });
        logTradeError('Missing orderId', { requestId, pair: p.currency_pair });
        return { requestId, side: actionSide, intendedQty: p.amount, filledQty: 0, avgExpectedPrice: p.limitPrice, avgFillPrice: 0, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 0 } as unknown as OrderLifecycleSummary;
    }
    logSignal('[SUBMIT]', { requestId, orderId: submitOrderId, side: actionSide, price: p.limitPrice, amount: p.amount });
    const expectedPx = p.limitPrice; // placeholder best price snapshot
    const snapshot1: { requestId: string; orderId: string; originalAmount: number; assumedFilled: number; createdAt: number; expectedPx: number; actionSide: 'BUY' | 'SELL' } = { requestId, orderId: submitOrderId, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx, actionSide };
    const nonce1 = getAndResetLastRequestNonceRetries();
    const poll1: any = await pollLoop(snapshot1, primaryTimeout);
    let submitRetryCount = 0; 
    let cancelRetryCount = 0; 
    let pollRetryCount = poll1.pollAttempts; 
    let filledQty = poll1.filledQty; 
    let avgFillPrice = poll1.avgFillPrice ?? (filledQty > 0 ? p.limitPrice : 0); 
    let nonceRetryCount = nonce1; 
    let improvedPrice = p.limitPrice; 
    let secondStats: any = null; 
    let repriceAttemptsTotal = poll1.repriceAttempts || 0;
    let filledCount = poll1.filledCount || (filledQty > 0 ? 1 : 0);
    if (poll1.status !== 'FILLED') {
        // cancel & resubmit improved
        submitRetryCount = 1;
        const improveFactor = p.side === 'ask' ? (1 - improvePct) : (1 + improvePct);
        improvedPrice = p.limitPrice * improveFactor;
        for (let c = 0; c < CANCEL_MAX; c++) {
            try { await cancelOrder({ order_id: submitOrderId as any }); break; } catch { cancelRetryCount++; await sleep(BACKOFF); }
        }
        const submitOrderId2 = await submit(improvedPrice);
        const nonce2 = getAndResetLastRequestNonceRetries();
        nonceRetryCount += nonce2;
        logSignal('[SUBMIT]', { requestId, orderId: submitOrderId2, improved: true, price: improvedPrice, amount: p.amount });
        const snapshot2: { requestId: string; orderId: string; originalAmount: number; assumedFilled: number; createdAt: number; expectedPx: number; actionSide: 'BUY' | 'SELL' } = { requestId, orderId: submitOrderId2, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx, actionSide };
        secondStats = await pollLoop(snapshot2, retryTimeout);
        pollRetryCount += secondStats.pollAttempts;
        repriceAttemptsTotal += secondStats.repriceAttempts || 0;
        if (secondStats.filledQty > 0) { filledQty = secondStats.filledQty; avgFillPrice = secondStats.avgFillPrice; }
        if (secondStats.filledCount) { filledCount += secondStats.filledCount; }
    }
    const durationMs = Date.now() - snapshot1.createdAt;
    const slippagePct = filledQty > 0 ? (avgFillPrice - p.limitPrice) / p.limitPrice : 0;
    const totalRetryCount = submitRetryCount + pollRetryCount + cancelRetryCount + nonceRetryCount;
    // Note: avgFillPrice may not always reflect the actual fill price from the exchange; in dry-run or fallback cases, it may be set to the intended price or an estimated value.
    // Please refer to the implementation for details on how avgFillPrice is determined.
    // Normalize filledCount to 0/1 for lifecycle-level count
    const filledCountNorm = filledQty > 0 ? 1 : 0;
    const summary = { requestId, side: actionSide as any, intendedQty: p.amount, filledQty, avgExpectedPrice: p.limitPrice, avgFillPrice, slippagePct, durationMs, submitRetryCount, pollRetryCount, cancelRetryCount, nonceRetryCount, totalRetryCount, filledCount: filledCountNorm, repriceAttempts: repriceAttemptsTotal } as unknown as OrderLifecycleSummary;
    execSvc.clog('ORDER', 'INFO', '[ORDER]', { requestId, pair: p.currency_pair, side: actionSide, amount: p.amount, price: p.limitPrice, orderId: submitOrderId, filledQty, totalRetryCount, pollRetryCount, submitRetryCount, cancelRetryCount, nonceRetryCount, repriceAttempts: repriceAttemptsTotal, durationMs, filledCount });
    return summary;
}

// Cache today's date for performance; update if date changes
let cachedToday = new Date().toISOString().slice(0, 10);
let lastDateCheck = Date.now();
/** 
 * Get today's date in 'YYYY-MM-DD' format, updating the cache if more than an hour has passed or the date has changed.
 * @returns {string} Today's date in 'YYYY-MM-DD' format.
 */
function getToday() {
    const now = Date.now();
    // Check if more than 1 hour has passed or date changed
    if (now - lastDateCheck > 3600000) {
        const newToday = new Date().toISOString().slice(0, 10);
        if (newToday !== cachedToday) {
            cachedToday = newToday;
        }
        lastDateCheck = now;
    }
    return cachedToday;
}

/** 
 * Incremental realized PnL for exit fills ((currently assumes long-only; extend for short/other strategies as needed)) 
 * @param {string} pair Currency pair
 * @param {number} fillPrice Fill price
 * @param {number} fillQty Fill quantity
 */
export function onExitFill(pair: string, fillPrice: number, fillQty: number) {
    const today = getToday();
    const pos = loadPosition(pair);
    if (!pos) return; // nothing to realize
    const preAvg = pos.avgPrice;
    const realized = (fillPrice - preAvg) * fillQty; // long exit
    appendFillPnl(today, realized, pair);
    // Reduce qty without changing avgPrice for leftover
    const newQty = Math.max(0, pos.qty - fillQty);
    pos.qty = newQty;
    if (newQty === 0) {
        pos.avgPrice = 0;
        pos.dcaRemainder = 0;
    }
    savePosition(pos);
}
