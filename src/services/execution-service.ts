import { listActiveOrders, fetchTradeHistory, placeLimitOrder, cancelOrder } from "./market-service";
import { PrivateApi } from "../types/private";
let privExec: PrivateApi | undefined;
export function init(privateApi: PrivateApi) { privExec = privateApi; }
import { logExecution, logTradeError, logSignal } from "../utils/trade-logger";
import { logInfo } from "../utils/logger";
import { updateOnFill, clearOpenOrderId, loadPosition, savePosition } from "./position-store";
import { appendFillPnl } from "../utils/daily-stats";
import { getAndResetLastRequestNonceRetries } from "../api/private";
import { OrderLifecycleSummary } from "../types/domain";

const RETRY_TIMEOUT_MS = Number(process.env.RETRY_TIMEOUT_MS || 15000);
const RETRY_PRICE_OFFSET_PCT = Number(process.env.RETRY_PRICE_OFFSET_PCT || 0.002); // 0.2%
const SLIPPAGE_TIME_WINDOW_MS = Number(process.env.SLIPPAGE_TIME_WINDOW_MS || 5000);
const TOL_QTY_PCT = Number(process.env.TOL_QTY_PCT || 0.005);
const TOL_PRICE_PCT = Number(process.env.TOL_PRICE_PCT || 0.01);
const ABS_QTY_TOL = Number(process.env.ABS_QTY_TOL || 1e-8);

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

function rid() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`; }

// submitTrackedLimit removed in favor of inline logic using placeLimitOrderRaw
/// to allow better retry handling from caller
/**
 * Poll the fill state of an order.
 * @param {string} pair The currency pair.
 * @param {OrderSnapshot} snap The order snapshot.
 * @param {number} maxWaitMs The maximum wait time in milliseconds.
 * @param {number} pollIntervalMs The polling interval in milliseconds.
 * @returns {Promise<OrderSnapshot>} The updated order snapshot.
 */
export async function pollFillState(pair: string, snap: OrderSnapshot, maxWaitMs: number, pollIntervalMs = 3000): Promise<OrderSnapshot> {
    if (!snap.orderId) return snap; // already filled
    const start = Date.now();
    let pollAttempts = 0;
    while (Date.now() - start < maxWaitMs) {
        try {
            const active = await listActiveOrders(pair);
            const still = active[String(snap.orderId)];
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
                                    value += add * f.price;
                                    if (!snap.fills) snap.fills = [];
                                    snap.fills.push({ price: f.price, amount: add, ts: tsMs });
                                    matchedTids.add(f.tid);
                                    const estFillPrice = snap.intendedPrice; // 改善余地: 板 or trade history
                                    if (snap.side === 'ask') onExitFill(pair, estFillPrice, add);
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
                    logExecution("Order filled", { requestId: snap.requestId, orderId: snap.orderId, filledAmt, avg: snap.avgFillPrice });
                    if (snap.filledAmount && snap.avgFillPrice) {
                        updateOnFill({ pair: pair, side: snap.side, price: snap.avgFillPrice, amount: snap.filledAmount, ts: Date.now(), matchMethod: "history" });
                    }
                    if (snap.orderId) clearOpenOrderId(pair, snap.orderId);
                } else {
                    snap.status = "CANCELLED";
                    logTradeError("Order missing from active list; marking CANCELLED", { requestId: snap.requestId, orderId: snap.orderId, filledAmt });
                    if (snap.orderId) clearOpenOrderId(pair, snap.orderId);
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
                            updateOnFill({ pair, side: snap.side, price: estFillPrice, amount: increment, ts: Date.now(), matchMethod: "active_partial" });
                            if (snap.side === 'ask') onExitFill(pair, estFillPrice, increment);
                        }
                        if (snap.filledAmount && snap.filledAmount < snap.originalAmount) snap.status = "PARTIAL";
                        pollAttempts++;
                    }
                }
            }
        } catch (e: any) {
            logTradeError("pollFillState error", { error: e.message });
        }
        await new Promise(r => setTimeout(r, pollIntervalMs));
    }
    snap.retries = (snap.retries || 0) + pollAttempts;
    // Timeout
    snap.status = "EXPIRED";
    try {
        if (snap.orderId) await cancelOrder({ order_id: snap.orderId });
    } catch {/* ignore */ }
    logTradeError("Order expired", snap);
    return snap;
}

export function computeSlippage(intendedPrice: number, avgFillPrice?: number) {
    if (!avgFillPrice) return 0;
    return (avgFillPrice - intendedPrice) / intendedPrice;
}

export interface SubmitRetryParams extends Omit<SubmitParams, 'timeoutMs'> {
    primaryTimeoutMs?: number;
    retryTimeoutMs?: number;
    improvePricePct?: number; // override
}

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

    function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

    const dryRun = process.env.DRY_RUN === '1';
    async function submit(price: number) {
        if (dryRun) return 'DRYRUN';
        const r = await placeLimitOrder(p.currency_pair, actionSide, price, p.amount);
        return String(r.order_id || '');
    }

    async function pollLoop(snapshot: { orderId: string; originalAmount: number; assumedFilled: number; createdAt: number, expectedPx: number }, timeoutMs: number) {
        let filledQty = 0; 
        let pollAttempts = 0;
        let status: 'PENDING' | 'FILLED' = 'PENDING';
        const startTs = Date.now();
        let avgFillPrice = 0;
        let repriceAttempts = 0;
        for (let i = 1; ; i++) {
            const open = (privExec as any).active_orders ? await (privExec as any).active_orders({ currency_pair: p.currency_pair }) : [];
            const found = open.find((o: any) => o.order_id === snapshot.orderId);
            if (found) {
                const remaining = Math.max(0, Number(found.amount));
                const newAssumed = Math.max(0, snapshot.originalAmount - remaining);
                if (newAssumed > snapshot.assumedFilled) {
                    const delta = newAssumed - snapshot.assumedFilled;
                    snapshot.assumedFilled = newAssumed; filledQty += delta;
                    avgFillPrice = snapshot.expectedPx; // crude
                }
                if (remaining === 0) { status = 'FILLED'; }
            } else {
                const hist = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 200 }) : [];
                const qtyById = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qtyById >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                } else if (qtyById > snapshot.assumedFilled) {
                    const delta = qtyById - snapshot.assumedFilled; snapshot.assumedFilled = qtyById; filledQty += delta;
                    const priceAvg = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qtyById || snapshot.expectedPx;
                    avgFillPrice = priceAvg;
                }
            }
            // first poll always cross-check trade history (already done for missing case; do for found too)
            if (i === 1) {
                const hist0 = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 100 }) : [];
                const qty0 = hist0.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qty0 > snapshot.assumedFilled) {
                    const delta0 = qty0 - snapshot.assumedFilled; snapshot.assumedFilled = qty0; filledQty += delta0;
                }
                if (qty0 >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                }
                logSignal('[POLL0]', { orderId: snapshot.orderId, openCount: open.length, openAmtById: found ? Number(found.amount) : null, histQtyById: qty0 });
            }
            // Extra trade_history every 2 polls (and always first already done above)
            if (i > 1 && i % 2 === 0 && status !== 'FILLED') {
                const hist2 = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 100 }) : [];
                const qty2 = hist2.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qty2 >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                } else if (qty2 > snapshot.assumedFilled) {
                    const delta = qty2 - snapshot.assumedFilled;
                    snapshot.assumedFilled = qty2;
                    filledQty += delta;
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
                            try { await cancelOrder({ order_id: snapshot.orderId as any }); } catch { }
                        }
                        const prevLimit = (actionSide === 'BUY' ? Math.max(refPx, snapshot.expectedPx) : Math.min(refPx, snapshot.expectedPx));
                        let newLimit = prevLimit;
                        if (actionSide === 'BUY') newLimit = Math.min(bestAsk, prevLimit * (1 + off));
                        else newLimit = Math.max(bestBid, prevLimit * (1 - off));
                        const newId = await submit(newLimit);
                        logInfo('[SLIPPAGE] reprice', { requestId: snapshot.orderId, orderId: newId, repriceAttempts: repriceAttempts + 1, slip, newLimit });
                        snapshot.orderId = newId;
                        repriceAttempts++;
                        continue; // next poll cycle
                    } else {
                        status = 'FILLED'; // give up and treat as filled or could mark canceled
                        logInfo('[SLIPPAGE] cancel', { requestId: snapshot.orderId, slip, repriceAttempts });
                    }
                }
            }
            if (filledQty >= snapshot.originalAmount) status = 'FILLED';
            if (filledQty > 0 && (Date.now() - startTs) >= 500) { 
                status = 'FILLED'; 
                return { filledQty, pollAttempts, status, repriceAttempts }; 
            }
            if (status === 'FILLED') return { filledQty, pollAttempts, status, repriceAttempts };
            if (i >= POLL_MIN_CYCLES && Date.now() - startTs >= timeoutMs) return { filledQty, pollAttempts, status, repriceAttempts };
            await sleep(POLL_INTERVAL_MS);
        }
    }

    // First submission
    if (dryRun) {
        logInfo('[DRYRUN] submitWithRetry simulate fill', { requestId, side: actionSide, qty: p.amount, price: p.limitPrice });
        return { requestId, side: actionSide as any, intendedQty: p.amount, filledQty: p.amount, avgExpectedPrice: p.limitPrice, avgFillPrice: p.limitPrice, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 1, repriceAttempts: 0 } as unknown as OrderLifecycleSummary;
    }
    const submitOrderId = await submit(p.limitPrice);
    if (!submitOrderId) {
        logTradeError('Missing orderId', { requestId });
        return { requestId, side: actionSide, intendedQty: p.amount, filledQty: 0, avgExpectedPrice: p.limitPrice, avgFillPrice: 0, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 0 } as unknown as OrderLifecycleSummary;
    }
    logSignal('[SUBMIT]', { requestId, orderId: submitOrderId, side: actionSide, price: p.limitPrice, amount: p.amount });
    const expectedPx = p.limitPrice; // placeholder best price snapshot
    const snapshot1: { requestId: string; orderId: string; originalAmount: number; assumedFilled: number; createdAt: number; expectedPx: number } = { requestId, orderId: submitOrderId, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx };
    const nonce1 = getAndResetLastRequestNonceRetries();
    const poll1: any = await pollLoop(snapshot1, primaryTimeout);
    let submitRetryCount = 0; 
    let cancelRetryCount = 0; 
    let pollRetryCount = poll1.pollAttempts; 
    let filledQty = poll1.filledQty; 
    let avgFillPrice = filledQty > 0 ? p.limitPrice : 0; 
    let nonceRetryCount = nonce1; 
    let improvedPrice = p.limitPrice; 
    let secondStats: any = null; 
    let repriceAttemptsTotal = poll1.repriceAttempts || 0;
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
        const snapshot2: { requestId: string; orderId: string; originalAmount: number; assumedFilled: number; createdAt: number; expectedPx: number } = { requestId, orderId: submitOrderId2, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx };
        secondStats = await pollLoop(snapshot2, retryTimeout);
        pollRetryCount += secondStats.pollAttempts;
        repriceAttemptsTotal += secondStats.repriceAttempts || 0;
        if (secondStats.filledQty > 0) { filledQty = secondStats.filledQty; avgFillPrice = improvedPrice; }
    }
    const durationMs = Date.now() - snapshot1.createdAt;
    const slippagePct = filledQty > 0 ? (avgFillPrice - p.limitPrice) / p.limitPrice : 0;
    const totalRetryCount = submitRetryCount + pollRetryCount + cancelRetryCount + nonceRetryCount;
    const summary = { requestId, side: actionSide as any, intendedQty: p.amount, filledQty, avgExpectedPrice: p.limitPrice, avgFillPrice, slippagePct, durationMs, submitRetryCount, pollRetryCount, cancelRetryCount, nonceRetryCount, totalRetryCount, filledCount: filledQty > 0 ? 1 : 0, repriceAttempts: repriceAttemptsTotal } as unknown as OrderLifecycleSummary;
    logInfo('[ORDER]', { requestId, orderId: submitOrderId, side: actionSide, filledQty, totalRetryCount, pollRetryCount, submitRetryCount, cancelRetryCount, nonceRetryCount, repriceAttempts: repriceAttemptsTotal, durationMs });
    return summary;
}

// Cache today's date for performance; update if date changes
let cachedToday = new Date().toISOString().slice(0, 10);
let lastDateCheck = Date.now();
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

// Incremental realized PnL for exit fills (long-only assumption for now)
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