import { getActiveOrders, fetchTradeHistory, placeLimitOrder, cancelOrder } from "./market";
import { PrivateApi } from "../types/private";
let privExec: PrivateApi | undefined;
export function initExecution(privateApi: PrivateApi) { privExec = privateApi; }
import { logExecution, logTradeError, logSignal } from "../utils/trade-logger";
import { logInfo } from "../utils/logger";
import { updateOnFill, clearOpenOrderId, loadPosition, savePosition } from "./position-store";
import { appendFillPnl } from "../utils/daily-stats";
import { logFeatureSample } from "../utils/features-logger";
import { calculateRsi, calculateSma } from "./risk";
import { sleep } from "../utils/toolkit";
import { getPriceSeries } from "../utils/price-cache";
import { getAndResetLastRequestNonceRetries } from "../api/zaif-private";
import { OrderLifecycleSummary } from "../types/private";

const RETRY_TIMEOUT_MS = Number(process.env.RETRY_TIMEOUT_MS || 15000);
const RETRY_PRICE_OFFSET_PCT = Number(process.env.RETRY_PRICE_OFFSET_PCT || 0.002);
const SLIPPAGE_TIME_WINDOW_MS = Number(process.env.SLIPPAGE_TIME_WINDOW_MS || 5000);
const TOL_QTY_PCT = Number(process.env.TOL_QTY_PCT || 0.005);
const TOL_PRICE_PCT = Number(process.env.TOL_PRICE_PCT || 0.01);
const ABS_QTY_TOL = Number(process.env.ABS_QTY_TOL || 1e-8);

export interface OrderBookLevel { price: number; amount: number; }
export interface OrderSnapshot { 
    side: "bid" | "ask"; 
    intendedPrice: number; 
    amount: number; 
    orderId?: number; 
    submittedAt?: number; 
    filledAmount?: number; 
    avgFillPrice?: number; 
    status?: "NEW" | "PARTIAL" | "FILLED" | "CANCELLED" | "EXPIRED"; 
    requestId?: string; 
    snapshotLevels?: { bids: OrderBookLevel[]; asks: OrderBookLevel[] }; 
    fills?: Array<{ price: number; amount: number; ts: number }>; 
    originalAmount?: number; 
    retryCount?: number; 
}
export interface SubmitParams { 
    currency_pair: string; 
    side: "bid" | "ask"; 
    limitPrice: number; 
    amount: number; 
    timeoutMs: number; 
    retry?: boolean; 
    orderBook?: { bids: [number, number][]; asks: [number, number][] }; 
}

/**
 * Handle actions needed when an exit (sell) fill occurs.
 * @param {string} pair The currency pair.
 * @param {OrderSnapshot} orderSnapshot The order snapshot.
 * @param {number} maxWaitMs The maximum wait time in milliseconds.
 * @param {number} pollIntervalMs The polling interval in milliseconds.
 * @returns {Promise<OrderSnapshot>}
 */
export async function pollFillState(pair: string, orderSnapshot: OrderSnapshot, maxWaitMs: number, pollIntervalMs: number = 3000): Promise<OrderSnapshot> {
    if (!orderSnapshot.orderId) return orderSnapshot;
    const start = Date.now();
    let pollAttempts = 0;
    while (Date.now() - start < maxWaitMs) {
        try {
            const active = await getActiveOrders(pair);
            const still = active[String(orderSnapshot.orderId)];
            if (!still) {
                const hist = await fetchTradeHistory(pair, { count: 200 });
                const windowStart = (orderSnapshot.submittedAt || 0) - SLIPPAGE_TIME_WINDOW_MS;
                const windowEnd = Date.now() + SLIPPAGE_TIME_WINDOW_MS;
                const sideChar = orderSnapshot.side === 'bid' ? 'bid' : 'ask';
                let filledAmt = orderSnapshot.filledAmount || 0;
                let value = (orderSnapshot.avgFillPrice || 0) * filledAmt;
                let usedOrderId = false;
                const matchedTids = new Set<number>();
                let matchedQtySoFar = orderSnapshot.filledAmount || 0;
                for (const f of hist as any[]) {
                    const tsMs = (f.timestamp || 0) * 1000;
                    if (orderSnapshot.orderId && f.order_id && Number(f.order_id) === orderSnapshot.orderId) {
                        usedOrderId = true;
                        if (filledAmt < orderSnapshot.amount) {
                            const add = Math.min(f.amount, orderSnapshot.amount - filledAmt);
                            filledAmt += add; value += add * f.price;
                            if (!orderSnapshot.fills) orderSnapshot.fills = [];
                            orderSnapshot.fills.push({ price: f.price, amount: add, ts: tsMs });
                            const estFillPrice = orderSnapshot.intendedPrice;
                            if (orderSnapshot.side === 'ask') onExitFill(pair, estFillPrice, add);
                        }
                    } else if (!usedOrderId && tsMs >= windowStart && tsMs <= windowEnd && f.side === sideChar) {
                        if (filledAmt < orderSnapshot.amount && !matchedTids.has(f.tid)) {
                            const refPrice = orderSnapshot.intendedPrice;
                            const priceOk = Math.abs(f.price - refPrice) / refPrice <= TOL_PRICE_PCT;
                            const targetRemaining = orderSnapshot.amount - matchedQtySoFar;
                            if (targetRemaining <= 0) break;
                            const qtyTol = Math.min(targetRemaining * TOL_QTY_PCT, ABS_QTY_TOL);
                            const qtyOk = f.amount <= targetRemaining + qtyTol;
                            if (priceOk && qtyOk) {
                                const addRaw = Math.min(f.amount, targetRemaining);
                                const add = Math.max(0, addRaw);
                                if (add > 0) {
                                    matchedQtySoFar += add;
                                    value += add * f.price;
                                    if (!orderSnapshot.fills) orderSnapshot.fills = [];
                                    orderSnapshot.fills.push({ price: f.price, amount: add, ts: tsMs });
                                    matchedTids.add(f.tid);
                                    const estFillPrice = orderSnapshot.intendedPrice;
                                    if (orderSnapshot.side === 'ask') onExitFill(pair, estFillPrice, add);
                                }
                                if (matchedQtySoFar >= orderSnapshot.amount) break;
                            }
                        }
                    }
                }
                if (usedOrderId) logSignal(`fill_match_method=order_id requestId=${orderSnapshot.requestId}`); else logSignal(`fill_match_method=heuristic requestId=${orderSnapshot.requestId}`);
                if (filledAmt > 0) { orderSnapshot.avgFillPrice = value / filledAmt; orderSnapshot.filledAmount = filledAmt; }
                if (filledAmt >= orderSnapshot.amount * 0.999) {
                    orderSnapshot.status = 'FILLED'; 
                    logExecution('Order filled', { 
                        requestId: orderSnapshot.requestId, 
                        orderId: orderSnapshot.orderId, 
                        filledAmt, 
                        avg: orderSnapshot.avgFillPrice, 
                        pnl: (orderSnapshot.avgFillPrice||0) - (orderSnapshot.intendedPrice||0), 
                        win: ((orderSnapshot.avgFillPrice||0) - (orderSnapshot.intendedPrice||0)) >= 0 
                    });
                    if (orderSnapshot.filledAmount && orderSnapshot.avgFillPrice) updateOnFill({ 
                        pair, 
                        side: orderSnapshot.side, 
                        price: orderSnapshot.avgFillPrice, 
                        amount: orderSnapshot.filledAmount, 
                        ts: Date.now(), 
                        matchMethod: 'history' 
                    });
                    if (orderSnapshot.orderId) clearOpenOrderId(pair, orderSnapshot.orderId);
                } else {
                    orderSnapshot.status = 'CANCELLED'; 
                    logTradeError('Order missing from active list; marking CANCELLED', { requestId: orderSnapshot.requestId, orderId: orderSnapshot.orderId, filledAmt });
                    if (orderSnapshot.orderId) clearOpenOrderId(pair, orderSnapshot.orderId);
                }
                return orderSnapshot;
            }
            if (still) {
                const remaining = still.amount; if (remaining != null && orderSnapshot.originalAmount) {
                    const filled = orderSnapshot.originalAmount - remaining; if (filled > 0) {
                        if (!orderSnapshot.filledAmount || filled > orderSnapshot.filledAmount) {
                            if (!orderSnapshot.fills) orderSnapshot.fills = [];
                            const increment = filled - (orderSnapshot.filledAmount || 0);
                            const estFillPrice = orderSnapshot.intendedPrice;
                            orderSnapshot.fills.push({ price: estFillPrice, amount: increment, ts: Date.now() });
                            const totalValue = (orderSnapshot.avgFillPrice || 0) * (orderSnapshot.filledAmount || 0) + increment * estFillPrice;
                            orderSnapshot.filledAmount = filled;
                            orderSnapshot.avgFillPrice = totalValue / orderSnapshot.filledAmount;
                            updateOnFill({ pair, side: orderSnapshot.side, price: estFillPrice, amount: increment, ts: Date.now(), matchMethod: 'active_partial' });
                            if (orderSnapshot.side === 'ask') onExitFill(pair, estFillPrice, increment);
                        }
                        if (orderSnapshot.filledAmount && orderSnapshot.filledAmount < orderSnapshot.originalAmount) orderSnapshot.status = 'PARTIAL'; pollAttempts++;
                    }
                }
            }
        } catch (e: any) { logTradeError('pollFillState error', { error: e.message }); }
    await sleep(pollIntervalMs);
    }
    orderSnapshot.retryCount = (orderSnapshot.retryCount || 0) + pollAttempts; orderSnapshot.status = 'EXPIRED';
    try { if (orderSnapshot.orderId) await cancelOrder({ order_id: orderSnapshot.orderId }); } catch { } logTradeError('Order expired', orderSnapshot); 
    return orderSnapshot;
}

export const computeSlippage = (intendedPrice: number, avgFillPrice?: number): number =>
    intendedPrice === 0 || !avgFillPrice ? 0 : (avgFillPrice - intendedPrice) / intendedPrice;
export interface SubmitRetryParams extends Omit<SubmitParams, 'timeoutMs'> {
    primaryTimeoutMs?: number;
    retryTimeoutMs?: number;
    improvePricePct?: number;
}
/**
 * Submit an order with retry logic.
 * @param {SubmitRetryParams} p The parameters for submitting the order with retry.
 * @returns {Promise<OrderLifecycleSummary>} The summary of the order lifecycle.
 */
export async function submitOrderWithRetry(p: SubmitRetryParams): Promise<OrderLifecycleSummary> {
    const actionSide = p.side === 'bid' ? 'BUY' : 'SELL';
    const requestId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL_MS || 300);
    const POLL_MIN_CYCLES = Number(process.env.POLL_MIN_CYCLES || 3);
    const primaryTimeout = p.primaryTimeoutMs ?? RETRY_TIMEOUT_MS;
    const retryTimeout = p.retryTimeoutMs ?? RETRY_TIMEOUT_MS;
    const improvePct = p.improvePricePct ?? RETRY_PRICE_OFFSET_PCT;
    const CANCEL_MAX = Number(process.env.CANCEL_MAX_RETRIES || 1);
    const BACKOFF = Number(process.env.RETRY_BACKOFF_MS || 300);
    // use shared sleep honoring FAST_CI
    const dryRun = process.env.DRY_RUN === '1';
    async function submit(price: number) {
        if (dryRun) return 'DRYRUN';
        const r = await placeLimitOrder(p.currency_pair, actionSide, price, p.amount);
        return String(r.order_id || '');
    }
    async function pollLoop(snapshot: {
        orderId: string;
        originalAmount: number;
        assumedFilled: number;
        createdAt: number;
        expectedPx: number
    }, timeoutMs: number) {
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
                    const delta = newAssumed - snapshot.assumedFilled; snapshot.assumedFilled = newAssumed;
                    filledQty += delta; avgFillPrice = snapshot.expectedPx;
                }
                if (remaining === 0) status = 'FILLED';
            }
            else {
                const hist = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 200 }) : [];
                const qtyById = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qtyById >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                } else if (qtyById > snapshot.assumedFilled) {
                    const delta = qtyById - snapshot.assumedFilled;
                    snapshot.assumedFilled = qtyById;
                    filledQty += delta;
                    const priceAvg = hist.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount) * Number(h.price), 0) / qtyById || snapshot.expectedPx;
                    avgFillPrice = priceAvg;
                }
            }
            if (i === 1) {
                const hist0 = (privExec as any).trade_history ? await (privExec as any).trade_history({ currency_pair: p.currency_pair, count: 100 }) : [];
                const qty0 = hist0.filter((h: any) => h.order_id === snapshot.orderId).reduce((s: number, h: any) => s + Number(h.amount), 0);
                if (qty0 > snapshot.assumedFilled) {
                    const delta0 = qty0 - snapshot.assumedFilled;
                    snapshot.assumedFilled = qty0;
                    filledQty += delta0;
                }
                if (qty0 >= snapshot.originalAmount) {
                    filledQty = snapshot.originalAmount;
                    status = 'FILLED';
                }
                logSignal('[POLL0]', { orderId: snapshot.orderId, openCount: open.length, openAmtById: found ? Number(found.amount) : null, histQtyById: qty0 });
            }
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
            if (!dryRun && status !== 'FILLED') {
                const maxSlip = Number(process.env.RISK_MAX_SLIPPAGE_PCT || '0.005');
                const off = Number(process.env.REPRICE_OFFSET_PCT || '0.001');
                const maxReprice = Number(process.env.REPRICE_MAX_ATTEMPTS || '2');
                const bestAsk = snapshot.expectedPx * (actionSide === 'BUY' ? 1 : 1.001);
                const bestBid = snapshot.expectedPx * (actionSide === 'BUY' ? 0.999 : 1);
                const refPx = avgFillPrice > 0 ? avgFillPrice : (actionSide === 'BUY' ? bestAsk : bestBid);
                const slip = Math.abs((refPx - snapshot.expectedPx) / snapshot.expectedPx);
                if (slip > maxSlip) {
                    if (repriceAttempts < maxReprice) {
                        if (snapshot.orderId !== 'DRYRUN') {
                            try { await cancelOrder({ order_id: snapshot.orderId as any }); } catch { }
                        }
                        const prevLimit = (actionSide === 'BUY' ? Math.max(refPx, snapshot.expectedPx) : Math.min(refPx, snapshot.expectedPx));
                        let newLimit = prevLimit;
                        if (actionSide === 'BUY') newLimit = Math.min(bestAsk, prevLimit * (1 + off)); else newLimit = Math.max(bestBid, prevLimit * (1 - off));
                        const newId = await submit(newLimit);
                        logInfo('[SLIPPAGE] reprice', { requestId: snapshot.orderId, orderId: newId, repriceAttempts: repriceAttempts + 1, slip, newLimit });
                        snapshot.orderId = newId;
                        repriceAttempts++;
                        continue;
                    } else {
                        status = 'FILLED';
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
    if (dryRun) {
        logInfo('[DRYRUN] submitOrderWithRetry simulate fill', { requestId, side: actionSide, qty: p.amount, price: p.limitPrice });
        return ({ requestId, side: actionSide as any, intendedQty: p.amount, filledQty: p.amount, avgExpectedPrice: p.limitPrice, avgFillPrice: p.limitPrice, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 1, repriceAttempts: 0 } as unknown) as OrderLifecycleSummary;
    }
    const submitOrderId = await submit(p.limitPrice);
    if (!submitOrderId) {
        logTradeError('Missing orderId', { requestId });
        return ({ requestId, side: actionSide, intendedQty: p.amount, filledQty: 0, avgExpectedPrice: p.limitPrice, avgFillPrice: 0, slippagePct: 0, durationMs: 0, submitRetryCount: 0, pollRetryCount: 0, cancelRetryCount: 0, nonceRetryCount: 0, totalRetryCount: 0, filledCount: 0 } as unknown) as OrderLifecycleSummary;
    }
    logSignal('[SUBMIT]', { requestId, orderId: submitOrderId, side: actionSide, price: p.limitPrice, amount: p.amount });
    const expectedPx = p.limitPrice;
    const snapshot1 = { requestId, orderId: submitOrderId, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx };
    const nonce1 = getAndResetLastRequestNonceRetries(); const poll1: any = await pollLoop(snapshot1, primaryTimeout);
    let submitRetryCount = 0, cancelRetryCount = 0, pollRetryCount = poll1.pollAttempts, filledQty = poll1.filledQty, avgFillPrice = filledQty > 0 ? p.limitPrice : 0, nonceRetryCount = nonce1, improvedPrice = p.limitPrice, secondStats: any = null, repriceAttemptsTotal = poll1.repriceAttempts || 0;
    if (poll1.status !== 'FILLED') {
        submitRetryCount = 1;
        const improveFactor = p.side === 'ask' ? (1 - improvePct) : (1 + improvePct);
        improvedPrice = p.limitPrice * improveFactor;
        for (let c = 0; c < CANCEL_MAX; c++) {
            try {
                await cancelOrder({ order_id: submitOrderId as any }); break;
            } catch {
                cancelRetryCount++;
                await sleep(BACKOFF);
            }
        }
        const submitOrderId2 = await submit(improvedPrice);
        const nonce2 = getAndResetLastRequestNonceRetries();
        nonceRetryCount += nonce2; logSignal('[SUBMIT]', { requestId, orderId: submitOrderId2, improved: true, price: improvedPrice, amount: p.amount });
        const snapshot2 = { requestId, orderId: submitOrderId2, originalAmount: p.amount, assumedFilled: 0, createdAt: Date.now(), expectedPx }; secondStats = await pollLoop(snapshot2, retryTimeout);
        pollRetryCount += secondStats.pollAttempts; repriceAttemptsTotal += secondStats.repriceAttempts || 0;
        if (secondStats.filledQty > 0) {
            filledQty = secondStats.filledQty;
            avgFillPrice = improvedPrice;
        }
    }
    const durationMs = Date.now() - snapshot1.createdAt;
    const slippagePct = filledQty > 0 ? (avgFillPrice - p.limitPrice) / p.limitPrice : 0;
    const totalRetryCount = submitRetryCount + pollRetryCount + cancelRetryCount + nonceRetryCount;
    const summary = ({ requestId, side: actionSide as any, intendedQty: p.amount, filledQty, avgExpectedPrice: p.limitPrice, avgFillPrice, slippagePct, durationMs, submitRetryCount, pollRetryCount, cancelRetryCount, nonceRetryCount, totalRetryCount, filledCount: filledQty > 0 ? 1 : 0, repriceAttempts: repriceAttemptsTotal } as unknown) as OrderLifecycleSummary; logInfo('[ORDER]', { requestId, orderId: submitOrderId, side: actionSide, filledQty, totalRetryCount, pollRetryCount, submitRetryCount, cancelRetryCount, nonceRetryCount, repriceAttempts: repriceAttemptsTotal, durationMs });
    return summary;
}

let cachedToday = new Date().toISOString().slice(0, 10); let lastDateCheck = Date.now(); function getToday() { const now = Date.now(); if (now - lastDateCheck > 3600000) { const newToday = new Date().toISOString().slice(0, 10); if (newToday !== cachedToday) cachedToday = newToday; lastDateCheck = now; } return cachedToday; }
export function onExitFill(pair: string, fillPrice: number, fillQty: number) {
    const today = getToday();
    const pos = loadPosition(pair);
    if (!pos) return;
    const preAvg = pos.avgPrice;
    const realized = (fillPrice - preAvg) * fillQty;
    appendFillPnl(today, realized, pair);
    // Feature logging (exit)
        try {
        const rsiPeriod = Number(process.env.RSI_PERIOD || 14);
        const priceSeries = getPriceSeries(Math.max(200, rsiPeriod + 2));
        const rsi = calculateRsi(priceSeries, rsiPeriod);
        const smaShort = calculateSma(priceSeries, Number(process.env.SMA_SHORT||9) || 9) as any;
        const smaLong = calculateSma(priceSeries, Number(process.env.SMA_LONG||26) || 26) as any;
            logFeatureSample({ ts: Date.now(), pair, side: 'ask', rsi: rsi ?? undefined, sma_short: smaShort ?? undefined, sma_long: smaLong ?? undefined, price: fillPrice, qty: fillQty, pnl: realized, win: realized >= 0, position: { qty: pos.qty, avgPrice: pos.avgPrice, side: (pos as any).side||'long' } });
    } catch {}
    // Reduce qty without changing avgPrice for leftover
    const newQty = Math.max(0, pos.qty - fillQty);
    pos.qty = newQty;
    if (newQty === 0) { pos.avgPrice = 0; (pos as any).dcaRemainder = 0; }
    savePosition(pos);
}
