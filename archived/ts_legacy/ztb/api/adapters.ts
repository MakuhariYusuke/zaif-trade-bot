import { PrivateApi } from "../types/private";

// Exchange adapter contract to unify access points.
export interface ExchangeAdapter {
    private: PrivateApi;
    // public endpoints could be added later, keeping compatibility for now.
}

/**
 * Lazy import per exchange to avoid bundling everything and keep deps minimal.
 * @param {string} exchange Exchange identifier
 * @returns {{ createPrivateReal: () => PrivateApi; createPrivateMock: () => PrivateApi }} Module with createPrivateReal and createPrivateMock functions
 */
function resolveFactories(exchange: string): { createPrivateReal: () => PrivateApi; createPrivateMock: () => PrivateApi } {
    switch (exchange) {
        case 'coincheck':
            return require('./private/coincheck');
        case 'paper':
            return require('./private/paper');
        case 'zaif':
        default:
            return require('./private/zaif');
    }
}

/**
 * Creates a PrivateApi instance based on environment configuration.
 * Uses real API if USE_PRIVATE_MOCK is not set to '1', otherwise uses a mock implementation.
 * The exchange is determined by the EXCHANGE environment variable (default: 'zaif').
 * @returns {PrivateApi} An instance of the PrivateApi interface.
 */
export function createPrivateApi(): PrivateApi {
    const ex = (process.env.EXCHANGE || 'zaif').toLowerCase();
    const { createPrivateReal, createPrivateMock } = resolveFactories(ex);
    console.log(`[EXCHANGE] ${ex}`);
    const useMockExplicit = process.env.USE_PRIVATE_MOCK === '1';
    const missingCoincheckSecrets = ex === 'coincheck'
        && (!process.env.COINCHECK_API_KEY || !process.env.COINCHECK_API_SECRET);

    if (useMockExplicit || missingCoincheckSecrets) {
        if (missingCoincheckSecrets) {
            console.log('[BACKEND] MOCK (coincheck secrets missing)');
        } else {
            console.log('[BACKEND] MOCK');
        }
        return createPrivateMock();
    }
    console.log('[BACKEND] REAL');
    return createPrivateReal();
}