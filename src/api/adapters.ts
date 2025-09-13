import { PrivateApi } from "../types/private";

/**
 * Lazy import per exchange to avoid bundling everything and keep deps minimal.
 * @param {string} exchange Exchange identifier
 * @returns {{ createPrivateReal: () => PrivateApi; createPrivateMock: () => PrivateApi }} Module with createPrivateReal and createPrivateMock functions
 */
function resolveFactories(exchange: string): { createPrivateReal: () => PrivateApi; createPrivateMock: () => PrivateApi } {
    switch (exchange) {
        case 'coincheck':
            return require('./exchanges/coincheck');
        case 'paper':
            return require('./exchanges/paper');
        case 'zaif':
        default:
            return require('./exchanges/zaif');
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
    if (process.env.USE_PRIVATE_MOCK === '1') {
        console.log('[BACKEND] MOCK');
        return createPrivateMock();
    }
    console.log('[BACKEND] REAL');
    return createPrivateReal();
}