import { createCoincheckPrivate } from '../coincheck-private';
import type { PrivateApi } from '../../types/private';

/**
 * Creates a real Coincheck private API instance using environment variables for credentials.
 * Requires COINCHECK_API_KEY and COINCHECK_API_SECRET to be set.
 * @throws Will throw an error if the API key or secret is missing.
 * @returns {PrivateApi} An instance of the Coincheck private API.
 */
export function createPrivateReal(): PrivateApi {
    const key = process.env.COINCHECK_API_KEY || '';
    const secret = process.env.COINCHECK_API_SECRET || '';
    if (!key || !secret) throw new Error('Coincheck API key/secret missing (set COINCHECK_API_KEY / COINCHECK_API_SECRET)');
    return createCoincheckPrivate(key, secret);
}

/**
 * Creates a mock private API instance for testing purposes.
 * This mock simulates order placement, fills, cancellations, and maintains balances.
 * It reads and writes state to a JSON file specified by the MOCK_ORDERS_PATH environment variable.
 * Balances can be initialized via the MOCK_BALANCES_JSON environment variable.
 * @example
 * process.env.MOCK_BALANCES_JSON = JSON.stringify({ jpy: 500000, btc: 0.05 });
 */
export function createPrivateMock() {
    const zaif = require('../zaif-private-mock');
    return zaif.createPrivateMock();
}
