import { createCoincheckPrivate } from '../coincheck-private';
import type { PrivateApi } from '../../types/private';

export function createPrivateReal(): PrivateApi {
  const key = process.env.COINCHECK_API_KEY || '';
  const secret = process.env.COINCHECK_API_SECRET || '';
  if (!key || !secret) throw new Error('Coincheck API key/secret missing (set COINCHECK_API_KEY / COINCHECK_API_SECRET)');
  return createCoincheckPrivate(key, secret);
}

export function createPrivateMock(): PrivateApi {
  const zaif = require('../zaif-private-mock');
  return zaif.createPrivateMock();
}
