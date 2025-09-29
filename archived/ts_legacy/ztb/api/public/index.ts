import * as coincheck from './coincheck';
import * as zaif from './zaif';

function isCC() { return (process.env.EXCHANGE || 'zaif').toLowerCase() === 'coincheck'; }

export async function getTicker(pair: string) { return isCC() ? coincheck.getTicker(pair) : zaif.getTicker(pair); }
export async function getOrderBook(pair: string) { return isCC() ? coincheck.getOrderBook(pair) : zaif.getOrderBook(pair); }
export async function getTrades(pair: string) { return isCC() ? coincheck.getTrades(pair) : zaif.getTrades(pair); }
