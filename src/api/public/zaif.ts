import axios from 'axios';

const ZAIF_BASE = 'https://api.zaif.jp/api/1';

export async function getTicker(pair: string) {
  const r = await axios.get(`${ZAIF_BASE}/ticker/${pair}`);
  return r.data;
}

export async function getOrderBook(pair: string) {
  const r = await axios.get(`${ZAIF_BASE}/depth/${pair}`);
  return r.data;
}

export async function getTrades(pair: string) {
  const r = await axios.get(`${ZAIF_BASE}/trades/${pair}`);
  return r.data;
}
