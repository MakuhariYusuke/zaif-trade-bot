import axios from 'axios';

const CC_BASE = 'https://coincheck.com/api';

export async function getTicker(_pair: string) {
  const r = await axios.get(`${CC_BASE}/ticker`);
  return r.data;
}

export async function getOrderBook(_pair: string) {
  const r = await axios.get(`${CC_BASE}/order_books`);
  const d = r.data || {};
  const map2 = (arr: any[] = []) => (arr || []).map((x: any) => [Number(x[0]), Number(x[1])]);
  return { asks: map2(d.asks), bids: map2(d.bids) };
}

export async function getTrades(pair: string) {
  const url = `${CC_BASE}/trades?pair=${encodeURIComponent(pair)}`;
  const r = await axios.get(url);
  const arr = r.data || [];
  return (Array.isArray(arr) ? arr : []).map((t: any) => ({
    tid: Number(t.id),
    price: Number(t.rate),
    amount: Number(t.amount),
    date: Math.floor(new Date(t.created_at).getTime() / 1000),
    trade_type: t.order_type === 'buy' ? 'bid' : 'ask',
  }));
}
