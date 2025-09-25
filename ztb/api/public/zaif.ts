import axios from 'axios';
import { BaseExchangePublic } from '../base-public';

const ZAIF_BASE = 'https://api.zaif.jp/api/1';

class ZaifPublic extends BaseExchangePublic {
  async getTicker(pair: string) {
    const r = await axios.get(`${ZAIF_BASE}/ticker/${pair}`);
    return r.data;
  }
  async getOrderBook(pair: string) {
    const r = await axios.get(`${ZAIF_BASE}/depth/${pair}`);
    return r.data;
  }
  async getTrades(pair: string) {
    const r = await axios.get(`${ZAIF_BASE}/trades/${pair}`);
    return r.data;
  }
}

const client = new ZaifPublic();
export async function getTicker(pair: string) { return client.getTicker(pair); }
export async function getOrderBook(pair: string) { return client.getOrderBook(pair); }
export async function getTrades(pair: string) { return client.getTrades(pair); }
