import dotenv from 'dotenv';
dotenv.config();
import { strategyOnce } from '../index';

(async()=>{
  const pair = process.env.PAIR || 'btc_jpy';
  const EXECUTE = process.env.DRY_RUN !== '1';
  await strategyOnce(pair, EXECUTE);
  process.exit(0);
})();
