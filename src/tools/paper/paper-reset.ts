import fs from 'fs';
import path from 'path';
import os from 'os';
import dotenv from 'dotenv';
dotenv.config();

function today(){ return new Date().toISOString().slice(0,10); }

(async ()=>{
  const store = process.env.PAPER_STORE || path.join(os.tmpdir(), 'paper-trader.json');
  try{
    if (fs.existsSync(store)) { fs.unlinkSync(store); console.log('[PAPER] store removed', store); }
    else console.log('[PAPER] store not found (ok)', store);
  }catch(e:any){ console.error('[PAPER] store remove error', e?.message||e); process.exitCode=1; }
  try{
    const logsDir = process.env.STATS_DIR || path.resolve(process.cwd(), 'logs');
    const snap = path.join(logsDir, `.stats-snapshot-${today()}.json`);
    if (fs.existsSync(snap)) { fs.unlinkSync(snap); console.log('[PAPER] stats snapshot cleared', snap); }
  }catch{}
})();
