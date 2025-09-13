import dotenv from 'dotenv';
dotenv.config();
import { setNonceBase, getLastNonce } from '../utils/signer';
import { loadAppConfig, restoreNonce, persistNonce } from '../utils/config';

// Simple CLI to bump nonce baseline quickly when server says "nonce out of range"
const app = loadAppConfig();
restoreNonce(app.nonceStorePath);
const target = Number(process.argv[2]||0);
if (Number.isFinite(target) && target>0){
  setNonceBase(target);
  persistNonce(app.nonceStorePath, target);
  console.log('[NONCE] bumped to', target);
} else {
  console.log('[NONCE] current', getLastNonce());
  console.log('Usage: npm run nonce:bump -- <newNonce>');
}
