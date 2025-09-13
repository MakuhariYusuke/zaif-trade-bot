import { logInfo, logError, logWarn } from "../../utils/logger";
import { PrivateApi } from "../../types/private";
import dotenv from "dotenv";
dotenv.config();
import { createPrivateApi } from "../../api/adapters";
import { getTicker, getOrderBook } from "../../api/public";

export async function printPrivateHealth(api: PrivateApi) {
    try {
        const res = await (api as any).healthCheck();
    const success = (res && (res.ok === true || res.isAllowed === true || res.success === 1));
    if (success) logInfo("Private API health OK", res.value || res.return || res);
    else logError("Private API health NOT OK", res.error || res);
    } catch (e: any) {
        logError("healthCheck threw", { message: e?.message });
    }
}

async function printPublicHealth(pair: string){
    try {
        const t = await getTicker(pair);
        const ob: any = await getOrderBook(pair);
        const bestBid = Number((ob?.bids?.[0]?.[0]) || 0);
        const bestAsk = Number((ob?.asks?.[0]?.[0]) || 0);
        logInfo("Public API OK", { pair, last: (t as any)?.last || (t as any)?.last_price || null, bestBid, bestAsk });
    } catch (e:any){
        logWarn('Public API error', e?.message||e);
    }
}

// CLI entrypoint
if (require.main === module) {
    (async () => {
        // If real keysが無く MOCK 指定もないなら、自動でモックに切り替え
        if (!process.env.ZAIF_API_KEY || !process.env.ZAIF_API_SECRET) {
            if (process.env.USE_PRIVATE_MOCK !== '1') {
                logWarn('APIキー未設定のため USE_PRIVATE_MOCK=1 を自動適用します');
                process.env.USE_PRIVATE_MOCK = '1';
            }
        }
        let api: PrivateApi | undefined;
        try {
            api = createPrivateApi();
        } catch (e:any) {
            logError('createPrivateApi failed', e?.message || e);
            process.exit(1);
        }
        if (!(api as any).healthCheck) {
            logWarn('healthCheck 未実装の PrivateApi です');
            process.exit(0);
        }
    await printPrivateHealth(api);
    await printPublicHealth(process.env.PAIR || 'btc_jpy');
    })();
}
