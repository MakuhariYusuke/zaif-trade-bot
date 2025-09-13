import { logInfo, logError, logWarn } from "../../utils/logger";
import { PrivateApi } from "../../types/private";
import dotenv from "dotenv";
dotenv.config();
import { createPrivateApi } from "../../api/adapters";

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
    })();
}
