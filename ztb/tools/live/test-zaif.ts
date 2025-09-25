import dotenv from 'dotenv';
dotenv.config();
import { createPrivateApi } from '../../api/adapters';
import { logInfo, logWarn } from '../../utils/logger';

(async () => {
    process.env.EXCHANGE = 'zaif';
    const api: any = createPrivateApi();
    logInfo('[TEST] Zaif start');
    try {
        const h = await (api.healthCheck ? api.healthCheck() : api.get_info2());
        logInfo('[TEST] HealthOrInfo', h);
    } catch (e: any) { logWarn('[TEST] health/info failed', e?.message || e); }
    try {
        const info = await api.get_info2();
        logInfo('[TEST] get_info2', info);
    } catch (e: any) { logWarn('[TEST] get_info2 failed', e?.message || e); }
})();
