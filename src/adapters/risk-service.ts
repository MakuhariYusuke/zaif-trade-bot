/**
 * @deprecated このファイルは次メジャーで完全削除予定です。
 * 以前は core/risk への再エクスポートを行っていましたが、
 * 利用を早期に検知するため実行時例外を投げる最小スタブに置き換えています。
 * import している箇所は core/risk 及び adapters/risk-config を直接参照してください。
 */
export function createServiceRiskManager(): never { throw new Error('deprecated: import CoreRiskManager from core/risk directly'); }
