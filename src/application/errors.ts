export type ErrorCode =
  | 'CIRCUIT_OPEN'
  | 'RATE_LIMITED'
  | 'API_ERROR'
  | 'CACHE_ERROR'
  | 'NETWORK'
  | 'NONCE'
  | 'SIGNATURE'
  | 'UNKNOWN';

export function normalizeErrorCode(err: any): ErrorCode {
  const code = (err?.code || err?.cause?.code || '').toString().toUpperCase();
  if (!code) return 'UNKNOWN';
  switch (code) {
    case 'CIRCUIT_OPEN':
    case 'RATE_LIMITED':
    case 'API_ERROR':
    case 'CACHE_ERROR':
    case 'NETWORK':
    case 'NONCE':
    case 'SIGNATURE':
      return code as ErrorCode;
    default:
      if (/ECONNRESET|ETIMEDOUT|ENETUNREACH|ECONNREFUSED|EAI_AGAIN/.test(code)) return 'NETWORK';
      return 'UNKNOWN';
  }
}

export function buildErrorEventMeta(base: {
  requestId?: string | null;
  pair?: string | null;
  side?: string | null;
  amount?: number | null;
  price?: number | null;
}, err: any) {
  const code = normalizeErrorCode(err);
  return {
    requestId: base.requestId ?? undefined,
    pair: base.pair ?? undefined,
    side: base.side as any,
    amount: base.amount ?? undefined,
    price: base.price ?? undefined,
    cause: { code, message: err?.message, detail: err?.cause || undefined },
  };
}
