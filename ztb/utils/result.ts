export interface ApiError { code: string; message: string; cause?: unknown }
export type Ok<T> = { ok: true; value: T }
export type Err<E = ApiError> = { ok: false; error: E }
export type Result<T, E = ApiError> = Ok<T> | Err<E>
export const ok = <T>(v: T): Ok<T> => ({ ok: true, value: v })
export const err = (code: string, message: string, cause?: unknown): Err => ({ ok: false, error: { code, message, cause } })