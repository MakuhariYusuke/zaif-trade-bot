import crypto from "crypto";

// Zaif private API requires HMAC-SHA512 over the POST body using secret, sent in Sign header.
export function signBody(body: string, secret: string): string {
    return crypto.createHmac("sha512", secret).update(body).digest("hex");
}

let lastNonce = 0;
let lastMs = 0;
let subCounter = 0;

export function setNonceBase(base: number) {
    lastNonce = Math.max(lastNonce, base);
}

// Monotonic nonce (integer). If multiple calls occur in the same ms, it auto-increments.
export function createNonce(): number {
    const now = Date.now();
    // ensure strictly increasing sequence even if lastNonce is bigger than now
    lastNonce = Math.max(lastNonce + 1, now);
    return lastNonce;
}

// Fractional nonce: "ms.subCounter" ensuring strict monotonic string order
export function createFlexibleNonce(): string {
    const ms = Date.now();
    subCounter = ms === lastMs ? subCounter + 1 : 0;
    lastMs = ms;

    const numericBase = ms * 1000 + subCounter;
    lastNonce = Math.max(lastNonce, numericBase);
    return `${ms}.${subCounter}`;
}

export function getLastNonce(): number {
    return lastNonce;
}
