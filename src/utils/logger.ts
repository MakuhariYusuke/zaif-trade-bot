// New: Thin interface and default logger instance (backward-compatible)
export interface Logger {
  debug(msg: string, meta?: any): void;
  info(msg: string, meta?: any): void;
  warn(msg: string, meta?: any): void;
  error(msg: string, meta?: any): void;
  log?(level: "TRACE"|"DEBUG"|"INFO"|"WARN"|"ERROR"|"FATAL", category: string, msg: string, meta?: any): void;
}
type Level = "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "FATAL";

let context: Record<string, any> = {};
let redactKeys = new Set<string>([
  'apiKey','apikey','key','secret','passphrase','signature','token','refreshToken',
  'privateKey','seed','mnemonic','accountId','authorization','auth','password',
  // event payload common
  'detail','raw'
]);
const onceFlags = new Set<string>();

/**
 * Converts a log level string to its corresponding numeric value.
 * The levels are defined as follows:
 * - "TRACE" = 0
 * - "DEBUG" = 10
 * - "INFO"  = 20
 * - "WARN"  = 30
 * - "ERROR" = 40
 * - "FATAL" = 50
 * @param {Level} l - The log level string to convert.
 * @returns {number} The numeric value corresponding to the log level.
 */
function levelValue(l: Level): number {
  switch (l) {
    case "TRACE": return 0;
    case "DEBUG": return 10;
    case "INFO": return 20;
    case "WARN": return 30;
    case "ERROR": return 40;
    case "FATAL": return 50;
  }
}

/**
 * Retrieves the current log level threshold from the environment variable LOG_LEVEL.
 * If LOG_LEVEL is not set or invalid, defaults to "INFO".
 * The threshold is used to filter out log messages below the specified level.
 * @returns {number} The numeric value of the current log level threshold.
 */
function currentThreshold(): number {
  const validLevels: Level[] = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"];
  const env = (process.env.LOG_LEVEL || "INFO").toUpperCase();
  const lvl: Level = validLevels.includes(env as Level) ? (env as Level) : "INFO";
  return levelValue(lvl);
}
function ts(): string { return new Date().toISOString(); }

function redactMeta(meta: any): any {
  if (meta == null) return meta;
  if (Array.isArray(meta)) return meta.map(redactMeta);
  if (typeof meta !== 'object') return meta;
  const out: any = Array.isArray(meta) ? [] : {};
  let redacted = false;
  for (const [k, v] of Object.entries(meta)) {
    const keyLower = k.toLowerCase();
    if (redactKeys.has(keyLower)) { out[k] = '***'; redacted = true; continue; }
    // nested common sensitive paths
    if (keyLower === 'headers' && typeof v === 'object') {
      out[k] = { ...v, Authorization: v && (v as any).Authorization ? '***' : (v as any)?.Authorization };
      redacted = true;
      continue;
    }
    out[k] = redactMeta(v);
  }
  if (redacted && typeof out === 'object' && !Array.isArray(out)) (out as any).redacted = true;
  return out;
}

/**
 * Emits a log message if its level is at or above the current threshold.
 * Supports both JSON and plain text output formats based on the LOG_JSON environment variable.
 * Includes context information if set.
 * @param {Level} level - The severity level of the log message.
 * @param {string} message - The main log message.
 * @param {any[]} args - Additional arguments to include in the log.
 */
function emit2(level: Level, category: string | undefined, message: string, meta?: any) {
  // Suppress lower levels only when in TEST_MODE with verbose logging (DEBUG/TRACE or unset)
  const lvlEnv = (process.env.LOG_LEVEL || '').toUpperCase();
  const verbose = lvlEnv === 'DEBUG' || lvlEnv === 'TRACE' || !lvlEnv;
  if (process.env.TEST_MODE === '1' && verbose && levelValue(level) < 40) return;
  // Sampling for TRACE/DEBUG (default 1/10). Disable in TEST_MODE.
  const isUnitTest = !!process.env.VITEST_WORKER_ID || process.env.NODE_ENV === 'test';
  if (process.env.TEST_MODE !== '1' && !isUnitTest && (level === 'TRACE' || level === 'DEBUG')) {
    const n = Math.max(1, Number(process.env.DEBUG_SAMPLING || '10'));
    if (Math.floor(Math.random() * n) !== 0) return;
  }
  if (levelValue(level) < currentThreshold()) return;
  const json = process.env.LOG_JSON === "1";
  const redMeta = redactMeta(meta);
  if (json) {
    const entry = {
      ts: ts(),
      level,
      category,
      message,
      data: redMeta != null ? [redMeta] : [],
      ...context
    };
    const line = JSON.stringify(entry);
  if (level === "ERROR" || level === "FATAL") console.error(line);
  else if (level === "WARN") console.warn(line);
  else console.log(line);
    return;
  }

  let ctxStr = "";
  if (context && Object.keys(context).length > 0) {
    ctxStr = " " + Object.entries(context).map(([k, v]) => `[${k}=${String(v)}]`).join(" ");
  }
  const prefix = `[${level}]${category ? `[${category}]` : ''}`;
  const line = `${prefix} ${message}${ctxStr}`;

  if (level === "ERROR" || level === "FATAL") console.error(line, ...(redMeta != null ? [redMeta] : []));
  else if (level === "WARN") console.warn(line, ...(redMeta != null ? [redMeta] : []));
  else console.log(line, ...(redMeta != null ? [redMeta] : []));
}

// Back-compat emit without category
function emit(level: Level, message: string, ...args: any[]) {
  const meta = args && args.length ? (args.length === 1 ? args[0] : args) : undefined;
  emit2(level, undefined, message, meta);
}
/**
 * Merges the provided context with the existing logger context.
 * Existing keys will be overwritten, but other keys are preserved.
 * The resulting context is a shallow copy of the previous context with overwritten keys.
 * @param {Record<string, any>} ctx - An object containing key-value pairs to add to the logger context.
 */
function setLoggerContext(ctx: Record<string, any>) {
  context = { ...context, ...ctx };
}

export function addLoggerRedactFields(keys: string[]) {
  for (const k of keys) redactKeys.add(String(k).toLowerCase());
}

export function warnOnce(id: string, message: string, meta?: any){
  if (onceFlags.has(id)) return;
  onceFlags.add(id);
  emit2('WARN', 'CONFIG', message, meta);
}


/** 
 * Clears the entire logger context if no keys are provided.
 * If keys are provided, only those keys are removed from the context.
 * @param {string[]} [keys] - Optional array of keys to remove from the context. If omitted, clears all context.
 */
function clearLoggerContext(keys?: string[]) {
  if (!keys) { context = {}; return; }
  for (const k of keys) delete context[k];
}

/** Log a trace message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logTrace(message: string, ...args: any[]) { emit("TRACE", message, ...args); }
/** Log a debug message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logDebug(message: string, ...args: any[]) { emit("DEBUG", message, ...args); }
/** Log an informational message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logInfo(message: string, ...args: any[]) { emit("INFO", message, ...args); }
/** Log a warning message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logWarn(message: string, ...args: any[]) { emit("WARN", message, ...args); }
/** Log an error message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logError(message: string, ...args: any[]) { emit("ERROR", message, ...args); }
/** Log a fatal error message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logFatal(message: string, ...args: any[]) { emit("FATAL", message, ...args); }
/** Log an assertion message (treated as ERROR level). @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logAssert(message: string, ...args: any[]) { emit("ERROR", `[ASSERT] ${message}`, ...args); }

// New: category-aware API
export function log(level: Level, category: string, message: string, meta?: any) {
  emit2(level, category, message, meta);
}

// Default DI-friendly logger implementation that forwards to emit2
export const logger: Logger = {
  debug: (msg: string, meta?: any) => emit2("DEBUG", undefined, msg, meta),
  info: (msg: string, meta?: any) => emit2("INFO", undefined, msg, meta),
  warn: (msg: string, meta?: any) => emit2("WARN", undefined, msg, meta),
  error: (msg: string, meta?: any) => emit2("ERROR", undefined, msg, meta),
  log: (level: any, category: string, msg: string, meta?: any) => emit2(level, category, msg, meta),
};