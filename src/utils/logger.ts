type Level = "TRACE" | "DEBUG" | "INFO" | "WARN" | "ERROR" | "FATAL";

let context: Record<string, any> = {};

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

/**
 * Emits a log message if its level is at or above the current threshold.
 * Supports both JSON and plain text output formats based on the LOG_JSON environment variable.
 * Includes context information if set.
 * @param {Level} level - The severity level of the log message.
 * @param {string} message - The main log message.
 * @param {any[]} args - Additional arguments to include in the log.
 */
function emit(level: Level, message: string, args: any[]) {
  // Suppress lower levels when in TEST_MODE
  if (process.env.TEST_MODE === '1' && levelValue(level) < 40) return;
  if (levelValue(level) < currentThreshold()) return;
  const json = process.env.LOG_JSON === "1";
  if (json) {
    const entry = {
      ts: ts(),
      level,
      message,
      data: args && args.length ? args : [],
      ...context
    };
    const line = JSON.stringify(entry);
    if (level === "ERROR" || level === "FATAL") console.error(line);
    else if (level === "WARN") console.warn(line);
    else if (level === "TRACE") console.debug(line);
    else console.log(line);
    return;
  }

  let ctxStr = "";
  if (context && Object.keys(context).length > 0) {
    ctxStr = " " + Object.entries(context).map(([k, v]) => `[${k}=${String(v)}]`).join(" ");
  }
  const prefix = `[${level}]`;
  const line = `${prefix} ${message}${ctxStr}`;

  if (level === "ERROR" || level === "FATAL") console.error(line, ...args);
  else if (level === "WARN") console.warn(line, ...args);
  else if (level === "TRACE") console.debug(line, ...args);
  else console.log(line, ...args);
}
/**
 * Merges the provided context with the existing logger context.
 * Existing keys will be overwritten, but other keys are preserved.
 * The resulting context is a shallow copy of the previous context with overwritten keys.
 * @param {Record<string, any>} ctx - An object containing key-value pairs to add to the logger context.
 */
export function setLoggerContext(ctx: Record<string, any>) {
  context = { ...context, ...ctx };
}


/** 
 * Clears the entire logger context if no keys are provided.
 * If keys are provided, only those keys are removed from the context.
 * @param {string[]} [keys] - Optional array of keys to remove from the context. If omitted, clears all context.
 */
export function clearLoggerContext(keys?: string[]) {
  if (!keys) { context = {}; return; }
  for (const k of keys) delete context[k];
}

/** Log a trace message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logTrace(message: string, ...args: any[]) { emit("TRACE", message, args); }
/** Log a debug message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logDebug(message: string, ...args: any[]) { emit("DEBUG", message, args); }
/** Log an informational message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logInfo(message: string, ...args: any[]) { emit("INFO", message, args); }
/** Log a warning message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logWarn(message: string, ...args: any[]) { emit("WARN", message, args); }
/** Log an error message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logError(message: string, ...args: any[]) { emit("ERROR", message, args); }
/** Log a fatal error message. @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logFatal(message: string, ...args: any[]) { emit("FATAL", message, args); }
/** Log an assertion message (treated as ERROR level). @param {string} message - The message to log. @param {...any} args - Additional arguments to log. */
export function logAssert(message: string, ...args: any[]) { emit("ERROR", `[ASSERT] ${message}`, args); }