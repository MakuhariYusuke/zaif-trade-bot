// Deprecated shim: re-export contracts from src/contracts
// Emit a once-per-process warning to guide migration
import { warnOnce } from "../utils/logger";
warnOnce('contracts-import-deprecated', 'Importing from src/types/contracts is deprecated. Use "@contracts" (tsconfig paths) or "src/contracts" instead.');

export type {
  PositionState,
  PositionStore,
  RiskError,
  TrailingAction,
  ClampedIntent,
  RiskManager,
} from "../contracts";
