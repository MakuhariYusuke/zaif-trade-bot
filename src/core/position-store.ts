// Pure core: delegate to adapter-backed FS implementation while keeping API stable
export type { StoredPosition, FillEvent } from "../adapters/position-store-fs";
export {
  loadPosition,
  savePosition,
  removePosition,
  updatePositionFields,
  addOpenOrderId,
  clearOpenOrderId,
  updateOnFill,
  CorePositionStore
} from "../adapters/position-store-fs";
export { createServicePositionStore as createCorePositionStore } from "../adapters/position-store-fs";
