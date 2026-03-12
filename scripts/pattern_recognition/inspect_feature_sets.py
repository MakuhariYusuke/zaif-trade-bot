from ztb.features.feature_set_config import FeatureSetConfig
from ztb.features.registry import FeatureRegistry
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

FeatureRegistry.initialize()
all_feats = FeatureRegistry.list()
logger.info("total registered features: %d", len(all_feats))
for set_name in FeatureSetConfig.FEATURE_SETS.keys():
    cfg = FeatureSetConfig()
    cfg.set_feature_set(set_name)
    excluded = cfg.get_excluded_features()
    feats = [f for f in all_feats if not any(ex in f for ex in excluded)]
    logger.info("%s -> %d", set_name, len(feats))
    logger.info("  sample: %s", feats[:10])
