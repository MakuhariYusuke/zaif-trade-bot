

import pandas as pd

def map_trained_features(
    df: pd.DataFrame, trained_feature_names: list[str]
) -> list[str]:
    """
    Attempt to map a list of trained feature names to column names in the provided DataFrame.
    Returns a list of matched column names (same length as trained_feature_names) if a full
    mapping can be found, otherwise returns an empty list.

    The function attempts an exact (case-insensitive) match first, then a fuzzy match where
    a trained feature substring exists in an existing column name, or vice-versa.
    """
    df_cols = [c for c in df.columns]
    lc_cols = {c.lower(): c for c in df_cols}
    matched = []
    for tf in trained_feature_names:
        tf_lc = tf.lower()
        if tf_lc in lc_cols:
            matched.append(lc_cols[tf_lc])
            continue
        found = None
        for col in df_cols:
            # check if substring matches either way
            if tf_lc in col.lower() or col.lower() in tf_lc:
                found = col
                break
        if found:
            matched.append(found)
        else:
            # can't find an exact or fuzzy match
            return []

    return matched
