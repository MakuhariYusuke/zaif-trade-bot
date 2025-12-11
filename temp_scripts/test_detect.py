from unittest.mock import Mock

import pandas as pd

from ztb.trading.real_data_validator import CrossValidator

validator = CrossValidator(Mock(), None)
data_dict = {
    "source1": pd.DataFrame({"price": [100, 105, 110, 115, 120]}),
    "source2": pd.DataFrame({"price": [100, 105, 110, 115, 120]}),
    "source3": pd.DataFrame({"price": [100, 200, 110, 115, 120]}),
}
print(validator._detect_data_discrepancies(data_dict))
