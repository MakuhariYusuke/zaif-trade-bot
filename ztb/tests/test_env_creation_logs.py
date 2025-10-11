"""環境作成時のログを確認するテストスクリプト"""

import logging
import pandas as pd
from ztb.trading.environment.schema_env_factory import create_env_from_model_path

# ログレベルをDEBUGに設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# データ読込
df = pd.read_csv('btc_jpy_real_dataset.csv')

# 環境作成（ログを表示）
print("\n=== Creating environment from v390 model ===\n")
env = create_env_from_model_path('models/ppo_profitable_v390_hybrid.zip', df)

# 環境属性確認
print(f"\n=== Environment attributes ===")
print(f"Random start: {getattr(env, 'random_start', 'ATTRIBUTE NOT FOUND')}")
print(f"Enable action masking: {getattr(env, 'enable_action_masking', 'ATTRIBUTE NOT FOUND')}")
print(f"Has action_validator: {hasattr(env, 'action_validator')}")

# HeavyTradingEnvの__init__シグネチャ確認
import inspect
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
sig = inspect.signature(HeavyTradingEnv.__init__)
print(f"\n=== HeavyTradingEnv.__init__ parameters ===")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    print(f"  {param_name}: {param.default if param.default != inspect.Parameter.empty else 'REQUIRED'}")
