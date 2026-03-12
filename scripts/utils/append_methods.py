file_path = r"c:\Users\Admin\dev\zaif-trade-bot\ztb\features\models\sac\sac_v427_feature_engineering.py"

new_methods = """
    def _generate_enhanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("DEBUG: _generate_enhanced_statistical_features called")
        features = pd.DataFrame(index=df.index)
        # Rolling Mean/Std
        for window in [14, 50]:
            features[f"RollingMean{window}"] = df["close"].rolling(window).mean()
            features[f"RollingStd{window}"] = df["close"].rolling(window).std()

        # ZScore
        features["ZScore"] = (df["close"] - features["RollingMean14"]) / features["RollingStd14"].replace(0, 1)

        # KalmanFilter (simplified)
        features["KalmanFilter"] = df["close"].ewm(span=10).mean() # Placeholder

        return features.fillna(0)

    def _generate_volume_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("DEBUG: _generate_volume_features_optimized called")
        features = pd.DataFrame(index=df.index)
        if "volume" in df.columns:
            # OBV
            features["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
            # MFI
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            money_flow = typical_price * df["volume"]

            # Vectorized MFI
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow.replace(0, 1)
            features["MFI"] = 100 - (100 / (1 + mfi_ratio))

            # VWAP
            features["VWAP"] = (df["volume"] * typical_price).cumsum() / df["volume"].cumsum().replace(0, 1)

        return features.fillna(0)

    def _generate_momentum_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("DEBUG: _generate_momentum_features_optimized called")
        features = pd.DataFrame(index=df.index)
        # Lags
        for lag in [1, 2, 3, 5]:
            features[f"Lags_close_{lag}"] = df["close"].shift(lag)
            # Also add generic "Lags" feature if expected
            if lag == 1:
                features["Lags"] = features[f"Lags_close_{lag}"]

        # Donchian
        features["Donchian_High"] = df["high"].rolling(20).max()
        features["Donchian_Low"] = df["low"].rolling(20).min()
        features["Donchian"] = (features["Donchian_High"] + features["Donchian_Low"]) / 2

        return features.fillna(0)
"""

with open(file_path, "a", encoding="utf-8") as f:
    f.write(new_methods)

print("Appended methods.")
