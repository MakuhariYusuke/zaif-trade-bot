"""
特徴量セットの統一管理システム
設定ファイルベースで特徴量を動的に管理し、簡単に増減できるようにする
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ztb.utils.config_loader import ConfigLoader
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureSetType(Enum):
    """特徴量セットの種類"""

    CURATED = "curated"
    FULL = "full"
    MINIMAL = "minimal"
    CUSTOM = "custom"


@dataclass
class FeatureSetConfig:
    """特徴量セットの設定"""

    name: str
    description: str
    features: List[str]
    enabled: bool = True
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None


class FeatureSetManager:
    """
    特徴量セットの統一管理マネージャー

    特徴量セットをYAML設定ファイルで管理し、
    動的な追加・削除・有効化・無効化をサポート
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs/features/feature_sets.yaml")
        self.feature_sets: Dict[str, FeatureSetConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """設定ファイルを読み込む"""
        if not self.config_path.exists():
            logger.warning(
                f"Feature sets config not found: {self.config_path}, creating default"
            )
            self._create_default_config()
            return

        try:
            config_data = ConfigLoader.load(self.config_path)
            self._parse_config(config_data)
            logger.info(
                f"Loaded {len(self.feature_sets)} feature sets from {self.config_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load feature sets config: {e}")
            self._create_default_config()

    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """設定データをパース"""
        for set_name, set_config in config_data.get("feature_sets", {}).items():
            if isinstance(set_config, dict):
                self.feature_sets[set_name] = FeatureSetConfig(
                    name=set_name,
                    description=set_config.get("description", ""),
                    features=set_config.get("features", []),
                    enabled=set_config.get("enabled", True),
                    version=set_config.get("version", "1.0"),
                    metadata=set_config.get("metadata", {}),
                )

    def _create_default_config(self) -> None:
        """デフォルト設定ファイルを作成"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # デフォルトの特徴量セット
        default_features = self._get_default_features()

        default_config = {
            "version": "1.0",
            "description": "特徴量セットの統一管理設定",
            "feature_sets": {
                "curated": {
                    "description": "質的に改善された特徴量セット（78個）",
                    "features": default_features,
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "production", "recommended": True},
                },
                "minimal": {
                    "description": "最小限の特徴量セット（20個）",
                    "features": default_features[:20],
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "testing", "recommended": False},
                },
                "full": {
                    "description": "全特徴量セット（curatedと同じ）",
                    "features": default_features,
                    "enabled": True,
                    "version": "1.0",
                    "metadata": {"category": "experimental", "recommended": False},
                },
            },
        }

        # YAMLファイルに保存
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Created default feature sets config: {self.config_path}")

        # 設定を再読み込み
        self._load_config()

    def _get_default_features(self) -> List[str]:
        """デフォルトの特徴量リストを取得（curated_features.pyから統合）"""
        # curated_features.py から統合した CURATED_FEATURES
        return [
            # 【価格基本】 (5個)
            "close",
            "open",
            "high",
            "low",
            "volume",
            # 【トレンド指標】 (47個) - 6つ追加
            "ADX_M1",  # 1分足ADX
            "ADX_M5",  # 5分足ADX
            "ADX_M15",  # 15分足ADX
            "ADX_H1",  # 1時間足ADX
            "ADX_H4",  # 4時間足ADX
            "ADX_D1",  # 日足ADX
            "PlusDI",  # +DI
            "PlusDI_M1",  # 1分足+DI
            "PlusDI_M5",  # 5分足+DI
            "PlusDI_M15",  # 15分足+DI
            "PlusDI_H1",  # 1時間足+DI
            "PlusDI_H4",  # 4時間足+DI
            "PlusDI_D1",  # 日足+DI
            "MinusDI",  # -DI
            "MinusDI_M1",  # 1分足-DI
            "MinusDI_M5",  # 5分足-DI
            "MinusDI_M15",  # 15分足-DI
            "MinusDI_H1",  # 1時間足-DI
            "MinusDI_H4",  # 4時間足-DI
            "MinusDI_D1",  # 日足-DI
            "MACD",  # トレンド方向
            "PSAR",  # パラボリックSAR
            "PSAR_Trend",  # PSARトレンド方向
            "EMACross_Diff",  # EMAクロス差分
            "EMACross_Diff_M1",  # 1分足EMAクロス差分
            "EMACross_Diff_M5",  # 5分足EMAクロス差分
            "EMACross_Diff_M15",  # 15分足EMAクロス差分
            "EMACross_Diff_H1",  # 1時間足EMAクロス差分
            "EMACross_Diff_H4",  # 4時間足EMAクロス差分
            "EMACross_Diff_D1",  # 日足EMAクロス差分
            "EMACross_Signal",  # EMAクロスシグナル
            "EMACross_Signal_M1",  # 1分足EMAクロスシグナル
            "EMACross_Signal_M5",  # 5分足EMAクロスシグナル
            "EMACross_Signal_M15",  # 15分足EMAクロスシグナル
            "EMACross_Signal_H1",  # 1時間足EMAクロスシグナル
            "EMACross_Signal_H4",  # 4時間足EMAクロスシグナル
            "EMACross_Signal_H1",  # 1時間足EMAクロスシグナル
            "EMACross_Signal_H4",  # 4時間足EMAクロスシグナル
            "EMACross_Signal_D1",  # 日足EMAクロスシグナル
            "TEMA",  # トリプルEMA
            "ema_5",  # 短期EMA
            "VWAP",  # 出来高加重平均価格
            "rolling_mean_20",  # 20期間移動平均
            "HeikinAshi_Color_M1",  # 1分足平均足色
            "HeikinAshi_Color_M5",  # 5分足平均足色
            "HeikinAshi_Color_M15",  # 15分足平均足色
            "HeikinAshi_Color_H1",  # 1時間足平均足色
            "HeikinAshi_Color_H4",  # 4時間足平均足色
            "HeikinAshi_Color_D1",  # 日足平均足色
            # 【オシレーター】 (13個) - 6つ追加
            "RSI",  # 相対力指数
            "RSI_M1",  # 1分足RSI
            "RSI_M5",  # 5分足RSI
            "RSI_M15",  # 15分足RSI
            "RSI_H1",  # 1時間足RSI
            "RSI_H4",  # 4時間足RSI
            "RSI_D1",  # 日足RSI
            "CCI",  # コモディティチャネル指数
            "Stochastic",  # ストキャスティクス
            "Stochastic_Trend_Alignment",  # ストキャトレンド整合
            "Stochastic_Signal_Strength",  # ストキャシグナル強度
            "Williams_R",  # ウィリアムズ%R
            "MFI",  # マネーフローインデックス
            # 【ボラティリティ】 (17個) - 12つ追加
            "ATR",  # 平均真の範囲
            "ATR_M1",  # 1分足ATR
            "ATR_M5",  # 5分足ATR
            "ATR_M15",  # 15分足ATR
            "ATR_H1",  # 1時間足ATR
            "ATR_H4",  # 4時間足ATR
            "ATR_D1",  # 日足ATR
            "ATR_simplified",  # 簡易ATR
            "ATR_simplified_M1",  # 1分足簡易ATR
            "ATR_simplified_M5",  # 5分足簡易ATR
            "ATR_simplified_M15",  # 15分足簡易ATR
            "ATR_simplified_H1",  # 1時間足簡易ATR
            "ATR_simplified_H4",  # 4時間足簡易ATR
            "ATR_simplified_D1",  # 日足簡易ATR
            "Normalized_ATR",  # 正規化ATR (パーセント表示)
            "atr_10",  # 10期間ATR
            "HV",  # 過去ボラティリティ
            "BB_Position",  # ボリンジャーバンド位置
            "Bollinger_Percent_B",  # ボリンジャー%B
            "BB_Width",  # ボリンジャー幅
            # 【ボリンジャーバンド】 (1個)
            "Bollinger_Band_Expansion",  # バンド拡大
            # 【ケルトナーチャネル】 (2個)
            "Keltner_Position",  # ケルトナー位置
            "Keltner_Width",  # ケルトナー幅
            # 【ドンチャンチャネル】 (3個)
            "Donchian_Pos_20",  # ドンチャン位置
            "Donchian_Squeeze_Ratio",  # スクイーズ比率
            "Donchian_Width_Rel_20",  # 相対幅
            # 【一目均衡表(組み合わせ+拡張+多時間軸)】 (34個) - 18つ追加
            # 基本ライン
            "Ichimoku_Tenkan",  # 転換線(Conversion Line)
            "Ichimoku_Kijun",  # 基準線(Base Line)
            "Ichimoku_Senkou_A",  # 先行スパンA(Leading Span A)
            "Ichimoku_Senkou_B",  # 先行スパンB(Leading Span B)
            "Ichimoku_Chikou",  # 遅行スパン(Lagging Span)
            # 基本分析
            "Ichimoku_Composite_Signal",  # 総合シグナル(複数線の組み合わせ)
            "Ichimoku_Price_Cloud_Distance",  # 価格と雲の距離
            "Ichimoku_Cloud_Thickness",  # 雲の厚み
            "Ichimoku_Trend",  # トレンド方向
            # 理論的拡張
            "Ichimoku_Time_Theory",  # 時間論: 転換線と基準線の時間的関係
            "Ichimoku_Wave_Theory",  # 波動論: 雲の波動的意味付け
            "Ichimoku_Value_Measurement",  # 値幅観測論: 価格変動の測定
            "Ichimoku_Momentum_Confirmation",  # 勢い確認: 遅行スパンのモメンタム的解釈
            # 高度な分析
            "Ichimoku_Cloud_Slope",  # 雲の傾き/角度
            "Ichimoku_Sanyaku_Kouten",  # 三役好転/逆転
            "Ichimoku_Cloud_Expansion",  # 雲の拡大/縮小
            # 多時間軸拡張 (各時間軸でComposite Signal, Trend, Cloud Thickness, Price-Cloud Distance)
            "Ichimoku_Composite_Signal_M1",  # 1分足総合シグナル
            "Ichimoku_Composite_Signal_M5",  # 5分足総合シグナル
            "Ichimoku_Composite_Signal_M15",  # 15分足総合シグナル
            "Ichimoku_Composite_Signal_H1",  # 1時間足総合シグナル
            "Ichimoku_Composite_Signal_H4",  # 4時間足総合シグナル
            "Ichimoku_Composite_Signal_D1",  # 日足総合シグナル
            "Ichimoku_Trend_M1",  # 1分足トレンド
            "Ichimoku_Trend_M5",  # 5分足トレンド
            "Ichimoku_Trend_M15",  # 15分足トレンド
            "Ichimoku_Trend_H1",  # 1時間足トレンド
            "Ichimoku_Trend_H4",  # 4時間足トレンド
            "Ichimoku_Trend_D1",  # 日足トレンド
            "Ichimoku_Cloud_Thickness_M1",  # 1分足雲の厚み
            "Ichimoku_Cloud_Thickness_M5",  # 5分足雲の厚み
            "Ichimoku_Cloud_Thickness_M15",  # 15分足雲の厚み
            "Ichimoku_Cloud_Thickness_H1",  # 1時間足雲の厚み
            "Ichimoku_Cloud_Thickness_H4",  # 4時間足雲の厚み
            "Ichimoku_Cloud_Thickness_D1",  # 日足雲の厚み
            "Ichimoku_Price_Cloud_Distance_M1",  # 1分足価格-雲距離
            "Ichimoku_Price_Cloud_Distance_M5",  # 5分足価格-雲距離
            "Ichimoku_Price_Cloud_Distance_M15",  # 15分足価格-雲距離
            "Ichimoku_Price_Cloud_Distance_H1",  # 1時間足価格-雲距離
            "Ichimoku_Price_Cloud_Distance_H4",  # 4時間足価格-雲距離
            "Ichimoku_Price_Cloud_Distance_D1",  # 日足価格-雲距離
            # 【スーパートレンド】 (4個) - Reversal Signal除外(離散値)
            "Supertrend",  # スーパートレンド値
            "Supertrend_Direction",  # 方向
            "Supertrend_Strength",  # 強度
            "Supertrend_Trend_Duration",  # トレンド継続期間
            # Note: Supertrend_Reversal_Signal, Supertrend_Volatility_Filter は除外
            # 【ボリューム分析】 (8個) - 2つ追加
            "OBV",  # オンバランスボリューム
            "CMF",  # チャイキンマネーフロー
            "Chaikin_AD",  # チャイキンA/Dライン
            "Chaikin_AD_Oscillator",  # チャイキンA/Dオシレーター
            "PriceVolumeCorr",  # 価格出来高相関
            "Volume_Profile_Distribution",  # 出来高プロファイル分布
            "Volume_Profile_Value_Area_High",  # 値域上限
            "liquidity_surge",  # 流動性急増
            # 【その他の重要指標】 (6個)
            "ROC",  # 変化率
            "ZScore",  # Zスコア
            "ReturnMA_Short",  # 短期リターン移動平均
            "ReturnStdDev",  # リターン標準偏差
            "Kalman_Residual_Norm",  # カルマン残差正規化
            # 【Ta-Lib活用拡張指標】 (3個追加)
            "Ultimate_Oscillator",  # アルティメットオシレーター(3期間モメンタム統合)
            "TSI",  # True Strength Index(真の強度指数)
            "KST",  # Know Sure Thing(ノウシュアシング)
            # 【マイクロ構造】 (2個) - 1つ削減
            "micro_volatility",  # マイクロボラティリティ
            "price_velocity",  # 価格速度
            "price_acceleration",  # 価格加速度
            # 【時間特徴】 (2個追加)
            "Time_Monthly_Cycle",  # 月次サイクル進行度
            "Time_Quarterly_Cycle",  # 四半期サイクル進行度
            # 【Optimizer特徴量】 (11個追加) - 学習プロセス中のoptimizer状態
            "optimizer_learning_rate",  # 学習率
            "optimizer_learning_rate_trend",  # 学習率トレンド
            "optimizer_gradient_norm_avg",  # 勾配ノルム平均
            "optimizer_gradient_norm_std",  # 勾配ノルム標準偏差
            "optimizer_step_size_avg",  # ステップサイズ平均
            "optimizer_momentum_avg",  # モメンタム平均
            "optimizer_training_progress",  # 学習進捗
            "optimizer_loss_trend",  # 損失トレンド
            "optimizer_update_frequency_avg",  # 更新頻度平均
            "optimizer_stability_score",  # 安定性スコア
            "optimizer_adaptive_lr_score",  # 適応学習率スコア
        ]

    def get_feature_set(self, name: str) -> List[str]:
        """指定された名前の特徴量セットを取得"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found, using 'curated'")
            name = "curated"

        feature_set = self.feature_sets[name]
        if not feature_set.enabled:
            logger.warning(f"Feature set '{name}' is disabled, using 'curated'")
            curated_set = self.feature_sets.get("curated")
            if curated_set and curated_set.enabled:
                return curated_set.features
            else:
                return self._get_default_features()

        logger.info(
            f"Using feature set '{name}' with {len(feature_set.features)} features"
        )
        return feature_set.features

    def add_feature_set(
        self,
        name: str,
        features: List[str],
        description: str = "",
        enabled: bool = True,
    ) -> bool:
        """新しい特徴量セットを追加"""
        if name in self.feature_sets:
            logger.warning(f"Feature set '{name}' already exists")
            return False

        self.feature_sets[name] = FeatureSetConfig(
            name=name,
            description=description,
            features=features,
            enabled=enabled,
            version="1.0",
        )

        self._save_config()
        logger.info(f"Added feature set '{name}' with {len(features)} features")
        return True

    def update_feature_set(
        self,
        name: str,
        features: Optional[List[str]] = None,
        description: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """特徴量セットを更新"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found")
            return False

        feature_set = self.feature_sets[name]

        if features is not None:
            feature_set.features = features
        if description is not None:
            feature_set.description = description
        if enabled is not None:
            feature_set.enabled = enabled

        self._save_config()
        logger.info(f"Updated feature set '{name}'")
        return True

    def remove_feature_set(self, name: str) -> bool:
        """特徴量セットを削除"""
        if name not in self.feature_sets:
            logger.warning(f"Feature set '{name}' not found")
            return False

        if name in ["curated", "minimal", "full"]:
            logger.warning(f"Cannot remove built-in feature set '{name}'")
            return False

        del self.feature_sets[name]
        self._save_config()
        logger.info(f"Removed feature set '{name}'")
        return True

    def add_features(self, set_name: str, features: List[str]) -> bool:
        """特徴量セットに特徴量を追加"""
        if set_name not in self.feature_sets:
            logger.warning(f"Feature set '{set_name}' not found")
            return False

        feature_set = self.feature_sets[set_name]
        # 重複を避ける
        existing_features = set(feature_set.features)
        new_features = [f for f in features if f not in existing_features]

        if not new_features:
            logger.info(f"All features already exist in set '{set_name}'")
            return True

        feature_set.features.extend(new_features)
        self._save_config()
        logger.info(f"Added {len(new_features)} features to set '{set_name}'")
        return True

    def remove_features(self, set_name: str, features: List[str]) -> bool:
        """特徴量セットから特徴量を削除"""
        if set_name not in self.feature_sets:
            logger.warning(f"Feature set '{set_name}' not found")
            return False

        feature_set = self.feature_sets[set_name]
        original_count = len(feature_set.features)

        # 指定された特徴量を削除
        features_to_remove = set(features)
        feature_set.features = [
            f for f in feature_set.features if f not in features_to_remove
        ]

        removed_count = original_count - len(feature_set.features)
        if removed_count == 0:
            logger.info(f"No features were removed from set '{set_name}'")
            return True

        self._save_config()
        logger.info(f"Removed {removed_count} features from set '{set_name}'")
        return True

    def list_feature_sets(self) -> Dict[str, Dict[str, Any]]:
        """利用可能な特徴量セットの一覧を取得"""
        return {
            name: {
                "description": fs.description,
                "feature_count": len(fs.features),
                "enabled": fs.enabled,
                "version": fs.version,
            }
            for name, fs in self.feature_sets.items()
        }

    def get_feature_count(self, name: str) -> int:
        """指定された特徴量セットの特徴量数を取得"""
        feature_set = self.feature_sets.get(name)
        return len(feature_set.features) if feature_set else 0

    def _save_config(self) -> None:
        """設定をファイルに保存"""
        config_data = {
            "version": "1.0",
            "description": "特徴量セットの統一管理設定",
            "feature_sets": {},
        }

        for name, feature_set in self.feature_sets.items():
            config_data["feature_sets"][name] = {
                "description": feature_set.description,
                "features": feature_set.features,
                "enabled": feature_set.enabled,
                "version": feature_set.version,
                "metadata": feature_set.metadata or {},
            }

        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)


# グローバルインスタンス
_feature_manager: Optional[FeatureSetManager] = None


def get_feature_manager() -> FeatureSetManager:
    """特徴量マネージャーのインスタンスを取得"""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureSetManager()
    return _feature_manager


def get_feature_set(name: str = "curated") -> List[str]:
    """特徴量セットを取得（後方互換性のための関数）"""
    manager = get_feature_manager()
    return manager.get_feature_set(name)


def get_features_to_remove(feature_set_name: str = "curated") -> List[str]:
    """削除すべき特徴量を取得（後方互換性のための関数）"""
    # この関数は現在は使用されないが、後方互換のために残す
    return []
