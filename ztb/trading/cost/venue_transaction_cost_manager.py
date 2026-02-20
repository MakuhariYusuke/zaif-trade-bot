"""
取引コスト柔軟設定システム
各交易所に応じた取引コストを動的に設定可能にする
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypedDict

from ztb.io.common import PathLike
from ztb.io.json_io import read_json_object, write_json

logger = logging.getLogger(__name__)


class TransactionCostConfigRecord(TypedDict):
    """Serializable transaction cost config."""

    venue_name: str
    maker_fee: float
    taker_fee: float
    min_fee: float
    max_fee: float
    currency: str
    last_updated: str


class TransactionCostConfigFile(TypedDict):
    """Top-level config payload."""

    venues: list[TransactionCostConfigRecord]


class VenueRecommendation(TypedDict):
    """Venue recommendation payload."""

    venue_name: str
    taker_fee_percent: float
    daily_cost_estimate: float
    score: float


class PortfolioImpact(TypedDict):
    """Portfolio cost impact payload."""

    venue_name: str
    avg_cost_per_trade_percent: float
    daily_cost_jpy: float
    annual_cost_jpy: float
    daily_cost_ratio_percent: float
    annual_cost_ratio_percent: float
    break_even_trades: int


class ErrorPayload(TypedDict):
    """Error payload."""

    error: str


@dataclass
class TransactionCostConfig:
    """
    取引コスト設定データクラス
    """

    venue_name: str
    maker_fee: float  # メイカー手数料（%）
    taker_fee: float  # テイカー手数料（%）
    min_fee: float  # 最低手数料
    max_fee: float  # 最高手数料
    currency: str  # 通貨
    last_updated: str  # 最終更新日

    def get_effective_cost(self, is_maker: bool = False) -> float:
        """
        実効取引コストを取得

        Args:
            is_maker: メイカー注文かどうか

        Returns:
            実効コスト（小数）
        """
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return min(max(fee_rate, self.min_fee), self.max_fee)


class VenueTransactionCostManager:
    """
    取引所取引コスト管理クラス
    """

    def __init__(
        self, config_file: PathLike = "config/venue_transaction_costs.json"
    ) -> None:
        """
        初期化

        Args:
            config_file: 設定ファイルパス
        """
        self.config_file = Path(config_file)
        self.venue_configs: dict[str, TransactionCostConfig] = {}
        self.load_configs()

    @staticmethod
    def _normalize_venue_name(venue_name: str) -> str:
        """Normalize venue key for consistent lookup."""
        return venue_name.strip().lower()

    @classmethod
    def _parse_config_record(cls, record: object) -> Optional[TransactionCostConfig]:
        """Parse and validate one config record."""
        if not isinstance(record, dict):
            return None

        venue_name_raw = record.get("venue_name")
        currency_raw = record.get("currency", "")
        last_updated_raw = record.get("last_updated", "")

        if not isinstance(venue_name_raw, str):
            return None
        venue_name = cls._normalize_venue_name(venue_name_raw)
        if not venue_name:
            return None

        if not isinstance(currency_raw, str) or not isinstance(last_updated_raw, str):
            return None

        try:
            return TransactionCostConfig(
                venue_name=venue_name,
                maker_fee=float(record.get("maker_fee", 0.0)),
                taker_fee=float(record.get("taker_fee", 0.0)),
                min_fee=float(record.get("min_fee", 0.0)),
                max_fee=float(record.get("max_fee", 0.0)),
                currency=currency_raw,
                last_updated=last_updated_raw,
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_record(config: TransactionCostConfig) -> TransactionCostConfigRecord:
        """Serialize in-memory config into JSON record."""
        return {
            "venue_name": config.venue_name,
            "maker_fee": config.maker_fee,
            "taker_fee": config.taker_fee,
            "min_fee": config.min_fee,
            "max_fee": config.max_fee,
            "currency": config.currency,
            "last_updated": config.last_updated,
        }

    def load_configs(self) -> None:
        """
        設定ファイルを読み込み
        """
        try:
            if not self.config_file.exists():
                self.create_default_configs()
                return

            data = read_json_object(self.config_file)
            self.venue_configs.clear()

            venues_payload = data.get("venues", [])
            if not isinstance(venues_payload, list):
                raise TypeError("'venues' must be a list")

            for venue_data in venues_payload:
                config = self._parse_config_record(venue_data)
                if config is None:
                    logger.warning(f"無効な取引コスト設定をスキップ: {venue_data}")
                    continue
                self.venue_configs[config.venue_name] = config

            if not self.venue_configs:
                logger.warning(
                    "取引コスト設定ファイルに有効な会場がないため、デフォルト設定を作成します"
                )
                self.create_default_configs()
                return

            logger.info(
                f"取引コスト設定を読み込みました: {len(self.venue_configs)}会場"
            )

        except Exception as e:
            logger.error(f"設定ファイル読み込みに失敗: {e}")
            self.create_default_configs()

    def create_default_configs(self) -> None:
        """
        デフォルト設定を作成
        """
        default_configs: list[TransactionCostConfigRecord] = [
            {
                "venue_name": "zaif",
                "maker_fee": 0.001,  # 0.1%
                "taker_fee": 0.001,  # 0.1%
                "min_fee": 0.0001,  # 0.01%
                "max_fee": 0.01,  # 1%
                "currency": "JPY",
                "last_updated": "2024-01-01",
            },
            {
                "venue_name": "bitflyer",
                "maker_fee": -0.0002,  # -0.02% (メイカー手数料負)
                "taker_fee": 0.0012,  # 0.12%
                "min_fee": 0.0001,
                "max_fee": 0.01,
                "currency": "JPY",
                "last_updated": "2024-01-01",
            },
            {
                "venue_name": "coincheck",
                "maker_fee": 0.0000,  # 無料（当座の間）
                "taker_fee": 0.0000,  # 無料（当座の間）
                "min_fee": 0.0000,
                "max_fee": 0.01,
                "currency": "JPY",
                "last_updated": "2024-01-01",
            },
            {
                "venue_name": "binance",
                "maker_fee": 0.001,  # 0.1%
                "taker_fee": 0.001,  # 0.1%
                "min_fee": 0.0001,
                "max_fee": 0.01,
                "currency": "USDT",
                "last_updated": "2024-01-01",
            },
            {
                "venue_name": "bybit",
                "maker_fee": -0.00025,  # -0.025%
                "taker_fee": 0.00075,  # 0.075%
                "min_fee": 0.0001,
                "max_fee": 0.01,
                "currency": "USDT",
                "last_updated": "2024-01-01",
            },
            {
                "venue_name": "generic_high_cost",
                "maker_fee": 0.002,  # 0.2%
                "taker_fee": 0.002,  # 0.2%
                "min_fee": 0.0005,
                "max_fee": 0.02,
                "currency": "USD",
                "last_updated": "2024-01-01",
            },
        ]

        self.venue_configs.clear()
        for config_data in default_configs:
            config = self._parse_config_record(config_data)
            if config is None:
                continue
            self.venue_configs[config.venue_name] = config

        self.save_configs()
        logger.info("デフォルト取引コスト設定を作成しました")

    def save_configs(self) -> None:
        """
        設定をファイルに保存
        """
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            data: TransactionCostConfigFile = {
                "venues": [
                    self._to_record(config)
                    for config in sorted(
                        self.venue_configs.values(), key=lambda item: item.venue_name
                    )
                ]
            }
            write_json(self.config_file, data, indent=2, ensure_ascii=False)

            logger.info(f"設定を保存しました: {self.config_file}")

        except Exception as e:
            logger.error(f"設定保存に失敗: {e}")

    def get_cost_config(self, venue_name: str) -> Optional[TransactionCostConfig]:
        """
        指定取引所のコスト設定を取得

        Args:
            venue_name: 取引所名

        Returns:
            コスト設定（見つからない場合はNone）
        """
        return self.venue_configs.get(self._normalize_venue_name(venue_name))

    def add_or_update_venue(self, config: TransactionCostConfig) -> None:
        """
        取引所設定を追加または更新

        Args:
            config: コスト設定
        """
        from datetime import datetime

        config.last_updated = datetime.now().strftime("%Y-%m-%d")
        config.venue_name = self._normalize_venue_name(config.venue_name)
        self.venue_configs[config.venue_name] = config
        self.save_configs()
        logger.info(f"取引所設定を更新しました: {config.venue_name}")

    def get_all_venues(self) -> list[str]:
        """
        設定済み取引所一覧を取得

        Returns:
            取引所名リスト
        """
        return list(self.venue_configs.keys())

    def calculate_portfolio_impact(
        self,
        venue_name: str,
        portfolio_value: float,
        daily_trades: int,
        is_maker_ratio: float = 0.5,
    ) -> PortfolioImpact | ErrorPayload:
        """
        ポートフォリオへのコスト影響を計算

        Args:
            venue_name: 取引所名
            portfolio_value: ポートフォリオ価値
            daily_trades: 1日あたりの取引回数
            is_maker_ratio: メイカー注文の割合

        Returns:
            コスト影響分析結果
        """
        config = self.get_cost_config(venue_name)
        if not config:
            return {"error": f"取引所設定が見つかりません: {venue_name}"}

        # 1取引あたりの平均コスト
        maker_cost = config.get_effective_cost(is_maker=True)
        taker_cost = config.get_effective_cost(is_maker=False)
        avg_cost_per_trade = maker_cost * is_maker_ratio + taker_cost * (
            1 - is_maker_ratio
        )

        # 1日あたりの総コスト
        daily_total_cost = avg_cost_per_trade * portfolio_value * daily_trades

        # 年間コスト（250営業日想定）
        annual_cost = daily_total_cost * 250

        # コスト比率
        daily_cost_ratio = daily_total_cost / portfolio_value
        annual_cost_ratio = annual_cost / portfolio_value

        return {
            "venue_name": venue_name,
            "avg_cost_per_trade_percent": avg_cost_per_trade * 100,
            "daily_cost_jpy": daily_total_cost,
            "annual_cost_jpy": annual_cost,
            "daily_cost_ratio_percent": daily_cost_ratio * 100,
            "annual_cost_ratio_percent": annual_cost_ratio * 100,
            "break_even_trades": int(1 / avg_cost_per_trade)
            if avg_cost_per_trade > 0
            else 0,
        }


class AdaptiveCostEnvironment:
    """
    適応型コスト環境クラス
    取引戦略に応じてコストを動的に調整
    """

    def __init__(self, base_venue: str = "zaif"):
        """
        初期化

        Args:
            base_venue: 基準取引所
        """
        self.cost_manager = VenueTransactionCostManager()
        self.base_venue = base_venue
        self.current_cost_config = self.cost_manager.get_cost_config(base_venue)

    def switch_venue(self, venue_name: str) -> bool:
        """
        取引所を切り替え

        Args:
            venue_name: 新しい取引所名

        Returns:
            切り替え成功フラグ
        """
        new_config = self.cost_manager.get_cost_config(venue_name)
        if new_config:
            self.current_cost_config = new_config
            self.base_venue = venue_name
            logger.info(f"取引所を切り替えました: {venue_name}")
            return True
        else:
            logger.error(f"取引所設定が見つかりません: {venue_name}")
            return False

    def get_current_cost(self, is_maker: bool = False) -> float:
        """
        現在の取引コストを取得

        Args:
            is_maker: メイカー注文かどうか

        Returns:
            現在のコスト
        """
        if self.current_cost_config:
            return self.current_cost_config.get_effective_cost(is_maker)
        return 0.001  # デフォルト0.1%

    def get_cost_presets(self) -> dict[str, dict[str, float]]:
        """
        コストプリセットを取得

        Returns:
            プリセット辞書
        """
        presets = {}

        for venue_name, config in self.cost_manager.venue_configs.items():
            presets[venue_name] = {
                "maker_fee": config.maker_fee,
                "taker_fee": config.taker_fee,
                "effective_taker": config.get_effective_cost(False),
            }

        return presets

    def recommend_venue_for_strategy(
        self, strategy_type: str, expected_daily_trades: int
    ) -> list[VenueRecommendation]:
        """
        戦略タイプに応じた取引所を推奨

        Args:
            strategy_type: 戦略タイプ ("high_frequency", "swing", "scalping")
            expected_daily_trades: 予想1日取引回数

        Returns:
            推奨取引所リスト（コスト順）
        """
        recommendations: list[VenueRecommendation] = []

        for venue_name, config in self.cost_manager.venue_configs.items():
            taker_cost = config.get_effective_cost(False)

            # 戦略タイプによる重み付け
            if strategy_type == "high_frequency":
                # 高頻度取引ではコストが重要
                score = 1 / (taker_cost + 0.0001)  # コストが低いほど高スコア
            elif strategy_type == "swing":
                # スイング取引ではコストより安定性
                score = 1 / (taker_cost + 0.0001) * 0.7
            else:  # scalping
                score = 1 / (taker_cost + 0.0001) * 0.8

            recommendations.append(
                {
                    "venue_name": venue_name,
                    "taker_fee_percent": taker_cost * 100,
                    "daily_cost_estimate": taker_cost
                    * 100000
                    * expected_daily_trades,  # 10万円ポートフォリオ想定
                    "score": score,
                }
            )

        # スコア順にソート
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]  # 上位5件


def create_cost_analysis_report():
    """
    取引コスト分析レポート作成
    """
    manager = VenueTransactionCostManager()

    print("=== 取引コスト分析レポート ===\n")

    # 全取引所の設定表示
    print("設定済み取引所:")
    for venue_name, config in manager.venue_configs.items():
        print(f"  {venue_name}:")
        print(f"    メイカー手数料: {config.maker_fee:.4f}")
        print(f"    テイカー手数料: {config.taker_fee:.4f}")
        print(f"    通貨: {config.currency}")
        print()

    # コスト影響分析（サンプル）
    print("コスト影響分析（10万円ポートフォリオ、10回/日取引想定）:")
    for venue_name in ["zaif", "bitflyer", "coincheck"]:
        impact = manager.calculate_portfolio_impact(venue_name, 100000, 10)
        if "error" not in impact:
            print(f"  {venue_name}:")
            print(f"    1日コスト: {impact['daily_cost_jpy']:.1f}円")
            print(f"    年間コスト: {impact['annual_cost_jpy']:.1f}円")
            print(f"    コスト比率: {impact['daily_cost_ratio_percent']:.3f}%")
            print()

    # 適応型コスト環境のデモ
    env = AdaptiveCostEnvironment()
    presets = env.get_cost_presets()

    print("コストプリセット:")
    for venue, costs in presets.items():
        print(
            f"  {venue}: メイカー {costs['maker_fee']:.4f}, テイカー {costs['taker_fee']:.4f}"
        )
    print("\n戦略別取引所推奨（高頻度取引、50回/日想定）:")
    recommendations = env.recommend_venue_for_strategy("high_frequency", 50)
    for rec in recommendations:
        print(
            f"  {rec['venue_name']}: スコア {rec['score']:.3f}, "
            f"推定日次コスト: {rec['daily_cost_estimate']:.0f}円"
        )


if __name__ == "__main__":
    create_cost_analysis_report()
