"""
マルチモーダル学習 - 合成データ生成器
データ不足時の代替手段としてリアルな合成データを生成
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MarketStateGenerator:
    """
    市場状態に基づくリアルなデータ生成器
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 市場状態の定義とパラメータ
        self.market_states = {
            'bull_trend': {
                'trend_strength': 0.8,
                'volatility': 0.15,
                'sentiment_bias': 0.6,
                'economic_momentum': 0.7,
                'duration_days': (30, 90)
            },
            'bear_trend': {
                'trend_strength': -0.8,
                'volatility': 0.25,
                'sentiment_bias': -0.6,
                'economic_momentum': -0.5,
                'duration_days': (20, 60)
            },
            'sideways': {
                'trend_strength': 0.0,
                'volatility': 0.08,
                'sentiment_bias': 0.0,
                'economic_momentum': 0.1,
                'duration_days': (45, 120)
            },
            'high_volatility': {
                'trend_strength': 0.0,
                'volatility': 0.35,
                'sentiment_bias': -0.3,
                'economic_momentum': -0.2,
                'duration_days': (10, 30)
            },
            'recovery': {
                'trend_strength': 0.4,
                'volatility': 0.12,
                'sentiment_bias': 0.3,
                'economic_momentum': 0.4,
                'duration_days': (25, 75)
            }
        }

        # 季節性パターン
        self.seasonal_patterns = {
            'q1': {'sentiment': -0.1, 'volatility': 1.2},  # 1月-3月
            'q2': {'sentiment': 0.2, 'volatility': 0.9},   # 4月-6月
            'q3': {'sentiment': 0.1, 'volatility': 1.1},   # 7月-9月
            'q4': {'sentiment': -0.2, 'volatility': 1.0}   # 10月-12月
        }

    def get_seasonal_multiplier(self, date: datetime) -> Dict[str, float]:
        """季節性に基づく乗数を計算"""
        month = date.month
        if month <= 3:
            return self.seasonal_patterns['q1']
        elif month <= 6:
            return self.seasonal_patterns['q2']
        elif month <= 9:
            return self.seasonal_patterns['q3']
        else:
            return self.seasonal_patterns['q4']

class SyntheticNewsGenerator:
    """
    合成ニュースデータ生成器
    市場状態に基づいてリアルなニュース記事を生成
    """

    def __init__(self, market_generator: MarketStateGenerator):
        self.market_gen = market_generator

        # ニューステンプレート
        self.templates = {
            'bull_trend': [
                "市場が{strength}上昇を続け、投資家心理が{condition}向上",
                "主要企業の決算が予想を上回り、株価指数が{hours}時間で{percent}%上昇",
                "経済指標の改善により{sector}セクターが牽引する強気相場",
                "機関投資家の買い意欲が高まり、市場全体が{condition}回復基調",
                "雇用統計の改善が確認され、FRBの政策期待が{condition}変化"
            ],
            'bear_trend': [
                "市場が{strength}下落を続け、投資家心理が{condition}悪化",
                "主要企業の決算が失望を誘い、株価指数が{hours}時間で{percent}%下落",
                "経済指標の悪化により{sector}セクターが売られる弱気相場",
                "機関投資家の売りが続き、市場全体が{condition}下落基調",
                "雇用統計の悪化が確認され、FRBの政策期待が{condition}変化"
            ],
            'sideways': [
                "市場が{condition}横ばいで推移、投資家は{condition}様子見姿勢",
                "経済指標が{condition}安定し、方向感に欠ける相場展開",
                "主要企業の決算が{condition}予想通りで市場に大きなインパクトなし",
                "ボラティリティの低下により短期トレーダーが{condition}減少",
                "地政学的リスクが{condition}後退し市場が落ち着きを取り戻す"
            ],
            'high_volatility': [
                "市場が{strength}変動を続け、投資家心理が{condition}不安定",
                "地政学的リスクにより株価指数が{hours}時間で{percent}%急変動",
                "経済指標の変動が大きく、市場参加者が{condition}混乱",
                "オプション市場のボラティリティが{hours}時間で{percent}%上昇",
                "通貨市場の変動が拡大し、リスク回避の動きが{condition}強まる"
            ],
            'recovery': [
                "市場が{condition}回復基調を見せ、投資家心理が{condition}改善",
                "経済指標の改善兆候が確認され、株価が{percent}%上昇",
                "主要企業の見通し改善により{sector}セクターが{condition}復調",
                "FRBの政策期待が{condition}変化し市場が{condition}落ち着き",
                "雇用市場の改善が確認され、消費関連株が{condition}買われる"
            ]
        }

        # テンプレート変数の選択肢
        self.template_vars = {
            'strength': ['大幅に', '着実に', '徐々に', '急速に'],
            'condition': ['大きく', '着実に', '徐々に', '顕著に', 'やや'],
            'hours': ['1', '2', '3', '6', '12', '24'],
            'percent': ['1-2', '2-3', '3-5', '5-8', '8-10'],
            'sector': ['テクノロジー', '金融', 'エネルギー', 'ヘルスケア', '消費財']
        }

    def generate_news_batch(self,
                           market_state: str,
                           num_articles: int,
                           date: datetime) -> List[Dict[str, Any]]:
        """
        指定された市場状態に基づいてニュース記事を生成

        Args:
            market_state: 市場状態
            num_articles: 生成する記事数
            date: ニュースの日付

        Returns:
            ニュース記事のリスト
        """

        if market_state not in self.templates:
            raise ValueError(f"Unknown market state: {market_state}")

        state_params = self.market_gen.market_states[market_state]
        seasonal_mult = self.market_gen.get_seasonal_multiplier(date)

        articles = []
        templates = self.templates[market_state]

        for i in range(num_articles):
            # テンプレート選択
            template = np.random.choice(templates)

            # 変数置換
            news_text = template
            for var, choices in self.template_vars.items():
                if '{' + var + '}' in news_text:
                    choice = np.random.choice(choices)
                    news_text = news_text.replace('{' + var + '}', choice)

            # 感情スコア計算
            base_sentiment = state_params['sentiment_bias']
            seasonal_sentiment = seasonal_mult['sentiment']
            noise = np.random.normal(0, 0.1)

            sentiment_score = base_sentiment + seasonal_sentiment + noise
            sentiment_score = np.clip(sentiment_score, -1, 1)

            # 信頼度
            confidence = np.random.uniform(0.7, 0.95)

            # 公開時刻のバリエーション
            time_offset = np.random.randint(0, 86400)  # 24時間以内
            publish_time = date + timedelta(seconds=time_offset)

            article = {
                'title': news_text[:50] + '...' if len(news_text) > 50 else news_text,
                'content': news_text,
                'summary': news_text[:100] + '...' if len(news_text) > 100 else news_text,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'market_state': market_state,
                'published_at': publish_time.isoformat(),
                'source': np.random.choice(['Reuters', 'Bloomberg', 'WSJ', 'FT', 'CNBC']),
                'category': np.random.choice(['economy', 'markets', 'companies', 'policy'])
            }

            articles.append(article)

        return articles

class SyntheticEconomicDataGenerator:
    """
    合成経済指標データ生成器
    """

    def __init__(self, market_generator: MarketStateGenerator):
        self.market_gen = market_generator

        # 経済指標のベース値と変動特性
        self.indicators_config = {
            'gdp_growth': {
                'base': 2.1, 'volatility': 0.8, 'trend': 0.02,
                'seasonal': {'q1': -0.3, 'q2': 0.2, 'q3': 0.1, 'q4': 0.0}
            },
            'unemployment_rate': {
                'base': 4.2, 'volatility': 0.3, 'trend': -0.01,
                'seasonal': {'q1': 0.2, 'q2': -0.1, 'q3': -0.1, 'q4': 0.0}
            },
            'inflation_rate': {
                'base': 2.3, 'volatility': 0.4, 'trend': 0.01,
                'seasonal': {'q1': -0.2, 'q2': 0.1, 'q3': 0.3, 'q4': -0.2}
            },
            'fed_funds_rate': {
                'base': 2.5, 'volatility': 0.2, 'trend': 0.005,
                'seasonal': {'q1': 0.0, 'q2': 0.0, 'q3': 0.0, 'q4': 0.0}
            },
            'sp500_index': {
                'base': 4200, 'volatility': 150, 'trend': 5,
                'seasonal': {'q1': -50, 'q2': 100, 'q3': 50, 'q4': -100}
            }
        }

    def generate_economic_data(self,
                              start_date: datetime,
                              end_date: datetime,
                              frequency: str = 'M') -> pd.DataFrame:
        """
        指定期間の経済指標データを生成

        Args:
            start_date: 開始日
            end_date: 終了日
            frequency: データ頻度 ('D'日次, 'W'週次, 'M'月次)

        Returns:
            経済指標データフレーム
        """

        # 日付範囲生成
        date_range = pd.date_range(start_date, end_date, freq=frequency)

        economic_data = {}

        for indicator, config in self.indicators_config.items():
            values = []

            for date in date_range:
                # ベース値
                base_value = config['base']

                # トレンド成分
                days_elapsed = (date - start_date).days
                trend_component = config['trend'] * (days_elapsed / 365.25)

                # 季節性成分
                seasonal_mult = self.market_gen.get_seasonal_multiplier(date)
                seasonal_component = config['seasonal'].get(
                    f"q{(date.month - 1) // 3 + 1}", 0
                )

                # 市場状態の影響（簡易版）
                market_influence = np.random.normal(0, 0.1)

                # ノイズ
                noise = np.random.normal(0, config['volatility'])

                # 最終値計算
                value = (base_value +
                        trend_component +
                        seasonal_component +
                        market_influence +
                        noise)

                # インデックスは対数正規分布に従うように調整
                if 'index' in indicator:
                    value = max(value, 100)  # 最低値を設定

                values.append(value)

            economic_data[indicator] = values

        # データフレーム作成
        df = pd.DataFrame(economic_data, index=date_range)

        # スムージング適用（市場の現実性を高める）
        for col in df.columns:
            if 'index' in col:
                # 株価指数は滑らかにする
                df[col] = savgol_filter(df[col], window_length=min(31, len(df)), polyorder=2)

        return df

class GANBasedDataGenerator(nn.Module):
    """
    GANベースの高度な合成データ生成器
    （オプション：よりリアルなデータ生成）
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 20, hidden_dim: int = 128):
        super().__init__()

        # Generator: ノイズからデータを生成
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        # Discriminator: 本物か偽物かを判定
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def generate_synthetic_batch(self, batch_size: int) -> torch.Tensor:
        """
        合成データをバッチ生成

        Args:
            batch_size: 生成するサンプル数

        Returns:
            合成データテンソル
        """
        noise = torch.randn(batch_size, self.input_dim)
        synthetic_data = self.generator(noise)
        return synthetic_data

class MultiModalSyntheticDataset:
    """
    マルチモーダル合成データセット統合クラス
    """

    def __init__(self, seed: int = 42):
        self.market_gen = MarketStateGenerator(seed)
        self.news_gen = SyntheticNewsGenerator(self.market_gen)
        self.economic_gen = SyntheticEconomicDataGenerator(self.market_gen)

    def generate_comprehensive_dataset(self,
                                     start_date: datetime,
                                     end_date: datetime,
                                     news_per_day: int = 5) -> Dict[str, Any]:
        """
        包括的なマルチモーダルデータセットを生成

        Args:
            start_date: 開始日
            end_date: 終了日
            news_per_day: 1日あたりのニュース記事数

        Returns:
            統合データセット辞書
        """

        logger.info(f"Generating synthetic dataset from {start_date} to {end_date}")

        # 日付範囲
        date_range = pd.date_range(start_date, end_date, freq='D')

        # 経済指標データ生成
        economic_data = self.economic_gen.generate_economic_data(start_date, end_date, 'D')

        # ニュースデータ生成
        all_news = []
        market_states_over_time = []

        for date in date_range:
            # 日付に応じた市場状態を決定（簡易版）
            day_of_year = date.timetuple().tm_yday
            market_state = self._get_market_state_for_date(date)

            # ニュース生成
            daily_news = self.news_gen.generate_news_batch(
                market_state, news_per_day, date
            )
            all_news.extend(daily_news)
            market_states_over_time.extend([market_state] * news_per_day)

        # ニュースデータをDataFrameに変換
        news_df = pd.DataFrame(all_news)

        # 日次集計（ニュースの感情スコアを日次平均に）
        daily_sentiment = news_df.groupby(
            news_df['published_at'].str[:10]
        )['sentiment_score'].agg(['mean', 'std', 'count']).reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_volatility', 'news_count']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        # データ統合
        integrated_data = {
            'economic_indicators': economic_data,
            'news_data': news_df,
            'daily_sentiment': daily_sentiment.set_index('date'),
            'market_states': market_states_over_time,
            'metadata': {
                'generation_date': datetime.now(),
                'date_range': (start_date, end_date),
                'total_news_articles': len(all_news),
                'avg_news_per_day': len(all_news) / len(date_range)
            }
        }

        logger.info(f"Generated dataset with {len(all_news)} news articles and {len(economic_data)} economic data points")

        return integrated_data

    def _get_market_state_for_date(self, date: datetime) -> str:
        """日付に応じた市場状態を決定"""
        # 簡易的なロジック（実際にはより複雑なルールベースを使用）
        month = date.month

        # 四半期ごとの傾向
        if month in [1, 2]:  # 1-2月: 不確実性が高い
            return np.random.choice(['high_volatility', 'bear_trend'], p=[0.6, 0.4])
        elif month in [3, 4]:  # 3-4月: 回復期
            return np.random.choice(['recovery', 'sideways'], p=[0.7, 0.3])
        elif month in [5, 6]:  # 5-6月: 安定期
            return np.random.choice(['bull_trend', 'sideways'], p=[0.5, 0.5])
        elif month in [7, 8]:  # 7-8月: 夏枯れ
            return np.random.choice(['sideways', 'high_volatility'], p=[0.6, 0.4])
        elif month in [9, 10]:  # 9-10月: 不確実性再燃
            return np.random.choice(['high_volatility', 'bear_trend'], p=[0.5, 0.5])
        else:  # 11-12月: 年末効果
            return np.random.choice(['bull_trend', 'recovery'], p=[0.6, 0.4])

    def save_dataset(self, dataset: Dict[str, Any], base_path: str):
        """データセットを保存"""
        import os

        os.makedirs(base_path, exist_ok=True)

        # 経済指標保存
        dataset['economic_indicators'].to_csv(f"{base_path}/economic_indicators.csv")

        # ニュースデータ保存
        dataset['news_data'].to_csv(f"{base_path}/news_data.csv", index=False)

        # 日次感情データ保存
        dataset['daily_sentiment'].to_csv(f"{base_path}/daily_sentiment.csv")

        # メタデータ保存
        with open(f"{base_path}/metadata.json", 'w') as f:
            # datetimeを文字列に変換
            metadata = dataset['metadata'].copy()
            metadata['generation_date'] = metadata['generation_date'].isoformat()
            metadata['date_range'] = [d.isoformat() for d in metadata['date_range']]
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved to {base_path}")

# 使用例
if __name__ == "__main__":
    # 合成データセット生成器の初期化
    generator = MultiModalSyntheticDataset(seed=42)

    # 2024年のデータ生成
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    # データセット生成
    dataset = generator.generate_comprehensive_dataset(
        start_date=start_date,
        end_date=end_date,
        news_per_day=3
    )

    # データ保存
    generator.save_dataset(dataset, "data/synthetic_multimodal")

    # 生成データの概要表示
    print("=== 合成データセット概要 ===")
    print(f"期間: {start_date.date()} - {end_date.date()}")
    print(f"ニュース記事数: {len(dataset['news_data'])}")
    print(f"経済指標データポイント数: {len(dataset['economic_indicators'])}")
    print(f"日次感情データポイント数: {len(dataset['daily_sentiment'])}")

    print("\n=== サンプルデータ ===")
    print("\n経済指標（先頭5行）:")
    print(dataset['economic_indicators'].head())

    print("\nニュースデータ（先頭3件）:")
    print(dataset['news_data'][['title', 'sentiment_score', 'market_state']].head(3))

    print("\n日次感情データ（先頭5行）:")
    print(dataset['daily_sentiment'].head())

    print("""
    🎯 合成データ生成の利点:

    1. データ不足を解消
       - リアルタイムデータがなくても学習可能
       - 特定の市場状態を増幅生成可能

    2. 制御されたデータ分布
       - 特定の感情バイアスや市場状態を意図的に生成
       - 学習のバランス調整が可能

    3. プライバシー保護
       - 実際のニュースデータを使わず合成データで学習
       - 機密情報の漏洩リスクなし

    4. 拡張性
       - 必要に応じてデータ量を無限に増やせる
       - 新しい市場状態やシナリオを追加可能

    ⚠️ 注意事項:
    - 合成データは現実の複雑さを完全に再現できない
    - 可能であれば実データと組み合わせることを推奨
    - 生成パラメータの調整が必要
    """)