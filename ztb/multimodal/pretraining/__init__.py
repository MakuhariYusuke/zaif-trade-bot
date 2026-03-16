"""
Self-supervised Pre-training Module for Financial Data
金融データ特化の自己教師あり事前学習モジュール

This module implements various self-supervised learning techniques
specifically adapted for financial time series data:
- Masked Price Modeling (MPM)
- Contrastive Learning for Time Series
- Anomaly Detection Pre-training

各手法の概要:
- MPM: 価格データをマスクし、予測するBERT-style学習
- Contrastive Learning: 類似/相違ペア生成による表現学習
- Anomaly Detection: 時系列異常検知のための事前学習
"""

from .anomaly_detection_pretraining import (
    AnomalyDetectionPretrainer,
    HybridAnomalyDetector,
    PredictionAnomalyDetector,
    ReconstructionAnomalyDetector,
)
from .contrastive_learning import (
    ContrastiveLearningModel,
    ContrastiveLearningTrainer,
    TimeSeriesAugmentation,
)
from .masked_price_modeling import MaskedPriceModel, MaskedPriceModelingTrainer
from .self_supervised_trainer import SelfSupervisedTrainer

__all__ = [
    # Masked Price Modeling
    "MaskedPriceModel",
    "MaskedPriceModelingTrainer",
    # Contrastive Learning
    "TimeSeriesAugmentation",
    "ContrastiveLearningModel",
    "ContrastiveLearningTrainer",
    # Anomaly Detection Pre-training
    "ReconstructionAnomalyDetector",
    "PredictionAnomalyDetector",
    "HybridAnomalyDetector",
    "AnomalyDetectionPretrainer",
    # Integrated Trainer
    "SelfSupervisedTrainer",
]
