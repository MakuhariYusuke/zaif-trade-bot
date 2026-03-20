"""
軽量レジーム検知 — fill_test 実測サイクルの mid_price 系列からマーケット状態を分類.

035# §4 準拠.

設計原則:
  - 4 状態: trending / ranging / high_vol / unknown (035# §4.2 #1)
  - ヒステリシス: 連続 N サイクル一致で状態確定 (035# §4.2 #2)
  - 信頼度ゲート: confidence 低時は unknown で適応停止 (035# §4.2 #3)
  - レジーム別評価を必須化 (035# §4.2 #4)

市場理論的根拠:
  **Markov-Switching Model** — Hamilton (1989) "A New Approach to the Economic
  Analysis of Nonstationary Time Series and the Business Cycle".
  市場状態を隠れマルコフ過程としてモデル化。本モジュールは
  二次モーメントと線形回帰スロープを状態変数として使用し、
  隠れ状態を trending / ranging / high_vol / unknown に分類する。

  **Adaptive Market Hypothesis (AMH)** — Lo (2004) "The Adaptive Markets
  Hypothesis: Market Efficiency from an Evolutionary Perspective".
  市場効率性は時間変動し、レジームに依存する。レジーム検知は
  AMH が予測する「市場状態依存の最適戦略」を実現する基盤となる。

  **ヒステリシスの意義**: 状態確定に連続 N サイクルを要求するのは、
  Bayes 更新 (posterior が十分な evidence で確定するまで待つ) の離散近似。

既存資産再利用:
  - ztb/metrics/metrics.py::classify_market_regime の分類ロジックを軽量化
  - fill_test サイクル ≈120 秒で得られる mid_price のみを入力とする
"""""

from ztb.trading.signal.regime.regime_detector import *  # noqa: F401,F403
