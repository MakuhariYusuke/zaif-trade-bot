"""
v456 環境設定の統一（Single Source of Truth）

すべての訓練・評価スクリプトはこの設定を参照
"""

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """訓練環境の設定（スケールされた仮想資金）"""
    
    # 資金
    INITIAL_BALANCE = 100000.0  # JPY（学習用スケール）
    
    # ポジション
    MAX_POSITION = 0.01  # BTC
    
    # リスク
    DRAWDOWN_LIMIT = 0.30  # 30%（初期学習時は寛容）
    
    # エピソード
    MAX_STEPS = 500
    PREWARM_STEPS = 100
    
    # 費用
    FEE_RATE = 0.001  # 0.1%
    SLIPPAGE_RATE = 0.0005  # 0.05%
    
    # TTL
    MAX_TTL_STEPS = 60  # ポジション保有最大ステップ
    COOLDOWN_STEPS = 5  # クールダウン
    
    # リキディティ制約
    MAX_DELTA_PER_STEP = 0.2  # ステップごとの最大変更率
    MIN_DELTA = 0.01  # デッドバンド
    
    # 訓練
    LEARNING_RATE = 3e-4  # SAC learning rate
    BUFFER_SIZE = 100000
    LEARNING_STARTS = 1000
    TOTAL_TIMESTEPS = 100000  # 10K → 100K に増加

@dataclass
class EvaluationConfig:
    """評価環境の設定（訓練と同じ）"""
    
    INITIAL_BALANCE = 100000.0
    MAX_POSITION = 0.01
    DRAWDOWN_LIMIT = 0.30
    MAX_STEPS = 500
    
    # 評価エピソード
    N_EPISODES = 50
    
    # OOS評価用
    USE_OOS_DATA = True
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    EMBARGO_DAYS = 7

@dataclass
class LiveConfig:
    """実運用環境の設定（本番資金）"""
    
    # ★ 訓練済みモデルを本番環境に適用する場合に使用
    # モデルの意思決定ロジックは変わらない
    
    INITIAL_BALANCE = 1000000.0  # JPY（本番資金、訓練時の10倍）
    MAX_POSITION = 0.1  # BTC（本番では大きめ）
    DRAWDOWN_LIMIT = 0.10  # 10%（本番は厳しい）
    
    FEE_RATE = 0.001
    SLIPPAGE_RATE = 0.0005

def get_training_config():
    """訓練設定を取得"""
    return TrainingConfig()

def get_evaluation_config():
    """評価設定を取得"""
    return EvaluationConfig()

def get_live_config():
    """本番設定を取得"""
    return LiveConfig()
