// RL/機械学習用の統一スキーマ定義
// Pythonでの学習を想定した命名規則と型定義

export interface FeatureRow {
  // キー情報
  ts: string;                 // ISO 8601 (UTC) - Python datetimeに変換しやすい
  exchange: 'coincheck' | 'zaif';
  pair: string;               // 'BTC/JPY' 形式 (Python標準)

  // 市場データ
  price: number;
  volume: number;
  spread: number | null;      // 売買スプレッド (bid-ask)
  depth_imbalance: number | null; // 板の不均衡度 (-1 to 1)
  order_flow: number | null;  // 注文フローの方向性

  // テクニカル指標 (Python MLライブラリでよく使う名前)
  sma_10?: number | null;     // 短期SMA
  sma_50?: number | null;     // 長期SMA
  rsi_14?: number | null;     // RSI 14期間
  atr_14?: number | null;     // ATR 14期間
  bb_width_20?: number | null; // Bollinger Band幅 20期間

  // リスク・流動性指標
  vol_ratio?: number | null;  // ボラティリティ比 (ATR/価格)
  liquidity_score?: number | null; // 流動性スコア (0-1)

  // 取引結果 (バックテストや実取引で埋められる)
  pnl?: number | null;        // 損益
  win?: number | null;        // 勝敗フラグ (1/0)
}

// RL (Reinforcement Learning) 用のステップデータ
export interface RLStep {
  ts: string;
  pair: string;
  episode_id: string;         // エピソード識別子

  // 状態 (正規化済み特徴量ベクトル)
  state: number[];           // 現在の特徴量ベクトル

  // 行動
  action: -1 | 0 | 1;        // sell / hold / buy

  // 報酬
  reward: number;            // 即時報酬 (PnLやSharpe近似)

  // 次の状態
  next_state: number[];      // 次の特徴量ベクトル

  // エピソード終了フラグ
  done: 0 | 1;              // 0: 継続, 1: エピソード終了
}

// 正規化設定 (Pythonでの前処理に使用)
export interface NormalizationConfig {
  method: 'zscore' | 'minmax' | 'robust'; // 正規化方法
  params: {
    mean?: number[];         // zscore用平均
    std?: number[];          // zscore用標準偏差
    min?: number[];          // minmax用最小値
    max?: number[];          // minmax用最大値
    median?: number[];       // robust用中央値
    mad?: number[];          // robust用MAD (Median Absolute Deviation)
  };
}

// 特徴量メタデータ (Pythonでの型推論に使用)
export interface FeatureMetadata {
  name: string;
  type: 'numeric' | 'categorical';
  nullable: boolean;
  description: string;
  normalization?: NormalizationConfig;
}

// デフォルトの特徴量メタデータ
export const DEFAULT_FEATURE_METADATA: FeatureMetadata[] = [
  { name: 'price', type: 'numeric', nullable: false, description: 'Market price' },
  { name: 'volume', type: 'numeric', nullable: false, description: 'Trading volume' },
  { name: 'spread', type: 'numeric', nullable: true, description: 'Bid-ask spread' },
  { name: 'depth_imbalance', type: 'numeric', nullable: true, description: 'Order book imbalance (-1 to 1)' },
  { name: 'order_flow', type: 'numeric', nullable: true, description: 'Order flow direction' },
  { name: 'sma_10', type: 'numeric', nullable: true, description: '10-period Simple Moving Average' },
  { name: 'sma_50', type: 'numeric', nullable: true, description: '50-period Simple Moving Average' },
  { name: 'rsi_14', type: 'numeric', nullable: true, description: '14-period RSI' },
  { name: 'atr_14', type: 'numeric', nullable: true, description: '14-period ATR' },
  { name: 'bb_width_20', type: 'numeric', nullable: true, description: '20-period Bollinger Band width' },
  { name: 'vol_ratio', type: 'numeric', nullable: true, description: 'Volatility ratio (ATR/price)' },
  { name: 'liquidity_score', type: 'numeric', nullable: true, description: 'Liquidity score (0-1)' },
  { name: 'pnl', type: 'numeric', nullable: true, description: 'Profit and Loss' },
  { name: 'win', type: 'numeric', nullable: true, description: 'Win flag (1/0)' },
];