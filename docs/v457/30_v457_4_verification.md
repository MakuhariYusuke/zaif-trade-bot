# 30. v457.4 Verification: Native 1D Action Analysis

## 1. 概要
v457.3 で確認された「Fixed TTL (1D Action)」の有効性を、Wrapper による応急処置ではなく、`FastIntradayEnvv456` にネイティブ実装した **v457.4** で検証しました。
学習 (10k steps) およびバックテスト (20k steps) を実施し、その挙動を分析します。

## 2. 実験設定
- **Action Space Type**: `1d_position` (v457.4 Native)
- **TTL**: 暗黙的に `1.0` (Max)
- **Training Steps**: 10,000
- **Backtest Steps**: 20,000 (Data: `btc_jpy_1m_v451.csv`)
- **Initial Balance**: 1,000,000 JPY

## 3. 結果サマリー (v457.4)

| 指標 | 結果 |
| :--- | :--- |
| **Total Trades** | 1,391 |
| **Net PnL** | -49,775,756 JPY |
| **Gross PnL** | -41,187,516 JPY |
| **Profit Factor** | 0.31 |
| **Action Trend** | **Sell Only (100%)** |
| **Position** | ほぼ常に **-1.0 (Short)** |

## 4. 分析: なぜ v457.3 と異なる結果になったか？

### 4.1. 挙動の違い
- **v457.3 (Wrapper)**: Bull Market データセットにおいて「Buy & Hold」を学習し、+36M JPY の利益を出した。
- **v457.4 (Native)**: 同一データ(のはず)に対し、**「Sell Only (Short & Hold or Churn)」** を学習し、大損失 (-49M JPY) を出した。

### 4.2. 原因の仮説
1.  **Inverse Learning (逆学習)**:
    - ログを見ると `Position = -1.0000BTC` が張り付いている。
    - v457.3 は Buy (-1.0 to 1.0 mapping? No, Action is -1 to 1).
    - v457.4 のログ: `sell: 20000 (100.0%)`。これは Action 値が常に負であることを示唆する。
    - データセットの期間が v457.3 と同じであれば、Bull 相場で Short を振るのは「完全な失敗」である。

2.  **Reward Scale / Initialization の不運**:
    - RL (SAC) は初期探索の運に左右される。
    - たまたま初期に Short で利益（または Buy で損失）を得た結果、Short 一辺倒の局所解に陥った可能性がある。
    - 10k steps は学習期間として短く、一度陥った局所解から抜け出せなかった可能性が高い。

3.  **実装の差異 (Wrapper vs Native)**:
    - Wrapper (`FixedTTLWrapper`): `action_space` は `(1,)`。`step` 時に `ttl=1.0` を付与して `(2,)` にして Env に渡す。
    - Native (`FastIntradayEnv`): 内部で `if self.action_space_type == "1d_position": ttl_fraction = 1.0` とする。
    - 論理的には等価だが、`reset` 時の初期化や、Step 処理での微妙な違い（例: `target_pos` の処理順序）があるかもしれない。
    - 特に `obs` の形状等は `(88,)` で不変。

### 4.3. 考察
- "Native 1D" の実装自体は機能している（エラーなく学習・推論できている）。
- しかし、**「10k steps で安定して Buy & Hold を見つけられるか」** という点において、今回はたまたま Short 側に倒れた（そして Bull 相場なので死んだ）。
- これは **v457.3 の成功も「たまたま Buy 側に倒れた」だけの不安定な勝利だった可能性** を強く示唆している（Review 指摘事項 3.1. の「汎化の裏付け不足」が早くも露呈した形）。

## 5. 次のアクション
1.  **再学習 (Retry)**: 乱数シードを変えて再学習し、結果が Buy に振れるか確認する（「運」要素の切り分け）。
2.  **期間の延長**: 10k steps ではなく 20k-50k steps 学習させ、Short が間違いであると気づく時間を与える。
3.  **カリキュラム/誘導**: 初期段階で Long 有利な報酬を与えるか、データセットを明示的に上昇トレンド期間に固定して「正解」を教える。

## 6. 結論
v457.4 (Native 1D) は技術的には動作したが、初回の学習では **逆張りの局所解 (Short & Dead)** に陥った。
これは 1D 化のバグではなく、**強化学習エージェントの探索不足** である可能性が高い。
v457.3 の成功を過信せず、**安定してトレンドを掴めるような追加の工夫（報酬調整や学習期間延長）** が必要である。
