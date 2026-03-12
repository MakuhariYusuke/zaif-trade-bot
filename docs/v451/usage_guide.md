# v451 Usage Guide

## 環境構築
v451 は新しい特徴量セットを使用するため、以下の手順で環境をセットアップしてください。

### 1. データ生成
v451用の特徴量を含むデータセットを生成します。
```bash
python scripts/generate_v451_data.py
```
これにより `data/btc_jpy_1m_v451.csv` が作成されます。

### 2. 学習の実行
以下のコマンドで学習を開始します。
```bash
python experiments/v451/run_training_v451.py
```
- 設定ファイル: `config/v451/model_config.json` (参照用)
- 出力先: `models/sac_v451_phase7_regime_aware.zip`
- ログ: `checkpoints/v451/phase7`

### 3. 推論・評価 (Evaluation)
学習済みモデルを評価するには、以下のスクリプトを使用します（現在はDLLエラーのため修正中ですが、基本形は以下の通りです）。
```bash
python analysis/evaluate_v451_chronos.py
```

## 設定のカスタマイズ
`ztb/features/feature_set_config.py` にて `v451` の特徴量定義を確認・変更できます。

```python
'v451': {
    'columns': [
        'close', 'volume', 'high', 'low',
        'hour_sin', 'hour_cos',          # 時間特徴量
        'day_of_week_sin', 'day_of_week_cos',
        'volatility_rank',               # レジーム特徴量
        'regime_trend', 'regime_volatility',
        # ... 既存のテクニカル指標 ...
    ]
}
```
