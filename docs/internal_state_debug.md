# 内部状態デバッグ機能のドキュメント

## 概要

`SELL`アクションに偏る問題（SELLロック）の調査のため、エージェント（SACモデル）の内部状態を詳細にログ出力する機能を実装しました。これにより、モデルが各ステップでどのような判断を下しているかを可視化し、問題の根本原因を特定する手がかりを得ます。

具体的には、以下の内部状態をログに出力します。

-   `actor_pre_tanh`: アクターネットワークの最終出力（`tanh`活性化関数適用前）。この値が極端に大きいか小さい場合、特定のアクションに強くバイアスがかかっていることを示します。
-   `actor_log_std`: アクション分布の対数標準偏差。アクションの探索範囲の広さを示します。
-   `critic_q1`: クリティックネットワークが評価したQ値。特定のアクションに対する期待収益を示します。

## 有効化する方法

このデバッグ機能は、パフォーマンスへの影響を避けるため、デフォルトでは無効になっています。必要に応じて、以下の方法で有効化できます。

### 1. `quick_train_v444_configurable.py` を使用する場合

トレーニングスクリプト実行時に、`--debug-internal-state`フラグを追加します。

```bash
python quick_train_v444_configurable.py --config "path/to/your/config.json" --debug-internal-state
```

### 2. `ztb.training.unified_trainer` を使用する場合

`unified_trainer`のCLI（`ztb/training/unified_trainer/main.py`）を使用する場合も同様に、`--debug-internal-state`フラグを追加します。

```bash
python -m ztb.training.unified_trainer.main --config "path/to/your/config.json" --debug-internal-state
```

### 3. 設定ファイルで有効化する場合

環境設定(`EnvironmentConfig`)に直接フラグを設定することも可能です。`ztb/trading/environment/utils/config.py`内の`EnvironmentConfig`データクラスに`debug_internal_state: bool = True`と記述します。

```python
# ztb/trading/environment/utils/config.py

@dataclasses.dataclass
class EnvironmentConfig:
    # ...
    debug_internal_state: bool = True  # Trueに設定
    # ...
```

## ログの確認方法

デバッグ機能を有効にしてトレーニングを実行すると、標準出力のログに以下のような形式で内部状態が出力されます。

```
DEBUG - {'step': 3825, 'continuous_action': '-1.000000', ..., 'actor_pre_tanh': [-10505.73046875], 'actor_log_std': [-20.0], 'critic_q1': [-20368.673828125]}
```

これらのログを分析することで、エージェントの行動決定プロセスを詳細に追跡できます。

## 今後の展望

このデバッグ機能を用いて得られた知見をもとに、以下の順で問題解決に取り組みます。

1.  **観測情報の正規化**: 入力データが適切な範囲にスケーリングされているか確認・修正します。
2.  **ネットワークの初期化と構造**: モデルのネットワーク構造や初期化方法が学習の安定性に与える影響を調査します。
3.  **報酬関数の見直し**: 上記の根本的な対策を施した上で、報酬設計がエージェントの行動に与える影響を再評価します。
