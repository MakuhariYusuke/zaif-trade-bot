# v452 TODO List

## 優先度: 高 (Immediate Actions)

- [ ] **設定値の恒久対応**
    - `config.py` のデフォルト値を修正し、`max_action_threshold` が 1.0 になるようにコミットする。
    - `adapters.py` などのバックテスト用アダプター内のハードコード値も修正する。
    - バリデーションロジック (`ThresholdManager._validate_config`) を見直し、勝手なクランプを行う場合は ERROR ログを出すか、例外を投げるように変更を検討する。

- [ ] **リポジトリの衛生化（肥大化対策）**
    - 現状、`venv311/`, `venv311_new/`, `.venv/` や `docs/_build/`, `data/metadata` などが含まれており履歴が肥大化している。
    - 直ちに `.gitignore` を更新し、`git rm --cached` により追跡を停止（履歴クリーンアップは別運用で実施）。
    - モデルやチェックポイント(例: `checkpoints/`, `models/`, `temp_model/`) はアーティファクトストレージへ移動。

- [ ] **ライブ取引の安全化（必須）**
    - 本番取引を実行する際は `--allow-production` CLI フラグまたは環境変数 `ZTB_ALLOW_PRODUCTION=1` が必須。
    - APIキーがあるだけで自動的に本番モードに入らず、明示的なフラグがない場合は起動を拒否する実装と、CIでの保護を追加する。

- [ ] **単一責任の浸透 & 巨大ファイル対応**
    - `ztb/trading/comprehensive_backtest.py` 等の 1 ファイルに集中しているロジックを、`runner` / `engine` / `metrics` / `adapter` 等に切り出す。テストを追加しながら段階的に分割する。
    - DI を導入してグローバル singleton を減らす（テスト容易性と安全性向上）。

- [ ] **型安全・静的チェックの段階的導入**
    - 重要モジュールから `mypy` を段階的に強化する。まずは `ztb/utils` と `ztb/trading` のクリティカルパスを対象に `disallow_untyped_defs` をオンにする。
    - `typing.Any` の使用箇所を減らし、`TypeGuard` や `pydantic` を活用する。

- [ ] **Market Regime Detector の精度検証**
    - 現在の `ThresholdManager` は `MarketRegimeDetector` の結果に依存している。
    - レンジ判定（Ranging）が適切に行われているか、可視化ツール等で検証する。

## 優先度: 中 (Optimization)

- [ ] **動的閾値パラメータのチューニング**
    - 現在は「10倍」という極端な値で成功しているが、これが最適値とは限らない。
    - 3倍、5倍、10倍などで比較検証を行い、機会損失とリスク回避のバランスを探る。

- [ ] **ロギングの適正化**
    - デバッグのために `INFO` レベルで大量のログを出している箇所を整理し、本番運用に耐えうるログ設計にする。

## 優先度: 低 (Future Features)

- [ ] **転移学習 (Fine-tuning) の検討**
    - 現在のモデルを親モデルとして、直近の相場データのみで短期間の追加学習を行うパイプラインの構築。
    - ステップ数を抑えつつ、最新の相場に対応させる。
