# 29. v457.4 Implementation: Native 1D Action Support

## 1. 概要
v457.3 のレビュー指摘事項に基づき、`FixedTTLWrapper` による一時的な対応から、`FastIntradayEnvv456` への **ネイティブ 1D Action (Position Only) サポート** へと実装を昇華させました。
これにより、環境設定のみで「1D (Buy & Hold 志向)」と「2D (TTL 指定)」を切り替え可能となり、コードベースの健全性が向上しました。

## 2. 変更内容

### 2.1. Environment (`FastIntradayEnvv456`)
- **Action Space Type 追加**: `__init__` に `action_space_type` 引数を追加。
  - `"2d_position_ttl"` (Default): 従来の2次元アクション (Position, TTL)。
  - `"1d_position"`: 新規1次元アクション (Positionのみ)。
- **ロジック変更**:
  - 1D モード時は `action[0]` を `target_position` として使用。
  - `ttl_fraction` は内部的に `1.0` (最大値) として扱われる（v457.3 の成功要因を継承）。

### 2.2. Factory (`EnvironmentFactory` / `utils`)
- Factory および Utility 関数を更新し、Config から `action_space_type` を受け取れるように変更。
- `fast_intraday_env_v456_utils.py` の `known_utils_keys` に `action_space_type` を追加。

### 2.3. Configuration (`config/v457_4`)
- 新しい設定ファイル `config/v457_4/train_config.json` を作成。
- `"environment"` セクションに `"action_space_type": "1d_position"` を明示。

### 2.4. Training Script (`scripts/v457/train_v457_4.py`)
- Wrapper のインポートと適用を削除。
- ネイティブ実装を用いた学習フローを構築。

## 3. 検証結果 (Dry Run)
- v457.4 スクリプトによる学習が正常に動作することを確認。
- ログ出力より `action=(1,)` となっていることを確認済み。

## 4. 今後の計画 (v457.4+)

### Phase 1: ベースライン評価 (現在)
- 1D アクション (Native) での学習を実行し、v457.3 (Wrapper版) と同等の性能が出ることを確認する。
- **指標**: Net PnL, Trade Count, Profit Factor。

### Phase 2: 汎化性能の検証 (Review 指摘対応)
- Bull 相場以外のデータセットでのバックテストを行う。
- 指摘のあった「レンジ/下落局面」での Buy & Hold の挙動（ドローダウン）を評価する。

### Phase 3: 頻度制御の洗練
- `min_delta` や `cooldown_steps` のパラメータチューニングを行い、1D アクション下での「適度な利確・損切り」を目指す。
- 完全に TTL に依存しない（無限 Hold）設定と、最大 TTL による強制決済の使い分けを検討。

## 5. 結論
指摘事項であった「1D Action のネイティブ実装」は完了しました。
既存実装（`FastIntradayEnv`）を最大限活用し、最小限の変更で機能追加を実現しました。
次は学習と多角的な評価フェーズに移行します。
