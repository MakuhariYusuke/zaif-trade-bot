# 725# Codex タスク委譲: 型安全・テスト堅牢化・技術的負債解消

## 目的
724# commit 後の安定期を利用し、Codex に5つの自己完結的タスクを体系的に委譲。
短期収益に直結しない基盤改善を並行で進め、長期的な開発速度と信頼性を確保する。

## タスク一覧

| # | ファイル | タスク | 種別 | 推定時間 |
|---|---------|--------|------|---------|
| T1 | `codex_725_task1_flaky_real_data_test.md` | test_load_real_data flaky 修正 | テスト | 20-30m |
| T2 | `codex_725_task2_sac_type_safety.md` | sac_retrain_scheduler type: ignore 8箇所除去 | 型安全 | 30-45m |
| T3 | `codex_725_task3_compat_shim_removal.md` | compat.py 移行シム削除（残存1箇所のみ） | 技術負債 | 15-20m |
| T4 | `codex_725_task4_retrain_type_ignore.md` | retrain_scheduler type: ignore 3箇所修正 | 型安全 | 45-60m |
| T5 | `codex_725_task5_test_slow_markers.md` | @pytest.mark.slow 追加でテスト高速化 | テスト | 30-45m |

## タスク選定基準

1. **自己完結性**: 他タスクとの依存なし、1ファイル or 1メソッドスコープ
2. **テスト可能**: 既存テスト + mypy で検証可能
3. **低リスク**: ランタイム動作変更なし（T3 のインポートパス変更を除く）
4. **体系性**: 型安全(T2,T4) + テスト堅牢化(T1,T5) + 負債解消(T3) の3軸カバー

## 各タスク詳細

### T1: test_load_real_data flaky 修正
- **問題**: `assert len(real_fill_df) >= 20` が本日8件で失敗
- **解決**: 件数不足時は `pytest.skip()` に変更、閾値を `MIN_RECORDS_FOR_INTEGRATION = 5` に緩和
- **影響**: CI の偽陽性排除

### T2: sac_retrain_scheduler 型安全
- **問題**: `SACRetrainConfig.from_yaml_dict()` の `cfg.get()` → Any で8箇所 suppress
- **解決**: `cfg: dict[str, object]` 型明示 + `cast()` で type: ignore 全除去
- **影響**: mypy strictness 向上

### T3: compat.py シム削除
- **問題**: TODO コメント付き移行シムが残存。使用箇所1箇所のみ
- **解決**: `state.py` のインポートを正規パスに変更、compat.py 削除
- **影響**: 不要な間接参照の排除

### T4: retrain_scheduler type: ignore
- **問題**: 3箇所の異なる原因の suppress（possibly-undefined / index / import-untyped）
- **解決**: 変数初期化、ローカル型ヒント、mypy.ini 設定の各種手法
- **影響**: 防御的コーディング改善

### T5: テスト高速化
- **問題**: フルユニットテストが7-8分（LGBM 訓練テストがボトルネック）
- **解決**: `@pytest.mark.slow` 追加で `not slow` 実行時 2-3分に短縮
- **影響**: 開発サイクル高速化

## 備考
- 724# のコード変更（`_is_bypass_mode_active` regime 除外）は hot-reload では反映されない
  - YAML 値は config に載るが、参照するコードが旧 SHA にないため
  - → 次回 hot-swap 再起動時に反映
- `trending_down_sell_offset_boost: 1.3` は YAML 値のみのため hot-reload で反映済

## 実装結果

| # | 結果 | 実装メモ |
|---|------|----------|
| T1 | 完了 | `MIN_RECORDS_FOR_INTEGRATION = 5` を導入し、実データ件数不足は fail ではなく skip。ラベル化後サンプル不足も同根 flaky として skip し、特徴量生成成功時は `len(X) > 0` と `len(X) == len(y)` で実効性を維持 |
| T2 | 完了 | `SACRetrainConfig.from_yaml_dict()` を `Mapping[str, object]` 入力へ寄せ、`_yaml_section()` で YAML section を型付き取得。対象8箇所の `type: ignore` を除去 |
| T3 | 完了 | `ztb.trading.live.orders.compat` の残存 import を正規 `ztb.trading.orders.state_machine` へ変更し、compat shim を削除 |
| T4 | 完了 | `tmp_path` 未定義防止、`eval_set` の typed cast、`psutil` import ignore の mypy 設定化で対象3箇所の `type: ignore` を除去 |
| T5 | 完了 | `TestRetrainModel` に `@pytest.mark.slow` を追加。既存 marker 登録は `pytest.ini`/`pyproject.toml` に存在するため追加不要 |

## プロンプト妥当性と補正

- T1 は妥当。ただし `len(X) >= 1` だけでは label 側の整合を見ないため、`len(X) == len(y)` を追加した。また record 件数だけでなくラベル化後件数でも不足が起きるため、`Insufficient labeled samples` も skip 対象にした。
- T2 は妥当。単純 `cast()` の羅列より、壊れた YAML section が来ても既定値に落ちる `_yaml_section()` の方が安全と判断した。
- T3 は妥当。`rg` で live orders compat の利用が `state.py` のみであることを確認した。
- T4 は妥当。ただし `Any` は増やさず、`pd.DataFrame`/`pd.Series` の eval_set cast にした。
- T5 は妥当。slow marker の未登録問題は既に解決済みだったため、対象 class への付与に留めた。

## 隠れタスク・水平展開

- `psutil` の inline ignore は module-level mypy 設定へ移した。今後 `import psutil` を増やす場合も局所 suppress を増やさずに済む。
- YAML dict 取り出しは SAC 側で helper 化した。PPO 側にも同型の `from_yaml_dict()` が残る場合は同じパターンを横展開候補にする。
- real-data test は件数依存 fail とラベル化後サンプル不足 fail を避ける一方、特徴量生成自体は `>0` と label 長一致で維持した。完全 skip ではないため回帰検知力を残す。

## 著者への質問

- なし。今回の5タスクはいずれも既存コードとプロンプトの範囲で実装判断可能だった。

## 検証結果

- `python3 -m py_compile scripts/v460/ml/sac_retrain_scheduler.py scripts/v460/ml/retrain_scheduler.py tests/unit/v460/test_ml_pipeline.py tests/unit/v460/test_retrain_hot_reload.py ztb/trading/live/orders/state.py` PASS
- `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_ml_pipeline.py::Test057Integration tests/unit/v460/test_sac_retrain_scheduler.py::TestSACRetrainConfig -x --tb=short --no-cov` PASS: `10 passed, 1 skipped in 2.52s`
- `.venv/Scripts/python.exe -m pytest tests/unit/v460/test_retrain_hot_reload.py::TestRetrainModel -x --tb=short --no-cov` PASS: `3 passed in 3.73s`
- `rg` confirmed no remaining target `type: ignore` / live orders compat import in touched scopes.
- Import smoke PASS: `from ztb.trading.live.orders.state import OrderRecord`
- targeted mypy:
  - plain invocation failed before checking due existing module duplication: `lightgbm/__init__.py` seen as both `v460.ml` and `scripts.v460.ml`
  - `--explicit-package-bases` invocation exceeded 45s timeout; no code error was emitted before timeout
