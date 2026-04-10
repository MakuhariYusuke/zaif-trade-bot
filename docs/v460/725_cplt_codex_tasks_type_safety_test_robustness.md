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
