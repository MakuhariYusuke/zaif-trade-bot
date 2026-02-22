# 141# ph2 impl: side 別モデル分離 + regime 別閾値 + online monitor

| key | value |
|---|---|
| 番号 | 141 |
| フェーズ | ph2 |
| 種別 | impl |
| 対象 | 134# Phase E 残 P1 タスク (P1-01/02, P1-04, P1-12) |
| 作成日 | 2026-02-23 |
| 前提 | Git `83962ec88` (140# commit), テスト 1171 baseline |
| 結論 | **P1-01/02 (buy/sell 分離モデル) + P1-04 (regime 別閾値) + P1-12 (online monitor) を実装。134# Phase E P1 群を全完了。テスト 1171→1213 (+42)。** |

---

## §0 エグゼクティブサマリ

134# ロードマップ Phase E の残 P1 タスクを全件実装。

| P1 ID | 施策 | 実装セッション | 状態 |
|---|---|---|---|
| P1-01 | buy/sell 分離モデル | **141#** | ✅ |
| P1-02 | target 二層化 (buy=pnl30, sell=pnl120) | **141#** | ✅ |
| P1-03 | score 校正 (isotonic regression) | 138# | ✅ |
| P1-04 | regime 別 PnL 閾値 | **141#** | ✅ |
| P1-06 | reprice 売側上限 (2→1) | 137# | ✅ |
| P1-08 | spread 狭小時の休止判定 | 137# | ✅ |
| P1-10 | preflight 失敗連続 → run pause | 138# | ✅ |
| P1-11 | PnL fee/slippage 控除統一 | 137# | ✅ |
| P1-12 | 直近 N fill online 比較 | **141#** | ✅ |

**P1 全 9 項目完了** (P1-05, P1-07, P1-09 は P2 降格済み)。

---

## §1 P1-01/02: buy/sell 分離モデル + target 二層化

### §1.1 背景

139# ログ診断: buy PnL = -0.073bps, sell PnL = -0.556bps (sell が 7.6 倍劣悪)。
統一モデルは sell の特性 (逆選別パターン、回復 horizon の違い) を十分に捕捉できない。

### §1.2 設計

```
[統一モデル]                    [分離モデル (141#)]
  retrain_model(all) ────→      retrain_model(all)       ← 統一 (フォールバック用)
                                 _retrain_side_specific:
                                    retrain_model(buy)    ← target=pnl30
                                    retrain_model(sell)   ← target=pnl120
                                
  evaluate(side) ──────→      _select_gate_for_side(side):
                                    buy  → gate_buy (or unified)
                                    sell → gate_sell (or unified)
```

- **統一モデルは維持**: side 別モデルが未生成/ロード失敗時のフォールバック
- **target 二層化**: buy=pnl30 (即座に平均回帰), sell=pnl120 (120s の回復 horizon)
- **warm_start 無効化**: side 別モデルは unified と feature 空間が異なるため

### §1.3 変更ファイル

| ファイル | 変更概要 |
|---|---|
| `scripts/v460/lib/fill_config.py` | `skip_gate_model_path_buy/sell` 追加, YAML `sg_map` 拡張 |
| `scripts/v460/ml/retrain_scheduler.py` | `side_filter` パラメータ, `_retrain_side_specific()`, `_DEFAULT_CONFIG` 拡張 |
| `scripts/v460/lib/skip_gate_evaluator.py` | `_gate_buy/_gate_sell`, `_load_side_models()`, `_select_gate_for_side()`, `_check_and_reload_side_models()`, `model_used` タグ |
| `configs/v460/fill_test.yaml` | `skip_gate.model_path_buy/sell`, `retrain.side_specific_enabled/target_buy/target_sell/side_min_samples` |

### §1.4 retrain_scheduler 変更詳細

- `retrain_model(cfg)`: `side_filter` パラメータ追加 — balance_forced_switch 除外後に side フィルタリング
- cache key: `|{side_filter or 'all'}` 接尾辞で買/売/全体のキャッシュ衝突を防止
- `_retrain_side_specific()`: 各 side に対して専用 model_path + target で retrain_model() 呼び出し
- `run_scheduler()` / `main() --once`: retrain 後に `_retrain_side_specific()` を呼び出し

### §1.5 SkipGateEvaluator 変更詳細

- `__init__`: 6 新属性 (`_gate_buy`, `_gate_sell`, path/hash ×2)
- `_load_side_models()`: pkl ファイルが存在する場合のみロード (graceful fallback)
- `_select_gate_for_side(side)`: side 別ゲートが存在すれば優先、なければ unified
- `evaluate()`: `active_gate = _select_gate_for_side(side)` でディスパッチ、`model_used` に `side_buy`/`side_sell`/`unified` タグ
- `_check_and_reload_side_models()`: hot-reload 拡張 — 新規モデル出現と更新の両方を検出

---

## §2 P1-04: regime 別 PnL 閾値オーバーライド

### §2.1 背景

レジームごとに予測精度が異なる:
- **high_vol**: ボラ上昇 → 予測精度低下 → 厳格化が必要
- **trending**: トレンド追従に優位性 → 緩和可能
- **ranging**: レンジ相場 → やや厳格化

### §2.2 設計

```python
# SkipGate.evaluate() の PnL 閾値決定ロジック
base_threshold = self.config.threshold_bps          # デフォルト (0.0)
if regime and self.config.regime_thresholds:
    base_threshold = self.config.regime_thresholds.get(regime, base_threshold)
# adaptive_threshold は base_threshold を起点に動的調整
```

- `SkipGateConfig.regime_thresholds: dict[str, float]` — レジーム名 → 閾値のマッピング
- `SkipGate.evaluate()`: `regime: str | None = None` パラメータ追加
- `SkipGateEvaluator.evaluate()`: `sg_regime` を `regime=` で引き渡し
- `FillTestConfig.skip_gate_regime_thresholds`: YAML 設定経路
- `_apply_config_overrides()`: regime_thresholds のオーバーライド追加

### §2.3 YAML 設定

```yaml
skip_gate:
  regime_thresholds:
    high_vol: 0.2       # 高ボラ: 予測保守化
    ranging: 0.1         # レンジ: やや厳格化
    trending: -0.1       # トレンド: 緩和 (追従優位)
```

---

## §3 P1-12: 直近 N fill オンラインパフォーマンスモニター

### §3.1 背景

134# P1-12: 「非定常環境では全履歴平均は過去に引きずられる。直近 N fill のみで online 比較が必要。」
sell_dynamic_kill (P0-10) が sell 側の rolling window 監視を部分的に担うが、skip gate 全体の判定品質評価は未実装。

### §3.2 設計

新規モジュール `ztb/ml/online_monitor.py`:

```python
class OnlineMonitor:
    """直近 N fill ベースの skip gate パフォーマンスモニター."""
    
    def evaluate(self, records: pd.DataFrame) -> OnlineMonitorResult:
        # 1. 直近 window 件に絞り込み
        # 2. skip/pass 分離
        # 3. pass 群: 実 PnL 分析 (mean, win_rate)
        # 4. skip 群: 予測スコア精度分析 (precision)
        # 5. degradation 判定
        # 6. side 別サマリー
```

`OnlineMonitorResult`:
- `n_total`, `n_passed`, `n_skipped`: 件数
- `pass_mean_pnl`: pass 群の平均 PnL (bps)
- `pass_win_rate`: pass 群の勝率
- `skip_precision`: skip したうち score < 0 だった割合
- `degraded`: degradation 判定
- `side_summary`: side 別内訳

### §3.3 retrain_scheduler 統合

`_run_online_monitor(cfg)`:
- retrain サイクル完了後に自動実行
- `fill_records` を読み込み、`OnlineMonitor.evaluate()` で直近 N fill を評価
- degraded 時は WARNING ログ出力
- `run_scheduler()` ループと `--once` モードの両方に統合

### §3.4 設定

```yaml
retrain:
  online_monitor_enabled: true
  online_monitor_window: 100
  online_monitor_pnl_column: post_fill_30s_pnl
  online_monitor_degraded_threshold_bps: -0.3
```

---

## §4 テスト

### §4.1 テストファイル

`tests/unit/v460/test_141_side_specific_models.py` — 42 テスト

| セクション | テスト数 | 対象 |
|---|---|---|
| §1 FillConfig side paths | 3 | default_none, explicit_paths, yaml_parsing |
| §2 retrain side_filter | 2 | side_filter_in_result, insufficient_samples |
| §3 _retrain_side_specific | 4 | calls_retrain, target_per_side, no_model_path, history |
| §4 Evaluator dispatch | 4 | select_buy, select_both, unified, file_missing |
| §5 YAML config | 2 | load_side_fields, no_side_fields |
| §6 Integration | 2 | model_used_tag_side, model_used_tag_unified |
| §7 Hot-reload | 2 | new_model, updated_model |
| §8 warm_start/cache | 2 | disabled, cache_key_differs |
| §9 regime thresholds config | 2 | SkipGateConfig default_empty, explicit |
| §10 FillConfig regime | 2 | default_empty, yaml_parsing |
| §11 regime evaluate | 4 | no_regime, override, relaxed, empty_thresholds |
| §12 config overrides | 2 | applied, empty_dict |
| §13 evaluator integration | 1 | regime_passed_to_gate |
| §14 online_monitor config | 1 | default_values |
| §15 online_monitor evaluate | 8 | basic, degraded, insufficient, empty, side_summary, precision, to_dict, window |
| §16 retrain integration | 1 | default_config_keys |

### §4.2 既存テスト修正

`tests/unit/v460/test_skip_gate_v3.py`: `TestSkipSellUnknownRegime` の 2 テスト — mock 化された SkipGateEvaluator に 141# で追加された `_gate_buy`, `_gate_sell`, `_gate_path`, `_model_file_hash` 等の属性を設定。

### §4.3 テスト結果

```
1213 passed, 0 failed, 91 warnings in 176s
```

テスト増減: 1171 (140#) → 1213 (+42)

---

## §5 変更ファイル一覧

| ファイル | 行数(概算) | 変更内容 |
|---|---|---|
| `scripts/v460/lib/fill_config.py` | +8 | model_path_buy/sell, regime_thresholds, YAML mapping |
| `scripts/v460/ml/retrain_scheduler.py` | +100 | side_filter, _retrain_side_specific, _run_online_monitor, _DEFAULT_CONFIG |
| `scripts/v460/lib/skip_gate_evaluator.py` | +95 | side dispatch, side model load/reload, regime pass-through, config override |
| `scripts/v460/ml/skip_gate.py` | +10 | regime param, regime_thresholds lookup, SkipGateConfig field |
| `ztb/ml/online_monitor.py` | 215 | **NEW**: OnlineMonitor, OnlineMonitorConfig, OnlineMonitorResult |
| `configs/v460/fill_test.yaml` | +15 | side model paths, regime thresholds, online monitor config |
| `tests/unit/v460/test_141_side_specific_models.py` | 1089 | **NEW**: 42 tests (P1-01/02/04/12) |
| `tests/unit/v460/test_skip_gate_v3.py` | +12 | mock 属性追加 (既存テスト修正) |

---

## §6 134# ロードマップ完了状況

### Phase A-D: ✅ 全完了 (135#-140#)

### Phase E P1: ✅ **全 9 項目完了**

| ID | 施策 | セッション | 状態 |
|---|---|---|---|
| P1-01 | buy/sell 分離モデル | 141# | ✅ |
| P1-02 | target 二層化 | 141# | ✅ |
| P1-03 | score 校正 | 138# | ✅ |
| P1-04 | regime 別閾値 | 141# | ✅ |
| P1-05 | skip_gate_ratio 自動 degrade | (P2 降格) | - |
| P1-06 | reprice 売側上限 | 137# | ✅ |
| P1-07 | timeout 動的化 | (P2 降格) | - |
| P1-08 | spread 狭小時休止 | 137# | ✅ |
| P1-09 | balance margin 動的化 | (P2 降格) | - |
| P1-10 | preflight pause | 138# | ✅ |
| P1-11 | PnL fee 控除統一 | 137# | ✅ |
| P1-12 | online fill 比較 | 141# | ✅ |

### 次ステップ: P2 群 + 運用検証

1. **運用検証**: side 別モデル + regime 閾値 + online monitor のライブ動作確認
2. **P2 群**: P2-09 (retrain lag 監視), P2-10 (OB micro-feature), etc.
3. **24h 再計測** (Phase C): retrain 修正後のフル 24h パフォーマンス測定

---

## §7 レビュー向け注意点

1. **side 別モデルファイルは retrain 実行後に生成される** — 初回起動時は unified のみ。hot-reload が新規ファイル生成を検出して自動ロード。
2. **regime_thresholds は adaptive_threshold との共存** — regime で base を決定後、adaptive がさらに動的調整。
3. **online_monitor は non-fatal** — load_fill_records 失敗やモジュールエラーでも retrain は中断しない。
4. **warm_start は side 別モデルで無効** — unified モデルの Booster は feature 空間が異なる可能性があるため。

---

## §8 142# 自己チェック修正 (2026-02-23)

141# 実装をセルフレビューした結果、**CRITICAL 1 件 + MEDIUM 2 件** を検出し即時修正。

### §8.1 修正内容

| # | 重大度 | 問題 | 修正 |
|---|---|---|---|
| C-1 | **CRITICAL** | `regime_thresholds` で上書きした `base_threshold` が `_calibrate_pnl_threshold()` に渡されず、`adaptive_threshold=True` 時に P1-04 が事実上の死コードになる | `_calibrate_pnl_threshold(side, pnl, base_threshold=base_threshold)` に引数追加。C-1 テスト 3 件追加 |
| M-1 | MEDIUM | `_select_gate_for_side()` が `self._gate_buy` に直接アクセス — `__init__` バイパス時の `AttributeError` リスク | `getattr(self, "_gate_buy", None)` ガードに変更。テスト 1 件追加 |
| M-3 | MEDIUM | `regime_thresholds` に未知キー (typo) が無警告で無視される | `_apply_config_overrides` でキーバリデーション + WARNING ログ追加。テスト 1 件追加 |

### §8.2 テスト追加

| テストクラス | テスト数 | 内容 |
|---|---|---|
| `TestRegimeAdaptiveThresholdIntegration` | 5 | C-1 回帰テスト (regime + warmup), M-1 属性欠損テスト, M-3 typo WARNING テスト |

### §8.3 検証

```
test_141_side_specific_models.py: 47 passed (42→47, +5)
tests/unit/v460: 1218 passed (1213→1218, +5)
```

### §8.4 残存課題 (低優先度)

| # | 重大度 | 内容 | 対応方針 |
|---|---|---|---|
| M-2 | LOW | `_run_online_monitor` が全 run のデータをロード (latest_run_only 未適用) | 次回改善候補 |
| M-4 | LOW | side 別モデルに `_apply_warm_start` が呼ばれない (意図的だがコメント不足) | ドキュメント化で対応 |
| L-1 | LOW | `_check_and_reload_side_models` でメソッド内 import | パフォーマンス影響なし、将来リファクタ |
| L-2 | LOW | online_monitor テストが 141 テストファイルに統合 | 独立テストファイル化は将来対応 |

