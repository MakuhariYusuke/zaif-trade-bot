# 059# 実装サマリ — 058# レビュー対応 + 追加見落とし修正

## 対応概要

059# レビューで指摘された **9 件** (P0×3, P1×4, P2×2) 全てを修正。  
加えて独自の見落とし確認で発見した **6 件** を追加修正。  
計 **15 件** の不具合を解消。テスト 620 件全 PASS。

---

## §1 P0 修正（即日修正必須）

### P0-1: CV 外リーク → Pipeline 化
- **根本原因**: `fillna(median())` が CV 分割前 (全データ) で実行 → test fold が train fold の中央値を参照
- **修正**: 全 4 ファイルで bare `StandardScaler` → `Pipeline([SimpleImputer(median), StandardScaler, Model])` に統一
  - `data_loader.py`: `build_as_features`, `build_fill_features` — NaN 保持
  - `feature_enricher.py`: `build_enriched_as_features`, `build_pnl_features` — NaN 保持
  - `as_classifier.py`: CV ループ + final model を Pipeline 化
  - `fill_classifier.py`: 同上
  - `run_ml_pipeline.py`: PnL pipeline CV ループ + final model を Pipeline 化
  - `skip_gate.py`: `train_and_save_skip_gate` を Pipeline 化

### P0-2: skip 率制御の不整合 → 最終決定記録
- **根本原因**: `_recent_skips.append(should_skip)` が force-pass 判定の **前** → rate limiter 振動
- **修正**: append を force-pass 判定の **後** に移動、最終決定を記録

### P0-3: in-sample 評価 → OOF 統一
- **根本原因**: `evaluate_skip_policy` が全データ再予測 (学習データ含む)
- **修正**: `oof_probs` パラメータ追加、OOF 予測値で skip simulation 実施
- `run_ml_pipeline.py`: `run_as_pipeline` が OOF を渡すよう更新

---

## §2 P1 修正（中期改善）

### P1-4: OOF 有効件数の報告
- PnL pipeline 結果に `n_oof_valid` フィールド追加、ログに表示

### P1-5: 時刻特徴の定義不一致
- `build_features_from_market_state` に `market_timestamp` パラメータ追加
- `market_timestamp` 指定時は `datetime.fromtimestamp()` で時刻計算 → 学習側と整合

### P1-6: trade_window_sec 未使用
- `market_timestamp` 指定時に `recent_trades` を `trade_window_sec` でフィルタ

### P1-7: O(N×M) → O(N log M) 最適化
- `_compute_trade_features` に `_sorted_ts` パラメータ追加
- `enrich_fill_records` で事前ソート + `np.searchsorted` でウィンドウスライス

---

## §3 P2 修正（品質向上）

### P2-8: pickle 信頼境界
- `SkipGate.save()`: SHA256 ハッシュファイル (`.pkl.sha256`) を同時保存
- `SkipGate.load()`: ハッシュ検証、不一致で `ValueError` raise
- 後方互換: ハッシュファイル不在時はスキップ

### P2-9: テスト追加 (12 件)
- `Test059LeakDetection` (2): NaN 保持検証 (data_loader, feature_enricher)
- `Test059SkipRateHistory` (2): 最終決定記録、振動なし検証
- `Test059TimestampConsistency` (2): market_timestamp 整合、trade_window_sec フィルタ
- `Test059PickleHash` (3): ハッシュ作成、改竄検出、後方互換
- `Test059SearchsortedOptimization` (2): brute-force 一致、空ウィンドウ

---

## §4 追加見落とし修正 (NEW-01〜NEW-07)

| ID | 重要度 | 対象 | 修正内容 |
|---|---|---|---|
| NEW-01 | HIGH | as_classifier.py | 戻り値型ヒント 3→4 要素に修正 (Pipeline + oof_probs) |
| NEW-02 | HIGH | skip_gate.py | Pipeline 全体を保存・推論時使用 (Imputer 欠落解消) |
| NEW-03 | HIGH | data_loader, feature_enricher | 整数 hour → 小数 hour (hour+minute/60.0) で推論側と統一 |
| NEW-04 | HIGH | skip_gate.py | 未提供特徴量を 0.0 → NaN に変更 (Pipeline Imputer で処理) |
| NEW-05 | MEDIUM | data_loader.py | `df.get()` → 明示的カラム存在チェックに修正 |
| NEW-07 | MEDIUM | as_classifier.py | OOF skip simulation で PnL NaN もフィルタ |

### 未対応 (影響小、次回以降)
- NEW-06: Fold スキップ時の n_folds 報告
- NEW-08: as_classifier/fill_classifier の戻り値意味不統一 (Pipeline vs Scaler)
- NEW-09〜16: DRY (hour/regime encoding 共通化), 軽微品質

---

## §5 変更ファイル一覧

| ファイル | 変更概要 |
|---|---|
| `scripts/v460/ml/data_loader.py` | NaN保持, Pipeline utility, 小数hour, cancel_reason安全化 |
| `scripts/v460/ml/as_classifier.py` | Pipeline CV, OOF probs返却, 型ヒント修正, PnL NaNフィルタ |
| `scripts/v460/ml/fill_classifier.py` | Pipeline CV, Pipeline返却 |
| `scripts/v460/ml/feature_enricher.py` | NaN保持, searchsorted最適化, 小数hour |
| `scripts/v460/ml/skip_gate.py` | skip率修正, Pipeline保存/推論, hash検証, NaN default, market_timestamp, trade_window filter |
| `scripts/v460/ml/run_ml_pipeline.py` | Pipeline CV (PnL), OOF受け渡し, n_oof_valid報告 |
| `tests/unit/v460/test_enricher_skip_gate.py` | 12テスト追加 (23→35件) |
| `tests/unit/v460/test_ml_pipeline.py` | 4-tuple unpacking修正 |

**テスト結果**: 620 passed / 0 failed
