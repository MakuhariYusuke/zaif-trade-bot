# 126# SkipGate 定期再学習 + Hot-Reload

> **種別**: impl (実装)  
> **フェーズ**: ph2 (G1.1-exec)  
> **セッション**: S125.1#  
> **前提**: 125# LGBM PnL120 モデル構築, 124# レビュー  
> **目的**: fill_test 稼働中に蓄積 fill_records から定期的にモデルを再学習し、無停止でモデルを差し替える

---

## 背景・課題

- 125# で LGBM PnL120 回帰モデルを構築しデプロイしたが、モデルは**静的** (デプロイ後は固定)
- fill_test の実行時間が数百時間に及ぶため、蓄積データに基づくモデル更新が必要
- 手動再学習→再デプロイは運用負荷が高く、タイムリーな改善が困難

## 解決策

2つの独立したコンポーネントで構成:

### 1. retrain_scheduler (バックグラウンドプロセス)

**ファイル**: `scripts/v460/ml/retrain_scheduler.py`

- fill_test と並行して別プロセスで実行
- 定期的 (デフォルト 1 時間) に fill_records を読み込み、モデルを再学習
- Walk-Forward OOS 評価で品質ゲート (新モデルが既存より劣化しないことを確認)
- アトミック書き込み (`tmp → os.replace`) で pkl ファイルを安全に差し替え
- 学習履歴を `logs/retrain_history.jsonl` に記録

#### パイプライン

```
fill_records_*.jsonl → enrich → build_features → 品質ゲート(WF-OOS) → train → atomic save
```

#### 品質ゲート

- Walk-Forward: 最初 80% で学習、最後 20% で OOS 評価
- OOS スコアが `min_score_improvement` (デフォルト: -0.05) 以上なら合格
- 不合格の場合は既存モデルを維持

#### 最小サンプル要件

- `min_total_samples`: 100 (特徴量構築+学習に必要な最低総サンプル数)
- `min_new_samples`: 30 (前回モデルからの新規サンプル数)

### 2. Hot-Reload (SkipGateEvaluator 内)

**ファイル**: `scripts/v460/lib/skip_gate_evaluator.py`

- `evaluate()` 呼び出し時にモデルファイルの SHA256 ハッシュをチェック (120 秒間隔)
- ハッシュ変更を検出したら新モデルをロード
- config overrides + warm_start を再適用
- ロード失敗時は既存モデルを維持 (安全フォールバック)
- fill_test プロセスの再起動は不要

## YAML 設定

`configs/v460/fill_test.yaml` に `retrain:` セクションを追加:

```yaml
retrain:
  interval_sec: 3600
  min_new_samples: 30
  min_total_samples: 100
  model_path: models/v460/skip_gate_lgbm_pnl120.pkl
  quality_gate_enabled: true
  min_score_improvement: -0.05
  wf_test_ratio: 0.2
  lgbm_n_estimators: 150
  lgbm_max_depth: 4
  lgbm_learning_rate: 0.05
  adaptive_threshold: true
  target_skip_rate_buy: 0.15
  target_skip_rate_sell: 0.20
```

## 使い方

```powershell
# fill_test が別ターミナルで稼働中に実行
.venv\Scripts\python.exe scripts/v460/ml/retrain_scheduler.py

# ワンショット実行 (テスト用)
.venv\Scripts\python.exe scripts/v460/ml/retrain_scheduler.py --once

# 設定ファイル指定
.venv\Scripts\python.exe scripts/v460/ml/retrain_scheduler.py --config configs/v460/fill_test.yaml
```

## アーキテクチャ図

```
┌──────────────────┐     ┌──────────────────────┐
│  fill_test       │     │  retrain_scheduler   │
│  (PID A)         │     │  (PID B)             │
│                  │     │                      │
│  SkipGateEval.   │     │  1h 周期:            │
│   ├ evaluate()   │     │  load_fill_records   │
│   │  └ hash check│     │  → enrich            │
│   │    120s 間隔 │◄────│  → quality gate(WF)  │
│   │    SHA256    │ pkl │  → train LGBM        │
│   └ hot-reload ──┤ swap│  → atomic save ──────┤
│                  │     │                      │
└──────────────────┘     └──────────────────────┘
         │                        │
         ▼                        ▼
   fill_records_*.jsonl    logs/retrain_history.jsonl
```

## テスト

- **14 テスト** (`tests/unit/v460/test_retrain_hot_reload.py`)
  - `TestHotReload` (7): 初期ハッシュ、未変更時スキップ、ファイル変更検出、失敗時フォールバック、チェック間隔、ハッシュ計算
  - `TestRetrainConfig` (2): デフォルト値、YAML オーバーライド
  - `TestBuildFullFeatures` (2): ベース特徴量 (16列)、OB 付き (19列)
  - `TestRetrainModel` (3): fill_records 不在、サンプル不足、新規サンプル不足

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/retrain_scheduler.py` | **新規** 定期再学習スケジューラ (~500行) |
| `scripts/v460/lib/skip_gate_evaluator.py` | hot-reload 機構追加 (SHA256 チェック + リロード) |
| `configs/v460/fill_test.yaml` | `retrain:` セクション追加 |
| `tests/unit/v460/test_retrain_hot_reload.py` | **新規** 14テスト |
| `docs/v460/126_ph2_impl_retrain_hot_reload.md` | **本ドキュメント** |

## 安全性

- **アトミック書き込み**: `os.replace()` で同一ファイルシステム上のアトミック差し替え (Windows 対応)
- **品質ゲート**: WF-OOS 評価で品質劣化モデルのデプロイを防止
- **フォールバック**: hot-reload 失敗時は旧モデルを維持
- **hasattr ガード**: テスト等で `__init__` がモックされた場合にも安全
- **fill_test 無停止**: 別プロセスなので fill_test に影響なし
