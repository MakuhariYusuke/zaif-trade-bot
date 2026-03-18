# 487# SAC メモリリーク防止 + 重複削減 + P0 観測基盤強化

> 種別: fix / refactor
> 対象: sac_common.py, sac_train.py, sac_retrain_scheduler.py, fill_quality.py, fill_record_builder.py, fill_cycle_executor.py
> コミット: ce10cdb69 (メモリ/重複), 後続コミット (観測基盤)
> 日付: 2026-03-19

---

## 0. 背景

483# / 484# / 485# の3レビューが一致して指摘した課題:

1. **寄与分解不能**: 481# (feasibility/veto緩和) と 482# (SAC sidecar) を同時投入したため、fill改善の原因を分離できない
2. **SAC評価系の脆弱性**: val_ratio=0.02, n_episodes=1 は採用判定に不十分
3. **sidecar live実効の不透明性**: offset実値・signal鮮度・confidence が記録されず検証不可能

486# で全数値・コード検証を行い、上記3点がコード上でも事実であることを確認した。

本ドキュメントでは:
- §1: メモリリーク防止 + 重複削減 (487# 第一コミット)
- §2: P0 観測基盤強化 — sidecar寄与分解に必要なフィールド追加
- §3: 今後の P1 対応方針

---

## 1. メモリリーク防止 + 重複削減

### 1.1 メモリリーク防止 (3件)

| 問題 | 場所 | 修正 |
|------|------|------|
| SB3 model ↔ env 循環参照が gc で回収されない | sac_train.py, retrain_scheduler.py | `cleanup_training_resources()` で model del → env close → gc.collect() |
| `best_model_loaded` が del されない | sac_train.py L210 付近 | finally 内に `del best_model_loaded` 追加 |
| `model` 変数が finally で参照不能 | sac_train.py | try 前に `model: SACModelProtocol | None = None` 宣言 |

**`cleanup_training_resources()`** (sac_common.py に新設):
```python
def cleanup_training_resources(
    *,
    models: list[object] | None = None,
    envs: list[TrainingEnvProtocol | None] | None = None,
    dataframes: list[object] | None = None,
) -> None:
    # 1. model del (env への参照を切る)
    # 2. env.close()
    # 3. del dataframes
    # 4. gc.collect() — 循環参照破壊
```

### 1.2 重複削減 (3件)

| 重複パターン | 統合先 (sac_common.py) |
|-------------|----------------------|
| sac_train._create_training_env / retrain._create_env の HeavyTradingEnv 生成 | `create_env_from_config(df, env_config)` |
| sac_train._create_sac_model / retrain のインライン SAC() | `create_sac_model(env, **kwargs)` |
| 両 finally の cleanup_envs + del df | `cleanup_training_resources()` に統合 |

**差分**: +156 / -59 行

---

## 2. P0 観測基盤強化 — sidecar 寄与分解

483#-485# 全文書が P0 に挙げた「寄与分解可能な観測設計」を実装する。

### 2.1 FillRecord に追加するフィールド

| フィールド | 型 | 目的 |
|-----------|---|------|
| `sidecar_confidence` | `float | None` | signal の confidence 値 (ROI margin から算出) |
| `sidecar_model_version` | `str | None` | deploy 元モデルバージョン |
| `sidecar_signal_status` | `str | None` | "fresh" / "stale" / "missing" / "error" |

→ **実装済み** (`ztb/metrics/fill_quality.py` FillRecord dataclass に3フィールド追加)

### 2.2 `read_sidecar_signal_with_status()` 新設

従来の `read_sidecar_signal()` は file-not-found / parse-error / TTL-exceeded を全て `None` に潰していた。
新関数 `read_sidecar_signal_with_status()` は `(SidecarSignal | None, str)` を返し、失敗モードを識別可能にする:

| 状態 | signal | status |
|------|--------|--------|
| 正常読込 | SidecarSignal | `"fresh"` |
| TTL超過 | None | `"stale"` |
| ファイル不在 | None | `"missing"` |
| JSON パースエラー / OS エラー | None | `"error"` |

→ **実装済み** (`scripts/v460/lib/sidecar_signal_io.py`)

### 2.3 データパイプライン貫通

新フィールドは以下のパスで FillRecord に到達する:

```
orchestrator_mid_cycle.py  (read_sidecar_signal_with_status → signal, status)
  → confidence, model_version を signal から抽出
  → run_single_cycle() に 3 パラメータ追加渡し
    → fill_cycle_executor.py (新パラメータ受取)
      → fill_record_builder._build_fill_market_fields() に貫通
        → build_fill_record() → FillRecord
```

変更ファイル一覧:
| ファイル | 変更内容 |
|---------|---------|
| `ztb/metrics/fill_quality.py` | FillRecord に 3 フィールド追加 |
| `scripts/v460/lib/sidecar_signal_io.py` | `read_sidecar_signal_with_status()` 新設 |
| `scripts/v460/lib/orchestrator_mid_cycle.py` | signal_status 取得 + 3 パラメータ渡し |
| `scripts/v460/lib/fill_cycle_executor.py` | `run_single_cycle()` 署名拡張 + ログ改善 |
| `scripts/v460/lib/fill_record_builder.py` | 2 メソッドで 3 フィールド貫通 |

### 2.4 Cycle log 改善

`_log_cycle_result()` に sidecar 要約を追加:
- signal status (fresh/stale — missing 時は省略)
- offset_bps 値 (0 以外の場合のみ)

例: `Cycle 142 result: filled=True, wait=3.2s, pnl=0.85bps, sidecar=fresh(+0.048bps)`

### 2.5 期待効果

これにより次回分析で:
- **sidecar_signal_status == "fresh" の fill** vs **"stale" or "missing" の fill** で PnL30s/AS率を比較可能
- **sidecar_offset_bps の分布** からモデルの active/inactive 比率を定量化可能
- **sidecar_model_version** で異なるモデル間の A/B 比較が可能

---

## 3. P1 今後の方針

| 優先度 | 項目 | 状態 |
|--------|------|------|
| P1-1 | SAC eval val_ratio ≥ 0.10 + n_episodes > 1 | YAML パラメータ変更で即時対応可能 |
| P1-2 | seed123 を対照群として live 比較 | sidecar_model_version フィールドで追跡基盤整備済み |
| P1-3 | 700JPY 緩和後の fill quality 点検 | sidecar_signal_status フィールドで分離解析が可能に |
| P2 | sidecar 権限昇格 (max_boost_bps 拡大) | 整合性確認後の将来課題 |
