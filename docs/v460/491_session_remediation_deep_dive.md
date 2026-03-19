# 491# Session: Fill Pipeline Remediation & Defense Layer Deep Dive

**日付**: 2026-03-19  
**セッション**: 488#→491# レビュー修正対応・深堀り・sidecar 復旧

---

## §1 セッション概要

本セッションでは 488#/489#/490# レビュー指摘に基づく修正を実施し、
fill pipeline の防御レイヤー多重共線性問題を定量的に解析した。

### 実施内容

| # | 項目 | 状態 | 対象 |
|---|------|------|------|
| 1 | sidecar retrain_scheduler 復旧 | ✅ 完了 | ops/windows/retrain_scheduler.ps1, sac_retrain_scheduler.py |
| 2 | P0-1 VPIN boost 勾配有効化 | ✅ 完了 | configs/v460/fill_test.yaml |
| 3 | P1-6 Config 相互参照バリデーション | ✅ 完了 | fill_config_validation.py + test |
| 4 | P1-7 Offset clamping 統一 | ✅ 非バグ確認 | 421# で完了済み (resolve_offset_ceiling) |
| 5 | P1-4/P1-5 Reanchor/Cooldown | ✅ 設計確認 | 設計通り動作、チューニング対象 |
| 6 | 防御レイヤー多重共線性分析 | ✅ 完了 | 46+ gate の定量分析 |
| 7 | retrain_scheduler Timestamp 修正 | ✅ 完了 | sac_retrain_scheduler.py L296 |
| 8 | retrain_scheduler PYTHONPATH 修正 | ✅ 完了 | retrain_scheduler.ps1 |
| 9 | P0 訓練例外→信号stale防止 | ✅ 完了 | sac_retrain_scheduler.py (neutral fallback) |
| 10 | ゾンビ検出 venv限定フィルタ | ✅ 完了 | retrain_scheduler.ps1 |
| 11 | hot_swap config 不整合修正 | ✅ 完了 | hot_swap_restart.ps1 |
| 12 | hot_swap graceful shutdown | ✅ 完了 | hot_swap_restart.ps1 |
| 13 | 防御レイヤー実運用定量分析 | ✅ 完了 | ログベース fill rate / blocking 分析 |

---

## §2 sidecar retrain_scheduler 復旧

### 問題

- 2026-03-11 以降 `cache/sidecar_signal.json` が 8 日間未更新 (stale)
- 6 個のゾンビ retrain_scheduler プロセスが `retrain_scheduler.py` (旧スクリプト) + `fill_test.yaml` (誤 config) で起動されていた
- DLL エラー: システム Python (3.11) の torch c10.dll ロード失敗

### 原因分析

1. **PYTHONPATH 欠落**: `Start-Process` で起動すると `sys.path[0]` がスクリプトディレクトリになり、プロジェクトルートの `ztb` パッケージが見つからない
2. **Timestamp 型不整合**: `_build_training_debug_details()` で `float(pd.Timestamp)` が TypeError

### 修正

- `retrain_scheduler.ps1`: `$env:PYTHONPATH = $ProjectRoot` を `Start-Process` の前に追加
- `sac_retrain_scheduler.py` L296-303: `float(timestamp)` → `ts.timestamp() if hasattr(ts, "timestamp") else float(ts)`

### 結果

- PID 85352 で正常起動、warm-start で `models/v460/sac_sidecar.zip` をロード
- 15,000 step 増分訓練が進行中
- signal file は訓練完了後に更新される

---

## §3 P0-1: VPIN Boost 勾配有効化

### 問題の定量分析

```
VPIN 統計 (直近200サイクル):
  平均: 0.69    最小: 0.42    最大: 0.84

旧設定: vpin_threshold=0.60, vpin_continuous_min=0.40
  → ランプ範囲: 0.40-0.60 (0.20幅)
  → VPIN 0.60+ で _norm=1.0 (飽和) → boost=2.0x 常時適用
  → 全サンプルが cont=1.00 を記録

新設定: vpin_threshold=0.80, vpin_continuous_min=0.40
  → ランプ範囲: 0.40-0.80 (0.40幅)
  → VPIN 分布全体をカバーする二次曲線
```

### boost 計算の変化

| VPIN | 旧 _norm | 旧 boost | 新 _norm | 新 boost | 勾配改善 |
|------|----------|----------|----------|----------|---------|
| 0.50 | 0.50 | 1.25x | 0.25 | 1.06x | ↓19% |
| 0.60 | 1.00 | 2.00x | 0.50 | 1.25x | ↓75% |
| 0.65 | 1.00 | 2.00x | 0.63 | 1.39x | ↓61% |
| 0.70 | 1.00 | 2.00x | 0.75 | 1.56x | ↓44% |
| 0.80 | 1.00 | 2.00x | 1.00 | 2.00x | 同等 |

### 即時効果 (ログ確認)

```
変更前: vpin=0.70(cont=1.00), vpin=0.80(cont=1.00)  ← 100%飽和
変更後: vpin=0.50(cont=0.25), vpin=0.72(cont=0.79)  ← 段階的勾配
```

### sell_dynamic_kill への期待効果

- 変更前: 12.5% ブロック率 (272 sell サイクル中 34 ブロック)
- offset 膨張が抑制されるため quote freshness 改善 → rolling PnL 改善 → kill 閾値到達頻度低下
- 24-48h 後の再測定が必要

---

## §4 防御レイヤー多重共線性分析 (490# Deep Dive)

### 定量結果

```
総サイクル要求: 4,120
実際の fill: 997
fill_rate: 24.2%
失敗: 3,123 (75.8%)

主要ブロッカー:
  sell_dynamic_kill:  979件 (最大)
  one_sided_*:        500+件
  ranging_low_vol:    ~400件
  unknown_regime:     ~300件
  spread_too_narrow:  ~250件
```

### 5つの多重共線性クラスタ

| クラスタ | 潜在因子 | 関連ゲート数 | 代表ゲート |
|---------|---------|-------------|----------|
| A: レジーム不確実性 | regime=unknown | 3-4 | unknown_regime_buy/sell, rule_skip_unknown |
| B: 流動性枯渇 | spread_bps | 5-6 | narrow_spread, SAD, no_feasible_quote, postonly |
| C: 市場ストレス | velocity_bps | 4-5 | ranging_low_vol, MCB, velocity_skip |
| D: 在庫偏重 | inv_imbalance | 3 | one_sided_freeze/cooldown, degraded_liquidation |
| E: 逆選択リスク | rolling_pnl | 3-4 | buy/sell_dynamic_kill, toxicity, cross_venue |

### 既存の部分統合

| 統合手法 | 対象 | 状態 | 範囲 |
|---------|------|------|------|
| Toxicity Budget (240#) | Gate 4/5 | 段階応答 | cycle_gate 内のみ |
| Alert Aggregation (211#) | MCB/SAD | 乗算型 | pre-cycle 内のみ |
| EV-Weighted (188#) | Skip Gate | 加算型 | skip_gate 内のみ |
| Soft Mode (195#/196#) | 各ゲート | 個別 offset_mult | 分散・統一なし |

### 改善方向性 (490# 推奨)

| Level | アプローチ | 期間 | 効果 |
|-------|----------|------|------|
| 1. 対症療法 | 閾値個別調整 | 即時 | 低 (多重共線性残存) |
| **2. 構造改善** | **Composite Risk Score + 単一閾値** | **2-4週** | **中** |
| 3. アーキテクチャ | SAC に gate 移行 | 1-3月 | 高 |

---

## §5 再学習装置 堅牢性強化 (Turn 3)

### 5.1 P0: 訓練例外時の信号stale防止

**問題**: `retrain_once()` が `ImportError` や汎用 `Exception` で失敗すると、
`_push_neutral_fallback()` が呼ばれず `sidecar_signal.json` が stale 状態のまま放置される。

**修正**: `sac_retrain_scheduler.py` — 両方の例外ハンドラに `_push_neutral_fallback()` 呼出しを追加。

```python
except ImportError as e:
    _push_neutral_fallback()  # ← 追加
    ...
except Exception as e:
    _push_neutral_fallback()  # ← 追加
    ...
```

### 5.2 ゾンビプロセス検出の精緻化

**問題**: `retrain_scheduler.ps1` の `Get-RetrainProcess` がシステム Python も検出し、
venv 外のプロセスをゾンビ誤判定。実際に PID 78540 (system Python) が検出されていた。

**修正**: `retrain_scheduler.ps1` — venv パス限定フィルタ。

```powershell
Where-Object {
    $_.CommandLine -like "*retrain_scheduler*" -and
    $_.CommandLine -like "*$($ProjectRoot.Replace('\\', '\\\\'))\\.venv*"
}
```

### 5.3 hot_swap_restart.ps1 config 不整合

**問題**: retrain_scheduler 再起動時に `$Config` (= fill_test.yaml) を渡していた。
本来は `g2_sac_train.yaml` が必要。

**修正**: ハードコード `configs/v460/experiments/g2_sac_train.yaml` に変更。

### 5.4 hot_swap graceful shutdown

**問題**: retrain_scheduler を `-Force` で即座に kill しており、
訓練中のモデル書き込みやチェックポイント保存が中断される可能性。

**修正**: 3段階停止シーケンス。

```
1. Stop-Process (通常) → _shutdown_event をトリガー
2. 15秒間隔で 3秒ごとにプロセス生存チェック
3. 残存時 → Stop-Process -Force
```

---

## §6 防御レイヤー実運用定量分析 (Turn 3)

### 6.1 直近ログ (613サイクル) の blocking 内訳

| カテゴリ | 件数 | 割合 | 性質 |
|---------|------|------|------|
| 残高不足 (JPY) | 409 | 最大 | 資金制約 (非バグ) |
| 残高不足 (BTC) | 266 | | 資金制約 (非バグ) |
| no_feasible_quote | 48 | 7.8% | cross_venue_lead_lag_veto burst |
| sell_dynamic_kill | 34 | 5.5% | VPIN 起因 offset 膨張 |
| unknown_regime | 19 | 3.1% | レジーム判定不確実 |
| buy_dynamic_kill | 19 | 3.1% | gate level blocker |
| ranging_low_vol | 14 | 2.3% | 低 vol 時の保護 |
| SAD/MCB | 0 | 0% | 正常動作 (非トリガー) |

### 6.2 Fill Rate

```
総サイクル:   613
発注済:       345 (56.5%)
約定:         222 (36.3% of total, 64.3% of placed)
```

### 6.3 重要な発見

1. **残高不足が最大のブロッカー** — JPY 255円 / BTC 0.000000 で繰り返しスキップ。
   資金追加無しではこれ以上の fill rate 改善は限定的。
2. **no_feasible_quote (48件)** — 100% が `cross_venue_lead_lag_veto` 起因。
   7回連続 burst パターンで NO_FEASIBLE_QUOTE (閾値3回) をトリガー。設計通りだが burst 頻度は注視対象。
3. **SAD/MCB は 0 トリガー** — スプレッド異常検知は正常域。過剰ブロックなし。
4. **sell_dynamic_kill (34件)** — VPIN threshold 修正 (0.60→0.80) により改善見込み。24-48h 後に再測定。

## §5 P1 課題ステータス

| ID | 項目 | 判定 | 理由 |
|----|------|------|------|
| P1-1 | cycle exception 分類 | ✅ 修正済 (18a7d66a8) | ConnectionError/TimeoutError → WARNING |
| P1-2 | サイクル結果ログ enrichment | 🟡 保留 | 技術的負債、優先度低 |
| P1-3 | プログレスログ欠落 | 🟡 保留 | 技術的負債、優先度低 |
| P1-4 | Reanchor budget 25bps | ✅ 設計確認 | 設計通り動作、チューニング課題 |
| P1-5 | Cooldown 段階的回復 | ✅ 設計確認 | rearm 機序 (249#) で対応済み |
| P1-6 | Config 相互参照 | ✅ 修正済 (本セッション) | VPIN min < threshold チェック追加 |
| P1-7 | Offset clamping 3重 | ✅ 非バグ確認 | 421# で resolve_offset_ceiling 統一済み |

---

## §6 コミット一覧

### 前セッション (489#-490# 対応)

| SHA | 内容 |
|-----|------|
| 5608149c6 | sidecar stat/read race + cross-venue 例外分類 |
| cc6c9466f | re-quote place_order kwargs 修正 |
| eddca6dc9 | hot_swap_restart.ps1 taskkill ErrorActionPreference |
| 17eb54d05 | 488# ドキュメント 489#/490# 補正反映 |
| 18a7d66a8 | P0-3 sigma_floor + P1-1 cycle exception + P0-2 非バグ確認 |

### 本セッション (491#)

| 対象ファイル | 変更内容 |
|-------------|---------|
| `configs/v460/fill_test.yaml` | vpin_threshold: 0.60→0.80 |
| `scripts/v460/ml/sac_retrain_scheduler.py` | Timestamp→float 変換修正 |
| `scripts/v460/lib/fill_config_validation.py` | VPIN min < threshold バリデーション追加 |
| `tests/unit/v460/test_346_fill_config_validation.py` | VPIN バリデーションテスト 3 件追加 |
| `ops/windows/retrain_scheduler.ps1` | PYTHONPATH 設定追加 |

---

## §7 次回アクション

### 短期 (次セッション)

1. **sidecar signal 更新確認**: retrain_scheduler 訓練完了後の signal freshness を検証
2. **VPIN threshold 効果測定**: 24-48h 後の sell_dynamic_kill ブロック率変化
3. **fill_rate 回復度測定**: 24.2% → 目標 35%+ への改善幅

### 中期 (2-4週)

4. **Composite Risk Score 設計** (490# Level 2): 5 クラスタの潜在因子を正規化して単一スコアに集約
5. **Gate 統合 PoC**: クラスタ B (spread 5-6 重) の 統一 → 単一 liquidity_score

### 長期 (1-3月)

6. **SAC gate 移行**: 観測化 + 学習最適化 (490# Level 3)
7. **マルチシード ensemble**: seed42/123 の prediction averaging
