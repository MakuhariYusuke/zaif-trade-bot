# 088# 087# レビュー対応: 構造的改善の即効策実装

> **状態**: 実装完了 — テスト 692/692 pass (新規 25 含む)  
> **前提**: 087# Codex レビュー (086# に対する外部分析)  
> **コミット**: 本文書と同時  

---

## §1 087# レビュー妥当性評価

087# は 086# の time_filter 片側蓄積バグ修正「後」のデータで、
**損失の根本原因がバグではなく構造的問題にある** と結論づけた。

| 指摘 | 評定 | 根拠 |
|---|---|---|
| P(AS) 52.3% ≈ ランダム → SkipGate 実質無効 | ✅ 妥当 | 閾値 0.65 に対し分布の大半が < 0.55 — skip 率 0% は必然 |
| sell -1.4bps vs buy -0.45bps の非対称 | ✅ 妥当 | 085# で spread 4677 時の -7.37bps 事例あり — 広スプレッド sell は構造的に不利 |
| status_unknown が損失直結 | ✅ 妥当 | 1 回リトライ (2s) では API 一時障害に対応不十分 |
| データ品質: quarantine 混入 | ✅ 妥当 | **更に深刻な盲点を発見** — 後述 §3 |
| param_adapter が buy/sell 混合 | ✅ 妥当 | sell 劣後が buy 正常値で埋没し、適応速度が遅い |

### 087# が見落としていた盲点

**FillRecord 早期 return パスの run_id/git_sha 欠落 (P0-4):**

3 つの早期 return (orderbook_error, skip_gate SKIP, api_error) で生成される
FillRecord に `run_id` と `git_sha` が設定されておらず、
fill_quality.py の `_quarantine_reason()` が「run_id 空」で quarantine 判定、
正常データが分析から除外されていた。

087# §2.2 は「quarantine 指定が曖昧」と指摘したが、
**根本原因はフィールド未設定** であった。

---

## §2 実装一覧

### P0-1: SkipGate 動的閾値較正

**問題**: P(AS) ≈ 0.51 に対し閾値 0.65 → skip 率 0% (SkipGate 無効化)  
**解法**: 目標 skip 率ベースの適応較正

| パラメータ | 値 | 説明 |
|---|---|---|
| `adaptive_threshold` | `true` | 動的較正有効化 |
| `target_skip_rate_buy` | `0.10` | buy 側 10% skip |
| `target_skip_rate_sell` | `0.20` | sell 側 20% skip (sell 劣後補正) |
| `adaptive_window` | `50` | 直近 50 件の P(AS) 分布 |
| `adaptive_min_samples` | `20` | ウォームアップ (20 件まで静的閾値) |
| `adaptive_step` | `0.02` | 1 回の最大調整幅 |
| `adaptive_floor` | `0.35` | 下限 (過剰 skip 防止) |
| `adaptive_ceiling` | `0.80` | 上限 (skip 無効化防止) |

**アルゴリズム**:
1. 各 cycle で P(AS) を side 別履歴に記録
2. ウォームアップ完了後、`(1 - target_rate)` 分位点を算出
3. base_threshold を step 幅で分位点に近づける
4. floor/ceiling でクランプ
5. `config.as_threshold_buy/sell` を更新 → 次回 evaluate で反映

**変更ファイル**: `skip_gate.py` (SkipGateConfig + _calibrate_threshold), `run_fill_test.py` (FillTestConfig + YAML parse + wiring), `fill_test.yaml`

### P0-2: Sell 専用ハードガード

**問題**: sell 側 -1.4bps (buy -0.45bps の 3 倍) — 広スプレッド時の sell 注文が主因  
**解法**: 2 層ガード

| ガード | 値 | 効果 |
|---|---|---|
| `sell_offset_floor` | `0.08` | sell offset の最低保証 (板から十分に離す) |
| `sell_max_spread_jpy` | `4000` | スプレッド 4000 円超で sell 即スキップ |

**変更ファイル**: `run_fill_test.py` (_compute_maker_price), `fill_test.yaml`

### P0-3: Status Unknown リトライ強化

**問題**: API 一時障害で 1 回 (2s) リトライ → status_unknown 確定 → 損失計上  
**解法**: 3 段階リトライ (2s, 3s, 5s = 合計 10s)

- 各リトライで状態回復チェック (filled/open/cancelled/rejected)
- filled 以外の回復状態にも適切に対処
- 回復時のログ出力強化

**変更ファイル**: `run_fill_test.py` (polling ループ内)

### P0-4: データ品質 — FillRecord 早期 return 修正 ⚠️ 盲点

**問題**: 3 つの早期 return パスで `run_id=` と `git_sha=` が未設定  
**影響**: quarantine 判定により clean データから除外 → 分析精度低下  
**解法**: 全 3 箇所に `run_id=self._run_id`, `git_sha=self._git_sha` を追加

対象箇所:
1. orderbook_error (L1054): `_compute_maker_price` 失敗時
2. skip_gate SKIP (L1161): SkipGate 判定によるスキップ時
3. api_error (L1267): 全 order attempt 失敗時

**検証**: ソースコード検査テスト (`test_088_features.py::TestDataQualityFillRecord`) で
全 `cancel_reason` 付き FillRecord に `run_id=` と `git_sha=` の存在を自動検証。

### P1-3: param_adapter Side 分離適応

**問題**: buy/sell 混合メトリクスで適応 → sell 劣後が buy 正常値で希釈  
**解法**: `compute_side_adaptation()` で buy/sell 独立に offset 調整

- `SideAdaptationResult` データクラス (`any_changed` プロパティ付き)
- `_try_auto_adapt()` を side 分離版にリファクタ
- 片方のサンプルが不足 (< 20) の場合は combined 適応にフォールバック

**変更ファイル**: `param_adapter.py`, `run_fill_test.py`

---

## §3 テスト

### 新規テスト: 25 件 (test_088_features.py)

| クラス | テスト数 | 対象 |
|---|---|---|
| `TestSkipGateAdaptiveThreshold` | 10 | 動的較正: ウォームアップ、閾値調整方向、floor/ceiling、evaluate 統合 |
| `TestSkipGateConfig` | 2 | 088# 新フィールドのデフォルト値・カスタム値 |
| `TestComputeSideAdaptation` | 8 | side 分離適応: 独立判定、デッドロック防止、サンプル不足 |
| `TestCalibrateThresholdEdgeCases` | 4 | 境界値: 全同値、skip 率 0/1.0、step 制限 |
| `TestDataQualityFillRecord` | 1 | ソースコード検査: cancel_reason 付き FillRecord に run_id/git_sha |

### 全体テスト: 692/692 pass

---

## §4 087# 残課題 (P1/P2 — 本セッション範囲外)

| 優先度 | 項目 | 状態 | 理由 |
|---|---|---|---|
| P1-1 | Round-trip を primary KPI に | ⏳ 未着手 | 設計変更が大きい — 次セッション |
| P1-2 | time_filter event-driven 化 | ⏳ 未着手 | SkipGate との役割重複整理が先 |
| P2-1 | Event-driven サイクル間隔 | ⏳ 未着手 | 根本設計変更 |
| P2-2 | Two-stage model | ⏳ 未着手 | モデル再訓練が必要 |

---

## §5 次のアクション

1. **fill_test 再起動**: 現行プロセス (PID 74612) は旧コードで稼働中 — 088# 適用のため再起動
2. **適応較正モニタリング**: `[skip_gate] 088# adaptive threshold` ログで較正動作を確認
3. **sell_guard モニタリング**: `[sell_guard] Spread ... > max ...` ログで売スキップ発生率を確認
4. **24h 後レビュー**: 適応較正が目標 skip 率に収束しているか評価
