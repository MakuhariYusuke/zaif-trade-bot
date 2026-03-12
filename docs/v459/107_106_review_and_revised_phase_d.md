# 107# 106# レビュー評価 + Phase D 改訂版

**Date**: 2026-02-10  
**対象**: 106# Phase D 再設計案, 105# Phase D 計画, 104# Phase C 総合レポート  
**方法**: 106# 全指摘のコードレベル照合、追加課題の独自発見、計画への統合  
**大義**: 短期間での高収益性 — 計器の正確さから始め、50K で因果を切り尽くす

---

## §0 結論（先に要点）

1. 106# の 4 指摘中 3 指摘は **妥当**、1 指摘は **部分的に妥当だが資産の実態を過大評価**
2. 106# が見落とした **重大問題が 4 件** ある（§2 に後述）
3. 特に **WinRate=49.2% の定義問題** は Gate 2 判定の根幹に関わる
4. 改訂 Phase D は 106# の「50K 固定・因果分離優先」を採用しつつ、計測バグ修正を D0 最優先に引き上げ

---

## §1 106# 指摘の逐条評価

### §1.1 「D0=学習量拡大は順序が逆」 — **✅ 妥当**

**根拠**:
- 98# §0: 「先にやるべきは最適化ではなく測定基盤の是正」
- 99# §1: 98# の「計測先行」方針を「完全に妥当」と判定
- 101# §4: C0(計測統一) → C1(コスト圧縮) → C2(約定現実化) → C3(報酬/ハイパラ) → C4(OOS) の順
- Phase C 実績: 50K 内で ROI -15% → -4.38% の改善を達成（まだ余地あり）

**追加考察**: 105# は Gate 判定 (2x 改善) を D0 に設定したが、200K で設計ミスを持ち込む危険は 98#/99# の教訓そのもの。106# の修正は正しい。

**採否**: ✅ **採用** — 50K 固定を Phase D の基本制約とする

### §1.2 「特徴量戦略が弱い」 — **⚠️ 部分的に妥当（資産実態を過大評価）**

**106# の主張**: 既存資産 (feature_sets.yaml, FeatureSetManager, run_ab_feature_test.py) をそのまま使え

**コード照合で判明した実態**:

| 資産 | 106# の想定 | 実際 | Gap |
|------|------------|------|-----|
| `feature_sets.yaml` | 3セット (minimal/curated/full) | **`full` = `curated` のエイリアス** → 実質 2 セット | curated 内に重複エントリあり |
| `v459_expanded_features.parquet` | 候補として列挙 | **24,374 行しかない** (1.2M 行の 2%) → 学習に使えない | 全期間再生成が必要 |
| `run_ab_feature_test.py` | ロジック流用可 | **UnifiedTrainer パイプライン** (Phase C の run_phase_c.py と全く別経路) | DD fix, eval_dd_threshold, VecNormalize 対応なし |
| `ablate_features.py` | アブレーション用 | **PPO 使用** → SAC 非互換 | actor/critic 構造が異なる |
| `FeatureSetManager` | セット切替に使用 | 機能はあるが **run_phase_c.py に配線なし** | 接続実装が必要 |

**結論**: 方針としては正しいが、106# が示唆する「新規実装なし」は楽観的すぎる。最低限以下が必要:
1. run_phase_c.py に特徴量プロファイル切替の接続
2. curated セット (78 特徴) の特徴量を全期間データで Parquet 再生成
3. curated の重複エントリ修正

**採否**: ✅ **方針は採用**（50K で既存セット比較）。ただし **実装コスト見積もりを修正**

### §1.3 「C2（約定/コスト現実化）を後回しにしすぎ」 — **⚠️ 部分的に妥当**

**妥当な部分**:
- transaction_cost 感度分析（0.0005 / 0.001 / 0.0015 の比較）は実装コストゼロで実施可能
- build_config の `transaction_cost` 値を変えるだけ
- 手数料支配が主因である以上、コスト感度の定量化は早期に行うべき

**妥当でない部分**:
- `ztb/trading/execution/realistic.py` 等の約定モデル導入は現段階では過剰
- 予測精度 (WinRate ~49%) が改善しない限り、約定モデルの精緻化は空回り
- 101# C2 の「約定モデル現実化」は PF>1.0 以降のフェーズで適切

**追加情報**: Zaif 実手数料は maker=0%, taker=0.1%。現在の 0.001 (0.1% 片道) は taker 想定で妥当。往復で 0.2%。

**採否**: ✅ **コスト感度分析のみ採用** (D0 に組込み)。約定モデル本格導入は後回し維持

### §1.4 「D0 の Gate 基準（2x 改善）は恣意的」 — **✅ 妥当**

105# の `avg_gross_pnl_per_trade >= 2x` は以下の問題がある:
- 単一指標への依存
- 分散・再現性の無視（seed=42 単独では分散推定不可）
- **そもそも avg_gross_per_trade はコードに未実装**（→ §2.3 で後述）

**採否**: ✅ **複合 Gate に変更** — PF, Net ROI, per-trade edge, seed 再現性を組合せ

---

## §2 106# が見落とした重大問題

### §2.1 **[CRITICAL] WinRate=49.2% はステップベースであり、取引ベースではない**

**発見経緯**: `win_rate()` の実装をコードレベルで追跡

**実装**:
```python
# ztb/metrics/metrics.py L468-L477
def win_rate(returns) -> float:
    returns = np.asarray(returns)
    return float(np.mean(returns > 0))
```

```python
# scripts/v459/run_phase_c.py (compute_gate2_metrics_from_balances)
returns = np.diff(balances) / np.maximum(balances[:-1], 1e-10)
gate2["win_rate"] = float(win_rate(returns))
```

`balances` は**ステップごとの portfolio_value** 配列 (最大 50K 要素)。
→ WinRate=49.2% は **「50K ステップ中 49.2% のステップで残高が前ステップより増加」** を意味する。
→ **309 取引の勝率ではない**。

**影響**:
- 104# の「WinRate ✅ PASS (49.2% > 35%)」は **定義の解釈次第** で判定が変わりうる
- ステップベースだと「取引していないステップ」(HOLD) でも残高変動があれば計上される → ポジション保有中の含み変動を反映
- **0# (Gate 2) の WinRate 定義が「取引ベース」の含意なら、現在の計測は不適切**

**対応**: D0 で取引ベース WinRate を追加計測し、ステップベースと並置する

### §2.2 **[HIGH] eval_dd_threshold=1.0（DD 停止完全無効化）の非現実性**

C3 の大幅改善 (ROI -15% → -4.38%) は eval_dd_threshold=1.0 の恩恵が大きい:
- C2 (DD 停止あり): step 31K で emergency_stop → 残り 19K steps は取引不能
- C3 (DD 停止なし): 全 50K steps で取引可能 → 真の性能評価

**しかし本番環境では**:
- `ztb/trading/live_trader/components/risk_manager.py`: `emergency_stop_loss = 0.05` (5%)
- Live とeval で DD 閾値が 20 倍異なる (5% vs 100%)

**106# はこの乖離に触れていない**。

**対応**: D0 で eval_dd_threshold=0.30 (中間値) での追試を追加。Gate 判定は `nodd` と `dd30` の両方で行う。

### §2.3 **[MEDIUM] avg_gross_per_trade / avg_fee_per_trade が未実装**

105# D0, 106# D0 の両方で「KPI に含める」と書かれているが、
`scripts/v459/run_phase_c.py` に per-trade 平均メトリクスは存在しない。

**対応**: D0 基盤整備で 2 行追加（`eval_gross_pnl / eval_trades`, `eval_total_fees / eval_trades`）

### §2.4 **[MEDIUM] 二項検定・ランダム超過検証がパイプラインに存在しない**

WinRate=49.2% (309 取引ベースと仮定) が Random (50%) と有意に異なるか:
- 二項検定: $p \approx 0.776$ (両側) → **ランダムと区別不可能**
- 104# / 105# / 106# のいずれも統計的有意性を検証していない

**106# は D4 で Mann-Whitney を言及** しているが、D0-D3 の段階でも per-experiment の Random 超過チェックが必要。

**対応**: D0 で簡易二項検定を eval パイプラインに追加

---

## §3 106# が列挙した資産の実用性再評価

| 資産 | 106# の評価 | 実態評価 | Phase D での扱い |
|------|------------|---------|-----------------|
| `feature_sets.yaml` (minimal/curated) | そのまま比較 | ✅ 使える (ただし curated 重複修正要) | D1 で minimal(20) vs curated(~76) vs 現行(8) を比較 |
| `FeatureSetManager` | セット切替 | ✅ 機能あり (run_phase_c.py への配線要) | D0 基盤で接続 |
| `run_ab_feature_test.py` | ロジック流用 | ⚠️ UnifiedTrainer 経路で Phase C 非互換 | 参考のみ。run_phase_c.py に直接実装 |
| `ablate_features.py` | アブレーション | ❌ PPO 用。SAC 非互換 | 不採用。run_phase_c.py 内で簡易実装 |
| `v459_expanded_features.parquet` | 候補 | ❌ 24K 行で使えない | 不採用。curated/minimal を全期間再生成 |
| `run_phase_c.py` | C3 ベースで転用 | ✅ 最も互換性が高い | D0 の主要改修対象 |
| `run_baselines.py` | Random 超過確認 | ✅ そのまま使える | D4 で活用 |
| `precompute_optimized_features_memory_safe.py` | Parquet 再生成 | ✅ そのまま使える | D1 前に特徴量セット別 Parquet 生成 |

---

## §4 Phase D 改訂版ロードマップ

### §4.0 設計原則（106# + 追加修正を統合）

1. **50K 固定** (106# §1.1): timesteps 拡大は全 Gate 通過後の最終確認のみ
2. **計測バグ修正 FIRST** (§2 の 4 件): WinRate 定義、per-trade メトリクス、二項検定
3. **既存資産活用** (106# §2): ただし実装コストを正しく見積もる
4. **eval_dd_threshold の多段評価** (§2.2): nodd + dd30 の並行評価
5. **因果分離** (98#/99#/101#): 一度に変える変数は 1 つ

### §4.1 D0: 計測基盤修正 + 実験フレーム確立（1 日）

**D0-a: 計測バグ修正** (最優先)
1. `run_phase_c.py` に **取引ベース WinRate** を追加
   - `win_trades / total_trades` (取引ごとの realized PnL > 0 の割合)
   - ステップベース WinRate と並記（`step_win_rate` / `trade_win_rate`）
2. **per-trade 平均メトリクス** を追加
   - `avg_gross_per_trade = eval_gross_pnl / eval_trades`
   - `avg_fee_per_trade = eval_total_fees / eval_trades`
3. **簡易二項検定** を eval 出力に追加
   - `binom_test_p_value`: WinRate が 0.5 と有意に異なるか

**D0-b: 実験フレーム拡張**
1. `run_phase_c.py` に **特徴量プロファイル切替** を接続
   - `DATA_PATH` を experiment definition から指定可能に
   - `feature_set` パラメータ → 対応 Parquet を選択
2. `eval_dd_threshold` を **C3 固定 (1.0) と中間 (0.30) の並行評価** に変更
3. KPI ログフォーマットを統一（106# D0 と同じ）

**Gate**: D0-a/b 完了 + テスト PASS

### §4.2 D1: 既存特徴量セット比較（50K、1.5 日）

**前提作業**: D0-b で特徴量切替が可能になった状態で実施

**Step 1: Parquet 生成** (curated/minimal)
- `precompute_optimized_features_memory_safe.py` を改修:
  - `--feature-set curated` → feature_sets.yaml の curated リストで計算
  - `--feature-set minimal` → minimal リストで計算
- 出力: `data/btc_jpy_1m_curated_features.parquet`, `data/btc_jpy_1m_minimal_features.parquet`

**Step 2: seed=42 粗選別** (3 実験)

| 実験 | 特徴量セット | 特徴数 | eval_dd | 固定パラメータ |
|------|------------|---------|---------|--------------|
| d1_v451opt | v451_optimized (現行) | 8 | 1.0 + 0.30 | ent001, thr70 |
| d1_minimal | minimal | 20 (OHLCV 含む) | 1.0 + 0.30 | ent001, thr70 |
| d1_curated | curated | ~76 (OHLCV 含む) | 1.0 + 0.30 | ent001, thr70 |

**Step 3: 上位 2 条件を 2-seed で再現確認**

**Gate**: trade_win_rate, PF, avg_gross_per_trade の **相対順位** で上位を選定
- 改善あり → D2 へ
- 全セット同等 → 現行 8 特徴で D2 へ（特徴量は非ボトルネックと判定）

### §4.3 D2: コスト感度分析 + 報酬微調整（50K、1 日）

**D2-a: コスト感度** (D1 best 条件ベース)

| 実験 | transaction_cost | 目的 |
|------|-----------------|------|
| d2_cost05 | 0.0005 (maker 想定) | 手数料半額での PF 変化 |
| d2_cost10 | 0.001 (taker 現状) | baseline |
| d2_cost15 | 0.0015 (悪条件) | 頑健性確認 |

**D2-b: 報酬微調整** (D1 best + D2-a best コスト)

| 実験 | 変更 | 根拠 |
|------|------|------|
| d2_asymm12 | loss × 1.2 | 91# v451 非対称報酬 |
| d2_cost_train | 学習時 cost=0.005, eval 時 cost=0.001 | コスト感度を明示学習 |

**Gate**: PF >= 1.05 かつ Net ROI >= -2% → D3 へ

### §4.4 D3: Multi-seed 再現性検証（50K、1 日）

D2 best 条件で 4 seeds [42, 123, 456, 789]:
1. **二項検定**: trade_win_rate vs 0.5 (各 seed)
2. **Mann-Whitney + Cliff's delta**: SAC vs Random の PF/ROI 比較 (gate_c3_comparison.py)
3. **分散評価**: 4 seed の PF/ROI の分散・最悪ケース確認

**Gate**: 4 seed 中 3 以上で PF>1.0 + trade_win_rate > 50% + Random 有意超過 (p<0.10)

### §4.5 D4: OOS + Step 拡大許可（条件付）

**D3 Gate 通過後のみ実施**:

1. **OOS**: walk-forward 4-split (ztb/evaluation/walk_forward/splitter.py)
2. **Step 拡大**: D3 best 条件のみ 100K → 200K で改善余地を確認
3. **最終 Gate 2 判定**: 全 KPI で GO/NO-GO

**Step 拡大を許可する条件** (106# §4 を踏襲):
- 50K で PF >= 1.05 かつ Net ROI >= 0% を複数 seed で再現
- avg_gross_per_trade が avg_fee_per_trade に実質接近
- Random 比較で有意超過

---

## §5 106# が列挙したが対応不要な項目

| 106# 記載 | 判断 | 理由 |
|-----------|------|------|
| `v459_expanded_features.parquet` 候補化 | ❌ 不採用 | 24K 行でデータ不足、特徴量は curated に含まれる |
| `ablate_features.py` ロジック流用 | ❌ 不採用 | PPO 用で SAC 非互換。run_phase_c.py 内で簡易 LOO が安価 |
| `run_ab_feature_test.py` ロジック流用 | ❌ 参考のみ | UnifiedTrainer 系で Phase C と別経路 |
| D2: 約定モデル本格導入 | ❌ 後回し | 予測精度改善が先。PF>1.0 後に着手 |

---

## §6 リスクと制約（改訂版）

### §6.1 計算資源

| Phase | 実験数 | 見込み時間 | 備考 |
|-------|--------|-----------|------|
| D0 | 0 (改修のみ) | 半日 | テスト追加含む |
| D1 | 3-7 experiments | 1-1.5 日 | Parquet 再生成 ~30 分含む |
| D2 | 5 experiments | 1 日 | |
| D3 | 4-8 experiments | 0.5-1 日 | 4 seeds 並行可 |
| D4 | 16+ experiments | 1-2 日 | 4 seeds × 4 splits |
| **合計** | | **4-6 日** | 50K 固定により 105# の半分以下 |

### §6.2 最大リスク

| リスク | 深刻度 | 緩和策 |
|--------|--------|--------|
| WinRate がステップベースの場合、Gate 2 PASS 取消の可能性 | CRITICAL | D0 で取引ベース WinRate 追加 |
| curated 78 特徴で PF が改善しない | HIGH | 現行 8 特徴のまま D2 に進む（特徴は非ボトルネック） |
| 50K で PF>1.05 に到達しない | HIGH | §4.5 の条件付き step 拡大 or 105# §4 代替戦略 |
| eval_dd_threshold=0.30 で C3 の改善が消失 | MEDIUM | nodd と dd30 の並行評価で影響を定量化 |

---

## §7 Phase D0 の具体的実装タスク（即実行可能・独自実装なし）

**設計方針**:
- **独自実装を排除**: 全メトリクス計算は既存パターンの再利用のみ
- **保守性**: 汎用ヘルパー関数を `ztb/metrics/` に追加し、Phase E 以降も再利用可能に
- **HeavyTradingEnv 非改変**: env の `realized_pnl` / `trades_count` の差分から外部追跡（eval ループ側で完結）

### D0-a: 計測基盤修正（既存パターン流用のみ）

1. **`_run_deterministic_eval` で取引ベース PnL 追跡**

   **流用元**: `PositionManager.close_position()` → `trades_count += 1` + `realized_pnl` 更新パターン
   ```python
   # eval ループ内（env の既存属性の差分を追跡するだけ）
   prev_trades = raw_env.trades_count
   prev_realized = raw_env.realized_pnl
   # ... step() ...
   if raw_env.trades_count > prev_trades:
       trade_pnls.append(raw_env.realized_pnl - prev_realized)
   ```
   → `trade_win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)`

2. **per-trade メトリクス追加**

   **流用元**: `run_baselines.py` L113-114
   ```python
   n = result["eval_trades"]
   if n > 0:
       result["avg_gross_per_trade"] = result["eval_gross_pnl"] / n
       result["avg_fee_per_trade"] = result["eval_total_fees"] / n
   ```

3. **簡易二項検定追加**

   **流用元**: `performance_validator.py` L290-294 (`scipy.stats.binomtest`)
   ```python
   from scipy.stats import binomtest  # scipy >= 1.10.0（pyproject.toml 確認済み）
   if n > 0:
       result["binom_p_value"] = binomtest(win_trades, n, 0.5).pvalue
   ```

4. **Gate 2 WinRate の明確化**
   - 既存 `win_rate` → `step_win_rate` にリネーム
   - `trade_win_rate` を Gate 2 結果辞書に追加（両方並記）

### D0-b: 実験フレーム拡張（build_config の設定追加のみ）

5. **`data_path` オーバーライド**
   - experiment definition に `data_path` フィールドを追加
   - `build_config` で `DATA_PATH` デフォルトからの上書き

6. **`eval_dd_threshold` 並行評価**
   - 実験定義に `eval_dd_thresholds: [1.0, 0.30]` のリスト対応
   - `_deterministic_eval_gate2` 内で複数閾値を順次評価

### D0-c: テスト（既存テストパターン踏襲）

7. **テスト追加**（`tests/scripts/test_run_phase_c_d0.py`）
   - `test_trade_pnl_tracking`: trade_pnls 追跡ロジック
   - `test_per_trade_metrics`: avg 計算
   - `test_binom_test_output`: binomtest 呼出し
   - `test_data_path_override`: build_config のパス上書き

---

## §8 105# Phase D との差分サマリ

| 項目 | 105# | 106# | 107# (本稿) |
|------|------|------|------------|
| Step 基本 | 200K-500K 先行 | 50K 固定 | **50K 固定** (106# 同意) |
| D0 | 学習量拡大 | 基盤ロック | **計測バグ修正 + 基盤ロック** |
| WinRate 定義問題 | 未認識 | 未認識 | **CRITICAL として D0 に組込** |
| eval_dd_threshold | nodd 固定 | nodd 固定 | **nodd + dd30 並行** |
| 二項検定 | なし | D4 で Mann-Whitney | **D0 で簡易二項検定追加** |
| per-trade メトリクス | 未実装 | KPI に含む (未実装) | **D0 で実装** |
| 特徴量比較 | D1 で曖昧 | D1 で具体化 | **D1 で具体化 (資産実態修正込み)** |
| コスト感度 | D2 で報酬のみ | D3 で実施 | **D2 で報酬と並行** |
| 再現性 | D3 で 4-seed | D4 で 4-seed | **D3 で 4-seed** |
| Gate 基準 | 2x 改善 (恣意的) | 未具体化 | **複合 Gate (PF+ROI+trade_WR+binom_p)** |
| 所要期間 | ~10 日 | ~4 日 (48 時間案) | **4-6 日** |
| 代替戦略 | §4 で定義 | 触れず | **105# §4 を維持** |

---

## §9 参照ドキュメント

| Doc# | 参照した要点 |
|------|------------|
| 0# | Gate 2 基準 (ROI>5%, PF>1.20, Sharpe>1.0, MaxDD<15%, WinRate>35%) |
| 66# | 統計検定不足、seed 数不足 |
| 91# | H1-H5 仮説マトリクス (H1 否定済み, H2 確認済み) |
| 98# | 「計測基盤の是正が最優先」「最適化の前に測る」 |
| 99# | 98# の方針を「完全に妥当」と追認 |
| 101# | C0→C1→C2→C3→C4 実行順、既存資産再利用マップ |
| 104# | Phase C 全結果、WinRate 49.2%, PF 0.990, fees/gross=4.2x |
| 105# | Phase D 初版 (200K 先行 → 106# で否定) |
| 106# | 50K 固定、既存特徴量活用、コスト早期確認 |
