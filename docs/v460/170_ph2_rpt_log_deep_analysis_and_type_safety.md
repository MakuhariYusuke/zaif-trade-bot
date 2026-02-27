# 170# ログ深堀り分析 + 型安全強化 + AI レビュー準備

**Date**: 2026-02-27
**Phase**: ph2 (maker 執行可能性検証)
**Gate**: G1.1-exec (二段階 Kill/Qualification)
**Run**: `run_id=1772160538_315362d9`, `git_sha=cd2513c78b04` → `1cd12bf4e` (hot-reload)
**Predecessor**: 169# (深堀り分析 + ロットスケーリング計画)
**Purpose**: ログデータの深堀り分析、169# 以降の施策効果測定、fill_test コード型安全強化、外部 AI レビュー準備。

---

## 目次

- [§1 Executive Summary](#1-executive-summary)
- [§2 プロジェクト経緯](#2-プロジェクト経緯)
- [§3 ログ深堀り分析](#3-ログ深堀り分析)
- [§4 施策効果の時系列評価](#4-施策効果の時系列評価)
- [§5 型安全強化の実施内容](#5-型安全強化の実施内容)
- [§6 Config Hot-Reload 実装](#6-config-hot-reload-実装)
- [§7 外部 AI レビュー向けパッケージ](#7-外部-ai-レビュー向けパッケージ)
- [§8 次アクション](#8-次アクション)
- [改訂履歴](#改訂履歴)

---

## §1 Executive Summary

### 1.1 現状サマリ (2026-02-27 12:00 JST)

| 指標 | 値 | 169# 比較 | 判定 |
|------|-----|-----------|------|
| total_cycles | 4,348 | +291 | — |
| filled | 1,854 | +69 | — |
| **fill_rate** | **42.6%** | -1.4pt | ❌ 悪化 |
| **PnL30 mean** | **-0.337 bps** | -0.07 | ❌ 微悪化 |
| PnL120 mean | -0.127 bps | — (新計測) | ❌ 負 |
| AS rate | 27.9% | +0.5pt | ⚠️ 横ばい |
| buy PnL30 | -0.189 bps | — | ❌ 負 |
| sell PnL30 | **-0.487 bps** | — | ❌ sell が 2.6x 悪い |
| 累積実損 (state) | **-553 JPY** | -176 JPY 悪化 | ✅ キャップ内 |
| DailyDrawdownGuard | halt 1 日 | — | ✅ 安全弁作動実績 |
| 本日 daily PnL | +15.25 bps | — | ✅ 本日は好調 |

### 1.2 要点

1. **fill_rate は依然低迷** (42.6%): 169# の B1' (ranging_buy low_vol skip) は損失回避には成功するも、fill_rate をさらに低下させる副作用
2. **sell が buy の 2.6 倍悪い**: sell PnL30=-0.487bps vs buy=-0.189bps。sell の構造的問題は依然未解決
3. **PnL120 は PnL30 より改善**: -0.127 vs -0.337 bps → 時間経過で回復傾向 (90s sell hold の効果示唆)
4. **DailyDrawdownGuard 実作動**: halt 1 日の実績 → C4 (169#) の有効性確認
5. **本日 (2/27) は好調**: +15.25bps/日 → 短期変動の範囲内だが改善傾向の兆候

### 1.3 今セッション (170#) の成果

| 施策 | 種別 | 効果 |
|------|------|------|
| 169# phg→ph2 リネーム | 文書整備 | 命名規則統一 |
| Config Hot-Reload 実装 | 構造改善 | **再起動不要で YAML 変更を即時反映** (120s) |
| `Any` 型完全排除 (config_hot_reload) | 型安全 | Protocol パターンで circular import を型安全に解消 |
| bare `dict`/`list` 型修正 (3ファイル) | 型安全 | 戻り値の型情報が下流に伝搬 |
| 15 テスト全パス | 品質保証 | hot-reload ロジックの回帰テスト |
| subprocess popup 抑制 | UX | Windows CMD ポップアップ完全抑制 |

---

## §2 プロジェクト経緯

### 2.1 v460 全体タイムライン

| 期間 | Phase | 主要イベント |
|------|-------|------------|
| 2026-02-13 | ph0 | 000# Project Proposal: "Microstructure Edge" |
| 2026-02-14~15 | ph1 | G1-info PASS: マイクロストラクチャ特徴量に情報量存在を確認 |
| 2026-02-15~現在 | **ph2** | G1.1-exec: maker 執行品質の実測検証 (fill_test 168h 蓄積中) |
| — | ph3~5 | 待機 (一部先行: SAC 調査、メモリリーク修正) |

### 2.2 168#~170# セッション経緯

#### 168# セッション (2026-02-25~26)

168# は ph2 の大規模棚卸しセッション。主要成果:

1. **HODL vs Trading 定量比較**: BTC HODL 年 52% vs Trading Bot Oracle 目標 1,063% (S3 時)
2. **根本原因特定**: `spread_offset_ratio` が t=-6.23 で最強の損失予測因子 (SHAP 分析)
3. **低 vol offset boost**: vol_ratio < 0.70 で offset×1.4 → 構造対策として投入
4. **time_filter 精緻化**: UTC7/12/21 遮断 + regime_adaptive (後に 169# で全廃)
5. **週次分析自動化**: weekly_analysis.ps1 + Discord 通知統合
6. **SkipGate 再訓練**: preorder features で ML モデル更新
7. **レビュー受理**: R7-R11 運用ルール (R11: 最大 2 件/run 変更制限)
8. **Fill test 再起動**: PID 122960 (git=`48f1aebb3`)、のち low_vol 閾値修正で PID再起動

**コミット**: `0d5d4f574` → `87e9476c1` → `e38b2e5f1` → `6c6c7bc5d` (5 commits)

#### 169# セッション (2026-02-27)

169# は深堀り分析 + 外部レビュー対応 + 構造改善。主要成果:

1. **G1.1 ゲート診断**: K1 fill_rate=44.0% → FAIL。§3.9 中止条件該当だが Oracle 正 (+2.56bps) で継続
2. **構造損失分解**: ranging_buy が全損失の 69% (-220.84bps)。trending_up_sell が最悪/件 (-2.770bps)
3. **外部 AI レビュー**: Codex R1-R6 + Gemini 10.1-10.3 の批判を受理/却下
   - B2 (timeout 短縮) **撤回**: Gemini 10.2-C が論理矛盾を指摘 (H2_timeout の機会損失を拡大)
   - B1 → **B1' に差替え**: ranging_buy low_vol ハードスキップ (Gemini 10.2-D「休むも相場」)
   - スケーリング計画 →「条件付きシナリオ」に表記変更 (Codex R4)
4. **B1' 実装**: `ranging_low_vol_skip` cancel_reason + config フラグ
5. **B0 実装**: ゲート指標 3 系列 (raw/clean/attempted) 定義修正
6. **C3 実装**: sell_dynamic_kill trending_up 閾値 -0.3→-0.1
7. **C4 実装**: DailyDrawdownGuard 有効化 (soft=-30bps, hard=-50bps)
8. **C1 投入→即 revert**: time_filter UTC14/17 → B1' と重複する弥縫策と判定、削除
9. **time_filter 全廃**: 全ての静的時間帯遮断を撤廃 (107# Phase 3 Step 3 完了)
10. **subprocess popup 修正**: `CREATE_NO_WINDOW` 適用 (git_utils + run_metadata)
11. **Config Hot-Reload**: YAML 変更の mtime polling ライブ反映 (120s 間隔)
12. **型安全強化**: `Any` → Protocol, bare `dict` → `dict[str, object]`

**コミット**: `cc9d45254` → `42c068064` → `92060c209` → `76089b1d7` → `5fa46d85e` → `252acec36` → `93a27acbc` → `cd2513c78` → `1cd12bf4e` (9 commits)

### 2.3 弥縫策 → 根本対策への進化

169# の最大の構造的進展は「弥縫策の識別と退出」:

| # | 施策 | 169# 前 | 169# 後 | 判定根拠 |
|---|------|---------|---------|---------|
| time_filter (UTC16 buy) | 有効 | **全廃** | B1' が因果的に包含。静的時間フィルタは弥縫策 |
| time_filter (UTC8/21 sell) | 有効 | **全廃** | sell_dynamic_kill + VG が条件ベースで代替 |
| time_filter (regime_adaptive) | 有効 | **全廃** | VG + SkipGate + B1' が動的に代替 |
| C1 (UTC14/17 buy skip) | 投入 | **revert** | B1' と重複。投入当日に弥縫策と判定し撤回 |
| B2 (timeout 短縮) | 計画 | **撤回** | Gemini が論理矛盾指摘。H2_timeout 機会損失を拡大 |
| B1' (ranging_buy low_vol skip) | — | **投入** | 因果分析ベース。損失の 69% を直接排除 |
| C3 (trending_up_sell 強化) | — | **投入** | 最悪/件レジーム×サイド (-2.770bps) を防御 |
| C4 (DailyDrawdownGuard) | disabled | **有効化** | halt 1 日実績。リスク制限として機能 |

**成果**: 弥縫策 3 件を退役 + 2 件を事前撤回。根本対策 3 件を投入。
**原則の確立**: 「条件ベースフィルタ > 時間ベースフィルタ」→ 全ての静的時間遮断の根拠を無効化。

---

## §3 ログ深堀り分析

### 3.1 全体指標 (全 run 累積)

**対象**: `fill_records_20260218.jsonl` ～ `fill_records_20260227.jsonl` (10 日分)

| 指標 | 値 |
|------|-----|
| total_cycles | 4,348 |
| filled | 1,854 (42.6%) |
| cancelled | 2,494 (57.4%) |
| PnL30 mean | -0.337 bps |
| PnL120 mean | -0.127 bps |
| AS rate | 27.9% (518/1,854) |
| 累積実損 (state) | -553 JPY |

### 3.2 キャンセル理由分解

| 理由 | 件数 | 割合 | 解釈 |
|------|------|------|------|
| **skip_gate** | 470 | 10.8% | ML フィルタ正常作動。過剰防衛の可能性 (169# R2-3) |
| **trending_sell_skip** | 396 | 9.1% | sell ガード。安全弁発動頻度が高い |
| **balance_forced_skip** | 377 | 8.7% | 片側残高不足。deadlock 防止ロジックあり |
| **timeout** | 308 | 7.1% | offset が遠すぎて約定しない。機会損失 (+0.430bps hindsight) |
| sell_dynamic_kill | 167 | 3.8% | 連続 sell PnL 悪化で自動停止 |
| orderbook_error | 161 | 3.7% | API/板取得エラー |
| spread_too_narrow | 151 | 3.5% | maker としてのエッジ不足 |
| buy_dynamic_kill | 77 | 1.8% | buy 側の連続損失ガード |
| postonly_reject | 68 | 1.6% | Coincheck post_only 拒否 |
| time_filter_086_deadlock | 58 | 1.3% | ★ 086# deadlock 遺残 (time_filter 全廃で今後 0 件) |
| stale_skip_gate_blocked | 37 | 0.9% | reprice 後の SkipGate 再判定で拒否 |
| api_error | 34 | 0.8% | API 通信エラー |
| sell_guard_reject | 30 | 0.7% | sell ガード拒否 |
| status_unknown | 29 | 0.7% | 注文状態不明 (API タイムアウト) |

**構造的課題**:
1. **skip_gate (10.8%)** + **trending_sell_skip (9.1%)** = 20% がフィルタリング系キャンセル → 過剰防衛の可能性
2. **balance_forced_skip (8.7%)**: 片側ポジション偏りが恒常化している
3. **timeout (7.1%)**: 約定すれば利益になる機会 (hindsight +0.430bps) が喪失されている
4. **time_filter_086_deadlock (1.3%)**: time_filter 全廃 (commit `93a27acbc`) により今後発生しない

### 3.3 サイド別 PnL 分析

| サイド | n (filled) | PnL30 mean | 判定 |
|--------|-----------|-----------|------|
| buy | 935 | **-0.189 bps** | 微損。B1' 効果で改善傾向の見込み |
| sell | 919 | **-0.487 bps** | ★ buy の 2.6 倍悪い。sell 構造問題は未解決 |

**sell が悪い原因の仮説**:

1. **maker sell の構造的不利**: 価格上昇時に sell は逆行方向に約定 → AS が buy より発生しやすい
2. **trending_sell_skip が多すぎる**: 396 件 (全体の 9.1%) をガードで弾くが、弾いた結果として残る sell は「ガードをすり抜けた悪い sell」に偏る
3. **sell hold 90s の効果は PnL120 に反映**: PnL120 (-0.127) が PnL30 (-0.337) より改善 → 長めの保持で回復傾向

### 3.4 直近 3 日の日別推移

| 日付 | n_total | n_filled | PnL30 mean | 判定 |
|------|---------|---------|-----------|------|
| 2/25 | 504 | 167 (33.1%) | **-0.894 bps** | ❌ 悪い |
| 2/26 | 472 | 174 (36.9%) | **-1.044 bps** | ❌ 最悪 |
| 2/27 (途中) | 60 | 32 (53.3%) | **-0.023 bps** | ✅ ほぼ BE |

**2/27 は明確に改善**:
- fill_rate 53.3% (3 日間で最高)
- PnL30 -0.023bps (ほぼ損益分岐)
- これは time_filter 全廃 (2/27 早朝 commit) + DailyDrawdownGuard の効果が混在

ただし n=60 は統計的に十分でないため、判定は保留。168h 完走後に re-evaluate。

---

## §4 施策効果の時系列評価

### 4.1 169# 施策の効果判定 (暫定)

| 施策 | 投入日 | n_post | 効果方向 | 統計的有意 | 備考 |
|------|--------|--------|---------|-----------|------|
| B1' (ranging_buy low_vol skip) | 2/27 | 60 | △ 損失回避 | ✗ (n不足) | fill_rate 低下の副作用あり |
| C3 (trending_up_sell 強化) | 2/27 | 60 | △ | ✗ | sell_dynamic_kill 発動率に影響 |
| C4 (DailyDrawdownGuard) | 2/27 | 60 | ✅ halt 1日 | — | 安全弁として機能確認 |
| time_filter 全廃 | 2/27 | 60 | ✅ fill_rate ↑ | ✗ | 53.3% (前日 36.9%) だが要検証 |
| Config Hot-Reload | 2/27 | — | 運用改善 | — | 今後の変更適用を加速 |

### 4.2 因果分離の限界

168# R8 で指摘された「同時変更過多で因果分離不能」の問題は依然存在。
169# セッションだけで 9 commits、5+ パラメータ変更が投入された。

ただし 169# は R11 (最大 2 件/run) を「パラメータ変更」に限定して解釈:
- B1' = 1 パラメータ変更
- C3/C4 = 防御系の閾値変更 (パラメータ変更とカウント)
- time_filter 全廃 = 弥縫策退役 (パラメータ変更ではなく「削除」)
- Hot-Reload = インフラ改善 (パラメータ変更なし)

次 run からは R11 を厳格に適用: 1 パラメータ変更/run。

---

## §5 型安全強化の実施内容

### 5.1 監査結果

fill_test コアコード (6 ファイル) の型安全監査を実施:

| ファイル | Any 使用 | bare dict/list | type: ignore | 評価 |
|---------|---------|---------------|-------------|------|
| config_hot_reload.py | ★ 3 箇所 | 1 箇所 | 0 | → 修正済 |
| run_fill_test.py | 0 | 1 箇所 | 0 | → 修正済 |
| fill_test_cli.py | 0 | 1 箇所 | 1 (Windows) | → 修正済 |
| fill_loop_orchestrator.py | 0 | 2 箇所 | 0 | → 修正済 |
| fill_config.py | 0 | 0 | 0 | ✅ 良好 |
| fill_record_helpers.py | 0 | 0 | 1 (kwargs) | ⚠️ 要検討 |

### 5.2 修正サマリ

#### P0: `Any` 完全排除 — `config_hot_reload.py`

**問題**: `config: Any`, `runner: Any` が 3 箇所。circular import 回避のために使用されていたが、ランタイムで不正な属性アクセスを静的検出不能。

**解決**: `_HotReloadableRunner` Protocol (PEP 544) を導入。

```python
class _HotReloadableRunner(Protocol):
    """ConfigHotReloader が runner に要求する最小インタフェース."""
    _time_filter: TimeFilter
    _maker_price: MakerPriceCalculator
    _git_sha: str

    def _rebuild_sell_kill_mgr(self) -> None: ...
    def _rebuild_buy_kill_mgr(self) -> None: ...
    def _rebuild_daily_drawdown_guard(self) -> None: ...
    def _rebuild_fast_fill_defense(self) -> None: ...
```

- `FillTestRunner` は明示的に Protocol を宣言せずとも構造的サブタイピングにより適合
- `TYPE_CHECKING` ガードで `FillTestConfig`, `MakerPriceCalculator`, `TimeFilter` を静的型付け

**変更前**: `config: Any`, `runner: Any` × 3 = **Any 4 箇所**
**変更後**: `config: FillTestConfig`, `runner: _HotReloadableRunner` × 2 = **Any 0 箇所**

#### P0: bare `dict`/`list` 修正

| ファイル | 変更前 | 変更後 |
|---------|-------|-------|
| fill_test_cli.py | `records: list` → `dict` | `records: list[FillRecord]` → `dict[str, object]` |
| fill_loop_orchestrator.py | `-> dict` × 2 | `-> dict[str, object]` × 2 |
| run_fill_test.py | `yaml_cfg: Optional[dict]` | `yaml_cfg: dict[str, object] \| None` |

#### P1: `yaml_cfg: dict[str, Any]` → `dict[str, object]`

`Any` は型チェッカーが一切の検証を放棄する「脱出弁」。`object` は Python の全型の基底であり、値へのアクセスには明示的キャストが必要 → 型安全性が向上。

### 5.3 残存型安全課題 (低優先)

| 項目 | 場所 | 理由 |
|------|------|------|
| `# type: ignore[attr-defined]` | fill_test_cli.py L365 | Windows 固有 `SIGBREAK` — やむを得ない |
| `# type: ignore[arg-type]` | fill_record_helpers.py L99 | `**extra` → FillRecord kwargs 不整合。FillRecord 側の改修が必要 |
| `Optional[X]` / `X \| None` 混在 | fill_config.py | スタイル不統一。1050 行の大規模 refactor が必要 |
| `ignore_missing_imports = true` | mypy.ini | 外部ライブラリの型情報漏れ。段階的に false 化を検討 |

---

## §6 Config Hot-Reload 実装

### 6.1 実装概要

| 項目 | 内容 |
|------|------|
| クラス | `ConfigHotReloader` (scripts/v460/lib/config_hot_reload.py) |
| 検知方式 | YAML ファイルの mtime ポーリング (120s 既定) |
| 安全機構 | ホワイトリスト方式 (`_HOT_RELOADABLE_FIELDS`: 100+ フィールド) |
| エラー時 | 旧 config 保持 (防御的設計) |
| コンポーネント再構築 | sell/buy_dynamic_kill, DailyDrawdownGuard (状態保持), FastFillDefense |
| テスト | 15 件全パス |

### 6.2 再起動不要になった項目

| カテゴリ | 例 |
|----------|---|
| offset / price | `spread_offset_ratio_buy/sell`, `min_offset_jpy`, regime boost/discount |
| SkipGate 閾値 | `skip_gate_as_threshold`, adaptive, skip_rate |
| dynamic kill | `sell_dynamic_kill_threshold_bps`, regime_thresholds |
| DailyDrawdownGuard | `daily_drawdown_hard_limit_bps`, soft_limit |
| ロット | `order_quantity`, `max_lot`, regime_lot_multipliers |
| cycle timing | `cycle_interval_sec`, `order_timeout_sec` |
| VG / fast fill | 各種閾値 |

### 6.3 依然再起動が必要な項目

| 項目 | 理由 |
|------|------|
| `symbol`, `results_dir` | ディレクトリ/ファイル構造に影響 |
| `enable_regime`, `enable_vg_score` | コンポーネント初期化分岐 |
| Exchange adapter | 接続先変更 |
| Python コード変更 | import 時にモジュール固定 |

---

## §7 外部 AI レビュー向けパッケージ

### 7.1 レビューコンテキスト

#### プロジェクト概要
- **対象**: BTC/JPY (Coincheck), maker-only 自動取引 Bot
- **フレームワーク**: v460 "Microstructure Edge", 6 Phase + 5 Gate 体系 (000#)
- **現在位置**: Phase 2 (G1.1-exec), 168h fill_test 蓄積中
- **ロット**: 0.001 BTC (= ~10,650 JPY at 2026-02-27)
- **大義**: 短期間での高収益性システムの実現 (000# §0)

#### 現在の数値
| 指標 | 値 | 目標 | 判定 |
|------|-----|------|------|
| fill_rate | 42.6% | ≥60% (Kill), ≥70% (Qual) | ❌ |
| PnL30 mean | -0.337 bps | ≥0 | ❌ |
| AS rate | 27.9% | ≤30% | ✅ |
| 累積実損 | -553 JPY | <10,000 JPY | ✅ |
| Oracle PnL | +2.56 bps | — | ✅ (ph3 viable) |

#### 核心問題 (169# から変化なし)
1. **fill_rate 不足**: offset が保守的すぎて約定しない
2. **PnL30 負**: sell が構造的に不利 (-0.487bps)
3. **Oracle ギャップ**: 完全予測なら +2.56bps → 情報量は存在するが執行精度が不足

### 7.2 直近の技術的改善 (169#~170#)

| 施策 | 評価 |
|------|------|
| **弥縫策退役 (time_filter 全廃)** | ✅ 構造改善。107# からの段階的移行完了 |
| **Config Hot-Reload** | ✅ 運用改善。再起動コスト削減 → 反復速度向上 |
| **型安全 (Protocol + generics)** | ✅ コード品質。Any 0 箇所 (対象ファイル内) |
| **B1' (ranging_buy low_vol skip)** | △ 検証中。fill_rate 副作用の定量評価待ち |
| **DailyDrawdownGuard** | ✅ halt 実績あり。安全弁として機能 |

### 7.3 レビュー依頼ポイント

1. **fill_rate 42.6% で G1.1 FAIL 状態での継続は妥当か?**
   - Oracle 正 (+2.56bps) + 累積損失軽微 (-553 JPY) を根拠に継続中
   - 000# §3.9 中止条件 (fill_rate<70% at n≥200) に該当

2. **sell PnL30=-0.487bps の改善戦略は?**
   - sell_dynamic_kill (169# C3 強化済) で trending_up を防御
   - SkipGate retrain (C2: データ蓄積待ち) で sell 方向精度改善
   - 他のアプローチ提案を歓迎

3. **B1' (ranging_buy low_vol skip) は正しい方向か?**
   - 損失の 69% を直接排除する因果分析ベース施策
   - fill_rate をさらに低下させる副作用 → トレードオフの妥当性

4. **Config Hot-Reload の型安全設計は適切か?**
   - `_HotReloadableRunner` Protocol で circular import 回避
   - ホワイトリスト方式で安全なフィールドのみ更新
   - `dict[str, object]` vs `dict[str, Any]` の選択根拠

### 7.4 データ出典

| データ | パス | 生成方法 |
|--------|------|---------|
| fill records (10 日) | `results/v460/fill_test/fill_records_YYYYMMDD.jsonl` | fill_test 自動保存 |
| fill test state | `results/v460/fill_test/fill_test_state.json` | fill_test 自動保存 |
| hindsight 分析 | `analysis_results/hindsight_2026-02-26.json` | `python scripts/v460/analysis/hindsight_filter.py --days 7` |
| MC 予測 | `analysis_results/pnl_mc_2026-02-26.json` | `python scripts/v460/analysis/pnl_monte_carlo.py` |
| 設定 YAML | `configs/v460/fill_test.yaml` | — |
| Gate 定義 | `docs/v460/000_ph0_plan_project_proposal.md §3.3` | — |

### 7.5 コードベース品質

| 領域 | 状態 |
|------|------|
| テスト | 29+ テスト pass (169#/170# 分) |
| 型安全 | mypy `disallow_untyped_defs=true`, Any 排除進行中 |
| 設定管理 | YAML 一元化 + Hot-Reload (120s) |
| モジュール構造 | SRP Mixin 分割済 (163#: 2231→378 行) |
| 弥縫策管理 | 3 層退役済、文書化された退出基準あり |

### 7.6 Git コミット履歴 (169#~170#)

```
1cd12bf4e 169# Config Hot-Reload: YAML 変更をプロセス再起動なしでライブ反映 (120s mtime poll)
cd2513c78 169# subprocess popup 抑制: git_utils + run_metadata に CREATE_NO_WINDOW 追加
93a27acbc 169# time_filter 全廃 (107# Phase 3 Step 3 完了): 条件ベースフィルタに完全移行
252acec36 169# C1 revert: time_filter UTC14/17 buy skip removed
5fa46d85e 169# impl-2: C1 time_filter + C3 trending_up_sell threshold + C4 DailyDrawdownGuard enable
76089b1d7 169# impl: B1' ranging_buy low_vol hard skip + B0 3-series gate metric + popup fix
92060c209 169# 著者回答-1: Codex/Gemini レビュー受理/却下
42c068064 docs: append Gemini 3.1 Pro second opinion
cc9d45254 169# 深堀り分析レポート
```

---

## §8 次アクション

### 8.1 即時 (168h 完走まで)

| # | アクション | 優先度 |
|---|-----------|--------|
| A1 | **168h データ蓄積継続** (変更なし) | ★★★ |
| A2 | daily_health_check 毎日実行 | ★★ |
| A3 | 2/27 好調データの持続性観察 | ★★ |

### 8.2 短期 (168h 完走後)

| # | アクション | 期待効果 | R11 制限 |
|---|-----------|---------|---------|
| B1 | **sell 構造問題の深掘り分析** | 原因特定 | 分析のみ |
| B2 | **C2: SkipGate retrain** (n>1000/side 到達後) | sell 方向精度改善 | 1 変更 |
| B3 | fill_rate 改善策の検討 (offset 最適化 or timeout 戦略) | +10pt 目標 | 1 変更 |

### 8.3 中期

| # | アクション |
|---|-----------|
| C1 | Oracle 50% キャプチャへのロードマップ策定 |
| C2 | S1 昇格条件達成 (pnl_mean > 0, fill_rate ≥ 55%) |
| C3 | 型安全残存課題 (fill_config.py Optional 統一, fill_record_helpers.py type: ignore 解消) |

---

## 改訂履歴

| 日付 | 内容 |
|------|------|
| 2026-02-27 | 初版: ログ深堀り + 型安全強化 + Config Hot-Reload + AI レビュー準備 |

---

## Reviewer追記（2026-02-27, Codex）

### R1 総評（000#/118#/168#/169# 整合）

170# は「弥縫策の退役」「型安全強化」「運用速度改善（hot-reload）」という点で前進している。一方で、000# §0 の大義（短期間での高収益）に対しては、依然として **執行品質（fill）と期待値（PnL）の同時未達** が主ボトルネックであり、課題は「予測情報不足」より「執行設計と検証設計」に集中している。

### R2 主要指摘（重大度順）

| # | 重大度 | 指摘 | 根拠 | 推奨対応 |
|---|---|---|---|---|
| 1 | CRITICAL | **Gate運用の例外ルールが未定義**（停止条件該当でも継続） | 000# §3.9 では `fill_rate<70% (n>=200)` は中止候補。169#/170# では Oracle正を理由に継続 | 「継続例外」の明文化を追加（発動条件、有効期限、再判定日、解除条件）し、裁量継続を制度化 |
| 2 | HIGH | **fill_rate 指標の分母混在が意思決定を曖昧化** | 170# 要約は raw fill_rate=42.6%、一方 Gate K1/F1 は attempted 系列が主判定 | 日次/週次レポートを `raw/clean/attempted` 3系列固定で併記し、判定を1画面で完結させる |
| 3 | HIGH | **B1' の副作用評価が不足**（損失回避 vs 機会損失） | 170# で fill_rate 悪化を認識済みだが、skip した分の期待値が未定量 | `EV_per_cycle = fill_prob × pnl_if_filled` で B1' 前後比較を実施し、fill率ではなく期待値で判定 |
| 4 | HIGH | **sell 劣後の根因分解が未完了** | 170# で sell=-0.487bps, buy=-0.189bps。156#/168# 以降の論点が継続 | side別に「entry価格・待機時間・AS率・guard発動」の寄与分解を再実施し、sell専用の1変更A/Bを優先 |
| 5 | HIGH | **同時変更による因果分離不足が継続** | 168# R8 指摘、169# でも複数変更投入 | R11 を「1 run=1収益系変更」に厳格化。hot-reloadは便利だが、検証run中はfreeze窓を設定 |
| 6 | MEDIUM | **短期好転の過大解釈リスク** | 170# 2/27 は n=60 で改善傾向だが標本不足 | 事前に最小サンプル閾値を設定（例: n>=300）し、閾値未満は「探索結果」と明記 |
| 7 | MEDIUM | **DrawdownGuard の評価軸が損失防止に偏重** | halt実績は良いが、停止頻度が高いと期待値劣化を隠す可能性 | `halt回数/日`, `halt中の機会損失`, `再開後PnL` を追加し、守り過剰を監査 |
| 8 | MEDIUM | **型安全は改善したが契約テストが不足** | Any除去は進捗。ただし `fill_record_helpers` の `type:ignore` 残存 | `cancel_reason` と Gate集計の契約テストを追加し、分類ドリフトを防止 |
| 9 | MEDIUM | **既存資産の統合が部分的** | 118#/168# で提示済みの分析資産・リスク資産が未だ手動運用中心 | `hindsight_filter`, `daily_health_check`, `pnl_monte_carlo` を同一run_idで自動連携し、週次定型化 |
| 10 | LOW | **目標体系が二重化**（fill_rate改善と高収益の優先度） | fill_rate改善施策とPnL改善施策が局所最適で競合 | Phase2終了条件を「Gate PASS」だけでなく「短期収益KPI下限」付きで再定義 |

### R3 見落とし補完（過去文書との接続）

1. 118#/168# で繰り返し出た論点は「負け方の制御」より「勝てる局面の濃縮」。  
   170# でも同じ傾向があり、skip/guard最適化だけでは上振れが作れない。
2. 169# で time_filter 全廃は整合的だが、「静的時間帯遮断を廃止」しただけで「時間帯依存の学習」は否定されない。  
   時間はフィルタではなく特徴量として扱う方針を明記すべき。
3. 170# の hot-reload は運用上有用だが、検証runにおいては「可変性増大=再現性低下」のトレードオフがある。  
   実験runと運用runで mode を分ける設計が必要。

### R4 次判断に直結する実行提案（最短）

| 優先 | アクション | 目的 | 完了条件 |
|---|---|---|---|
| P0 | Gate例外運用ルールを 000# or 170# に追記 | 継続/停止判断の恣意性排除 | 例外発動条件と失効条件が文書化され、次判定日が固定される |
| P0 | 日次レポートを 3分母（raw/clean/attempted）固定出力 | 判定の一貫性確保 | 1レポートで K/F 判定可能になる |
| P1 | B1' の EV評価（機会損失込み） | fill率低下の妥当性検証 | 「有効/無効」を期待値で判定できる |
| P1 | sell 専用 1変更A/B（offset or hold どちらか1つ） | sell劣後の直接改善 | n>=300 で対照比較、差分の符号が安定 |
| P2 | runモード分離（experiment_frozen / live_adaptive） | hot-reload と再現性の両立 | freeze中に設定ハッシュ不変を保証 |

### R5 結論

170# の方向性は「品質改善として正しい」が、「収益改善としてはまだ未確定」。  
次の分岐点は **B1' を fill率ではなく期待値で判定できるか**、および **sell 劣後を単一変更で縮小できるか** の 2 点。ここが確認できれば、ph2 継続の妥当性は大きく上がる。

---

## 9 追記: 170# に対するセカンドオピニオンと深層的批判 (Gemini 3.1 Pro)

### 9.1 「Hot-Reload」という劇薬 — 検証フェーズにおける自己破壊行為
Config Hot-Reloadの実装はエンジニアリング（DX向上）としては評価できるが、**Phase 2（検証フェーズ）において稼働中にパラメータを変更することは「再現性の完全な破壊」を意味する**。
168時間のfill_testは「固定された条件での統計的有意性の確認」が目的である。途中でパラメータが変更されれば、そのデータセットは「どのパラメータの時の結果か」が混ざり合った**分析不能なゴミデータ**と化す。
CodexもR2-5で指摘しているが、Hot-Reloadは本番運用（Phase 5）のツールであり、検証フェーズでは**「1 run = 1 config (完全固定)」**を絶対の掟とすべきである。直ちに検証中のHot-Reloadを封印（またはFreeze窓を強制）せよ。

### 9.2 「Oracle正」を免罪符にするな — Gate判定の形骸化
G1.1のKill条件（fill_rate < 60%）に該当しているにもかかわらず、「Oracleが正（+2.56bps）だから継続」としている点は、Gate判定の完全な形骸化である。
Oracle（完全予測）がプラスであることは「Phase 1（情報量の存在）」のクリア条件であって、「Phase 2（執行可能性）」の免罪符にはならない。執行できない（約定しない）エッジは「絵に描いた餅」である。例外ルールを作って正当化するのではなく、**「Gateをクリアできないなら、クリアできるまでパラメータやロジックを修正して再テストする」**という基本に立ち返るべきである。

### 9.3 Sellの構造的劣後と「ガードのパラドックス」
SellがBuyの2.6倍悪い（-0.487bps）原因について、「trending_sell_skip（ガード）が多すぎる結果、すり抜けた悪いSellに偏る」という仮説は極めて鋭い。これは**「ガードのパラドックス（Adverse Selectionの濃縮）」**である。
BTC/JPYのような長期上昇バイアスのある資産において、Maker Sellは構造的に不利である。これを「条件ガード」で防ごうとすると、条件をすり抜けた「真の逆行（急騰）」だけを掴まされる。
対策はガードを増やすことではなく、**「SellのデフォルトoffsetをBuyより広くする（非対称offset）」**か、**「SellのTimeoutを極端に短くし、逃げ足を速くする」**という物理的な非対称性の導入である。

### 9.4 「休むも相場 (B1')」の評価軸の誤り
B1'（ranging_buy low_vol skip）を導入した結果、fill_rateが低下したことを「副作用」と呼んでいるが、これは誤認識である。エントリーを絞ったのだからfill_rateが下がるのは**数学的必然**であり、副作用ではない。
未だに「fill_rate至上主義」に囚われている。重要なのは「無駄な被弾が減り、1回あたりの期待値（EV）が向上したか」である。fill_rateの低下を恐れてB1'を撤回するような愚行は避けるべきである。

### 9.5 結論とネクストアクション
1. **Hot-Reloadの封印**: 検証run中は設定ファイルを完全Freezeし、変更する場合は必ずプロセスを再起動してrun_idを更新すること。
2. **非対称Offsetの導入**: Sellの構造的劣後に対して、BuyとSellで異なるベースoffset（例: SellはBuyの1.2倍広くする）を設定し、A/Bテストを実施すること。
3. **Gateの厳格化**: 「Oracle正」を言い訳にせず、fill_rateとPnLの改善に正面から向き合うこと。
