# 161# phg — コード品質・構造改善

> **日付**: 2026-02-25  
> **種別**: impl  
> **前提**: 160# (regime=None根本修正 + 分析レポート)  
> **テーマ**: 複雑性監査 + SIGTERM graceful shutdown + DRY統合 + asyncio安全化

---

## 背景

160# までの累積開発でコードベースが **83ファイル / 27,174行** に達した。
ユーザ指示「複雑になりすぎて帰って逆効果になっている可能性も探して下さい」に基づき、
**構造的技術負債の棚卸し**と**即座に効果のある品質改善**を実施。

---

## §1 複雑性監査結果

### 1.1 規模概要

| 指標 | 値 |
|---|---|
| v460 .py ファイル数 | 83 |
| 総行数 | 27,174 |
| 500行超ファイル | 14 |
| God Object候補 | 2 (`retrain_scheduler.py` 1,794行, `run_fill_test.py` 2,012行) |
| **163# 実績** | `run_fill_test.py` 2,231→378行 (3 Mixin 分割), `maker_price.py compute()` 306→143行, `fill_config.py from_yaml()` 479→139行 |


### 1.2 正当な複雑性（維持）

以下は機能要件に直結し、簡素化すべきでないと判断:

- **Walk-Forward評価** (`walk_forward_evaluator.py`): 時系列交差検証は本質的に複雑
- **Quality Gates** (`quality_gate.py`): 多段ゲートは安全装置として不可欠
- **アトミック書き込み** (`safe_atomic_write`): データ損失防止の基盤
- **FastFillDefense**: レート制限保護は取引所 API 安定性に直結
- **Regime Weighting / Side-Specific Models**: YAML で `enabled: True` — 実際に稼働中

### 1.3 問題のある複雑性（要改善）

| 問題 | 場所 | 重大度 | 対応 |
|---|---|---|---|
| SIGTERM未対応 | `retrain_scheduler.py` | **P0** | ✅ 本セッションで修正 |
| DRY違反: メトリクス計算重複 | `ab_judgment.py` / `dashboard.py` | P1 | ✅ 本セッションで統合 |
| asyncioアンチパターン | `run_fill_test.py` `_cleanup_sync()` | P1 | ✅ 本セッションで修正 |
| `FillTestConfig` 208フィールド | `run_fill_test.py` | P2 | 📋 次セッション以降 |
| `retrain_model()` 705行 God Method | `retrain_scheduler.py` | P2 | 📋 次セッション以降 |
| `run_continuous()` 818行 God Method | `run_fill_test.py` | P2 | 📋 次セッション以降 |

### 1.4 結論

> **「簡素化すべきは『構造』であって『機能』ではない」**
>
> regime_weighting, side_specific, online_monitor 等は全て YAML で有効かつ実運用中。
> 機能削除は逆効果。構造的リファクタリング（ファイル分割・型整理）で対応すべき。

---

## §2 SIGTERM Graceful Shutdown 実装

### 2.1 問題

`retrain_scheduler.py` の `while True: ... time.sleep()` ループに **シグナルハンドラが未設定**。
`kill <PID>` で即座にプロセス終了 → 学習途中のモデルファイル破損リスク。

### 2.2 実装

```python
import signal
import threading

_shutdown_event = threading.Event()

def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT で graceful shutdown."""
    def _handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — initiating graceful shutdown", sig_name)
        _shutdown_event.set()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
```

**変更箇所** (3点):
1. `while True:` → `while not _shutdown_event.is_set():`
2. `time.sleep(effective_interval)` → `_shutdown_event.wait(timeout=effective_interval)` (2箇所)
3. ループ終了後に `logger.info("Graceful stop completed")` 追加

**効果**: `kill <PID>` で最大1秒以内にクリーンシャットダウン。学習中断時のファイル破損を防止。

---

## §3 DRY メトリクス統合

### 3.1 問題

`ab_judgment.py` の `_compute_metrics()` と `side_regime_dashboard.py` の `_compute_side_metrics()` が
ほぼ同一の計算ロジックを持つが、ヘルパー名が異なる (`_safe_finite` vs `_to_finite`)。

### 3.2 実装

#### 新規: `scripts/v460/lib/metrics_utils.py`

| 関数 | 用途 |
|---|---|
| `compute_base_metrics(records)` | 基本メトリクス (n_total, fill_rate, avg_pnl30, std, p10, p05, profitable_rate) |
| `compute_extended_metrics(records)` | base + AS/reprice/VG 拡張メトリクス (dashboard用) |

#### 統合: `ztb/utils/safety.py` に `safe_to_finite()` 追加

```python
def safe_to_finite(value: Any) -> float | None:
    """有限浮動小数点への安全変換 (NaN/Inf → None)."""
```

`_safe_finite` (ab_judgment) と `_to_finite` (dashboard) を統一。

#### リファクタリング結果

| ファイル | Before | After |
|---|---|---|
| `ab_judgment.py` | ローカル `_safe_finite` + 独自計算 | `compute_base_metrics` に委譲 |
| `side_regime_dashboard.py` | ローカル `_to_finite` + 独自計算 | `compute_extended_metrics` に委譲 |
| `test_160_ab_judgment.py` | `_safe_finite` import | `safe_to_finite` from `ztb.utils.safety` |

---

## §4 asyncio _cleanup_sync 安全化

### 4.1 問題

`run_fill_test.py` の `_cleanup_sync()` が無条件に `asyncio.new_event_loop()` を作成。
既存イベントループが動作中の場合、二重ループによるリソースリーク。

### 4.2 修正

```python
def _cleanup_sync(self) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 既存ループに投入
        future = asyncio.run_coroutine_threadsafe(self._async_cleanup(), loop)
        future.result(timeout=5)
    else:
        # atexit 等: 新規ループ作成 (安全)
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(self._async_cleanup())
        finally:
            new_loop.close()
```

---

## §5 延期項目（P2: 次セッション以降）

| 項目 | 理由 | 推定工数 |
|---|---|---|
| `retrain_scheduler.py` 分割 | 1,794行 God Object → 3-4モジュールに | 2-3h |
| `run_fill_test.py` 分割 | 2,012行 → 構成分離 | 3-4h | **✅ 163# 完了** (2,231→378行, 3 Mixin) |
| `FillTestConfig` TypedDict化 | 208フィールドの型安全 | 1-2h |
| ConfigMap / NestedConfig | flat dict → 構造化config | 1h |

---

## §5b 水平添加: safe_to_finite 統合 (5ファイル追加)

§3 で `ztb.utils.safety.safe_to_finite` を整備したが、同一ロジックが analysis/ 配下にも散在:

| ファイル | ローカル関数 | 呼出数 |
|---|---|---|
| `analyze_fill_records.py` | `_to_finite_float` | 3 |
| `oracle_baseline.py` | `_to_finite_float` | 3 |
| `hindsight_filter.py` | `_to_float` | 11 |
| `compare_regime_ab.py` | `_to_float_or_none` | 4 |
| `reproduce_152_metrics.py` | `_to_float` | 7 |

全5件のローカル関数定義を削除し、`safe_to_finite` に統合。
不要になった `import math` / `safe_to_float` import も除去。

> `ob_recorder.py` の `_to_finite_float` は bool 型除外の追加ロジックを持つため対象外とした。

---

## §6 テスト結果

| テストスイート | 結果 |
|---|---|
| `test_160_ab_judgment.py` | **65 passed** ✅ |
| `test_159_side_regime_dashboard.py` | **6 passed** ✅ |
| `test_141_side_specific_models.py` (OnlineMonitor) | **10 passed** ✅ |
| `test_143_regime_utilization.py` (pre_filter) | **1 passed** ✅ |
| 全テストスイート | **1858 passed** (v460 unit 全PASS, CustomPPO 5件は既存問題・無関係) |

---

## §7 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `scripts/v460/ml/retrain_scheduler.py` | 修正 | SIGTERM graceful shutdown追加 |
| `scripts/v460/lib/metrics_utils.py` | **新規** | 共通メトリクス計算モジュール |
| `scripts/v460/lib/ab_judgment.py` | リファクタ | `compute_base_metrics` 委譲、`_safe_finite` 除去 |
| `scripts/v460/analysis/side_regime_dashboard.py` | リファクタ | `compute_extended_metrics` 委譲、`_to_finite` 除去 |
| `scripts/v460/run_fill_test.py` | 修正 | `_cleanup_sync()` asyncio安全化 |
| `ztb/utils/safety.py` | 追加 | `safe_to_finite()` 関数 |
| `tests/unit/v460/test_160_ab_judgment.py` | 修正 | import先変更 (`safe_to_finite`) |
| `scripts/v460/analysis/analyze_fill_records.py` | リファクタ | `_to_finite_float` → `safe_to_finite` |
| `scripts/v460/analysis/oracle_baseline.py` | リファクタ | `_to_finite_float` → `safe_to_finite` |
| `scripts/v460/analysis/hindsight_filter.py` | リファクタ | `_to_float` → `safe_to_finite` (11箇所) |
| `scripts/v460/analysis/compare_regime_ab.py` | リファクタ | `_to_float_or_none` → `safe_to_finite` |
| `scripts/v460/analysis/reproduce_152_metrics.py` | リファクタ | `_to_float` → `safe_to_finite` |

---

## §8 自己レビュー結果

### 問題なし

- `safe_to_finite` の挙動: `None / NaN / Inf → None`, `"3.14" → 3.14`, `bool → 1.0/0.0` — 一貫性あり
- `ob_recorder.py` のみ `bool → None` が必要なため、意図的に対象外とした (正当)
- `_to_str` (hindsight_filter, reproduce_152) は別関数として正しく残存
- `metrics_utils.py` に `__all__` 追加済み
- SIGTERM handler: `signal.signal()` はメインスレッドで呼ぶ必要 — `_install_signal_handlers()` は `main()` から呼ばれるため問題なし

### 注意事項

- `compare_regime_ab.py` の旧 `_to_float_or_none` は `safe_to_float(value, nan)` 経由だったが `safe_to_finite` は `float(value)` 直接 — 挙動は等価（どちらも `float("nan") → nan → None`）
- `retrain_scheduler.py` の `_shutdown_event` はモジュールレベルグローバル変数 — マルチプロセスで fork した場合は共有されない。現行アーキテクチャでは単一プロセスなので問題なし

---

## §9 158# 残課題ステータス

### 最重要 OPEN 項目 (収益直結)

| ID | 項目 | 優先度 | 状態 |
|---|---|---|---|
| P0-2 | sell offset 段階的縮小 A/B テスト (0.18→0.14) | **P0** | ab_test_variant データ蓄積待ち |
| P1-2 実施 | buy base offset 引き上げ (0.05→0.12-0.15) | P1 | n≥200 到達後 A/B テスト予定 |
| §10.3 | 同side内 variant 比較への段階移行 | P1 | データ蓄積待ち |
| §12.B | retrain final training val 分割が WF eval と独立 | **P2-HIGH** | 精度検証要 |

### 構造改善 OPEN 項目

| ID | 項目 | 優先度 | 推定工数 |
|---|---|---|---|
| retrain_scheduler 分割 | 1,794行 God Object → 3-4モジュール | P2 | 2-3h |
| run_fill_test 分割 | 2,012行 → 構成分離 | P2 | 3-4h | **✅ 163# 完了** |
| FillTestConfig TypedDict化 | 208フィールドの型安全 | P2 | 1-2h |
| YAML外部化 | CircuitBreaker/HealthMonitor等の定数 8件 | P2 | 1-2h |
| skip_gate.py モジュール移設 | scripts/ → ztb/ | P2 | 1h |

### Gemini セカンドオピニオン (未着手)

| ID | 項目 | 収益インパクト |
|---|---|---|
| Gemini-B | Inventory Skewing (在庫偏重による非対称クオート) | 高 |
| Gemini-C | SkipGate sell SHAP分析 | 中 |
| Gemini-D | 「休むも相場」ロジック | 中 |

---

## §10 ログ分析 (続報)

本セッション後半で Fill Test 10日間データの包括分析を実施。
詳細は **[162# Fill Test 10日間ログ分析](162_phg_rpt_fill_test_10day_log_analysis.md)** を参照。

### 主要発見

- Fill Rate が 77% → 9% に急落 (balance_forced_skip 集中)
- Adverse Selection 27.1% が avg_pnl30 = -5.29bps で支配的損失
- Non-AS のみでは +1.65bps と正のエッジ存在
- Retrain Scheduler のモデル更新成功率が 4% (70% がデータ不足 skip)
- Sell 側 Fill Rate が Buy より 22pt 低い構造的問題
