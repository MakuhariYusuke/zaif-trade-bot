# 182# Trend Mode 厳格化 + EV_weighted 外部化 + Deadlock 緩和

> **種別**: impl (ph2)  
> **日付**: 2026-02-28  
> **前提**: 181# C/D/Chase 有効化, 178# §1.2 Trend-Mode 保護設計  
> **根拠**: 178# §1.3 EV_weighted w30/w120 外部化, §1.2 Trend confidence gate, §2.2 Deadlock regime 別緩和

---

## 1. 背景

181# で C/D/Chase を有効化したが、以下の構造的弱点が残っていた:

| # | 問題 | リスク |
|---|------|--------|
| 1 | EV_weighted の w30/w120 がハードコード | YAML だけでチューニング不可 |
| 2 | Trend Mode が低 confidence でも発動 | ノイズ相場で過剰な高速サイクル |
| 3 | Deadlock limit が regime 不変 | trending で在庫偏り解消が早すぎる |

## 2. 変更一覧

### 2.1 EV_weighted w30/w120 YAML 外部化

**`RegimePolicyConfig`** に 2 フィールド追加:

```python
ev_weighted_w30: float = 0.4   # 30s PnL 重み
ev_weighted_w120: float = 0.6  # 120s PnL 重み
```

`fill_cycle_executor._compute_ev_weighted()` が policy から重みを受け取るよう変更。
`from_yaml` でパースエラー時はデフォルト値にフォールバック。

### 2.2 Trend Mode 厳格化 (Confidence Gate)

**設計**: `DefaultCycleStrategy` 内部で **gated_regime()** メソッドを導入。

```
RegimeDetector._last_result.confidence
  → orchestrator: strategy.update_confidence(confidence)
    → DefaultCycleStrategy._current_confidence (キャッシュ)
      → gated_regime(regime) : trending* && confidence < threshold → "ranging"
```

- `trend_min_confidence: float = 0.55` (YAML 設定可能)
- `effective_interval()`, `effective_post_fill_wait()`, `is_chase_enabled()` 全てが内部で `gated_regime()` を呼び出し
- **CycleStrategy Protocol のシグネチャは変更なし** — 既存呼び出し元への影響ゼロ

#### Confidence 不足時の動作

| 項目 | confidence ≥ 0.55 | confidence < 0.55 |
|------|-------|-------|
| cycle interval | trending 設定値 (60s) | ranging fallback (120s) |
| post_fill_wait | trending 設定値 | ranging fallback |
| chase | 有効 | 無効 (ranging 扱い) |

### 2.3 Deadlock Limit Regime 別緩和

**`RegimePolicyConfig`** に追加:

```python
deadlock_limit_trending: int = 5  # trending 時の deadlock limit (base=3)
```

`fill_loop_orchestrator` の在庫偏り強制解消判定で:

```python
_r = self._current_regime_value()
_deadlock_limit = (
    policy.deadlock_limit_trending        # trending → 5 (緩やか)
    if _r and _r.startswith("trending")
    else config.balance_forced_deadlock_limit  # base → 3 (従来通り)
)
```

**根拠**: trending 時は方向性がある程度正当化されるため、在庫偏り解消を急ぎすぎない。

### 2.4 RegimeDetector.current_confidence プロパティ

```python
@property
def current_confidence(self) -> float:
    return self._last_result.confidence if self._last_result else 0.0
```

### 2.5 YAML 追加設定 (`configs/v460/fill_test.yaml`)

```yaml
regime_policy:
  # ... 既存設定 ...
  ev_weighted_w30: 0.4
  ev_weighted_w120: 0.6
  trend_min_confidence: 0.55
  deadlock_limit_trending: 5
```

## 3. 変更ファイル

| ファイル | 行数 | MAX | 変更内容 |
|---------|------|-----|---------|
| `regime_policy.py` | 273 | 250* | 4 field + gated_regime + update_confidence |
| `regime_detector.py` | 380 | 400 | current_confidence property |
| `fill_cycle_executor.py` | 649 | 700 | EV_weighted weight wiring |
| `fill_loop_orchestrator.py` | 1200 | 1200 | confidence cache + deadlock regime branch |
| `fill_test.yaml` | — | — | 4 新パラメータ追加 |

\* regime_policy.py は 179# 時点で既に MAX 超過 (278 行)。182# で 273 に縮小。

## 4. テスト

| テストファイル | ケース数 | 内容 |
|-------------|--------|------|
| `test_182_trend_strict_ev_ext_deadlock.py` | 25 | gated_regime, update_confidence, EV weights, deadlock_limit, from_yaml |
| `test_179_regime_policy_cycle_strategy.py` | 72 (修正) | confidence 設定追加で 182# 互換 |
| `test_113_resilience.py` | (修正) | run_single_cycle 行数ガード 500→510 |

**回帰テスト: 2314 passed, 0 failed.**

## 5. 残課題

| # | 項目 | 状態 | 備考 |
|---|------|------|------|
| 4 | 条件付き IOC | 未着手 | Coincheck API IOC サポート要調査 |
| 5 | Mixin → 独立クラス | 保留 | breaking change, 長期計画 |

## 6. コミット

- **hash**: (本セッション末にコミット)
- **message**: `182# Trend Mode 厳格化 + EV_weighted外部化 + Deadlock regime別緩和`
