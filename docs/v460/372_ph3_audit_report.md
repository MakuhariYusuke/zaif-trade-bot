# 372# Phase 3: Post-Implementation Audit Report

**監査日**: 2026-03-10
**対象範囲**: 372# dust sweep buy-to-clear, SAC sidecar wiring, FillRecord 監査証跡, confidence 動的計算, deploy gate min_trade_count
**監査者**: AI audit (GitHub Copilot)

---

## 全体サマリ

| 分類 | 件数 |
|------|------|
| CRITICAL | 1 |
| IMPORTANT | 3 |
| INFORMATIONAL | 4 |

---

## Finding 1: `sidecar_bias` FillRecord round-trip 断絶 [IMPORTANT]

**ファイル**: `scripts/v460/lib/fill_cycle_executor.py` L1082
**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py` L397

### 現状

```python
# fill_cycle_executor.py L1082
sidecar_bias=None,  # 372# bias は gate_result 経由で取得 (後続改善)
```

`sidecar_offset_bps` は `gate_result → run_single_cycle() → _build_fill_record()` と正しく伝搬されるが、
`sidecar_bias` (SAC directional_bias raw 値) は `run_single_cycle()` のパラメータに存在せず、
常に `None` で記録される。

`CycleGateResult.sidecar_bias` には正しい値が設定されているが（`cycle_gate_aggregator.py` L384）、
`orchestrator_mid_cycle.py` L397 では `sidecar_offset_bps` のみ渡され `sidecar_bias` は渡されていない。

### 影響

- FillRecord の監査証跡が不完全（offset は記録されるが、元の bias 値が欠損）
- 事後分析で SAC モデルの推論品質を評価できない
- **金銭的損失リスク**: なし（offset 自体は正しく適用される）

### 推奨アクション

1. `run_single_cycle()` に `sidecar_bias: float | None = None` パラメータ追加
2. `orchestrator_mid_cycle.py` で `gate_result.sidecar_bias` を渡す
3. `_build_fill_record()` 呼出しで `sidecar_bias` を正しく伝搬

---

## Finding 2: `from_yaml_dict()` が `confidence_roi_full` / `min_trade_count` を未パース [CRITICAL]

**ファイル**: `scripts/v460/ml/sac_retrain_scheduler.py` L113-164

### 現状

```python
@classmethod
def from_yaml_dict(cls, cfg: dict) -> SACRetrainConfig:
    retrain_cfg: dict = cfg.get("sac_retrain", {})
    return cls(
        # ... 多数のフィールドをパース ...
        min_new_rows=int(retrain_cfg.get("min_new_rows", 120)),
        history_path=Path(str(retrain_cfg.get("history_path", ...))),
        # ❌ confidence_roi_full が未パース
        # ❌ min_trade_count が未パース
    )
```

`SACRetrainConfig` に 372# で追加された 2 フィールドが `from_yaml_dict()` で読み込まれていない:
- `confidence_roi_full` (default 0.005) — YAML で調整不可
- `min_trade_count` (default 3) — YAML で調整不可

### 影響

- YAML 設定ファイルでこれらの値をチューニングしても無視される
- 本番環境で deploy gate の閾値を変更できない
- **金銭的リスク**: 不適切な min_trade_count でモデルが通過/却下される可能性

### 推奨アクション

`from_yaml_dict()` の `return cls(...)` に以下を追加:

```python
confidence_roi_full=float(retrain_cfg.get("confidence_roi_full", 0.005)),
min_trade_count=int(retrain_cfg.get("min_trade_count", 3)),
```

YAML サンプル (g2_sac_train.yaml) にもキーを追記すること。

---

## Finding 3: sidecar signal TTL vs retrain interval の不整合 [IMPORTANT]

**ファイル**: `scripts/v460/lib/sidecar_types.py` L44
**ファイル**: `scripts/v460/ml/sac_retrain_scheduler.py` L105-106

### 現状

```
DEFAULT_SIGNAL_TTL_SEC = 600   (10分)
retrain_interval_sec   = 7200  (2時間)
check_interval_sec     = 300   (5分)
```

Signal は retrain 成功時にのみ更新されるため:
- retrain 成功 → signal 有効 10 分 → stale → **残り 110 分間 sidecar 無効**
- sidecar offset が適用されるのは全体の約 8% の時間のみ

### 影響

- SAC sidecar の効果が大幅に制限される
- スケジューラ正常動作中でも 90% 以上の時間帯で offset=0

### 推奨アクション

以下のいずれかを検討:

1. **推論専用ループ追加**: retrain とは別に `check_interval_sec` 毎に推論のみ実行し signal 更新
2. **TTL 延長**: `DEFAULT_SIGNAL_TTL_SEC` を `retrain_interval_sec` の 1.5 倍程度に設定 (例: 10800s)
3. **折衷**: TTL は 1800s (30分) にし、推論ループを 15 分毎に実行

選択肢 1 が SAC の設計意図に最も合致（最新市場データでの推論）。

---

## Finding 4: `_evaluate_model()` が最終エピソードのみの trade_count / ROI を使用 [IMPORTANT]

**ファイル**: `scripts/v460/ml/sac_retrain_scheduler.py` L571-599

### 現状

```python
def _evaluate_model(model, env, cfg):
    total_reward = 0.0
    for _ in range(cfg.n_eval_episodes):  # default 3
        obs, _ = env.reset()         # ← trades_count = 0 にリセット
        done = False
        while not done:
            ...
    # env.trades_count は最後の 1 エピソードの値のみ
    return {
        "gross_roi": roi,            # 最後のエピソードの ROI
        "trade_count": int(getattr(env, "trades_count", 0)),  # 最後のエピソード
    }
```

`HeavyTradingEnv.reset()` は `trades_count = 0` にリセット（`state_manager.py` L25）。
`n_eval_episodes=3` のとき、ROI と trade_count は最終エピソードのみの値。

### 影響

- `min_trade_count` gate が不安定: エピソードによって trade 数が変動
- ROI gate がエピソード間平均ではなく最後の 1 回だけで判定

### 推奨アクション

累積またはエピソード平均を使用:

```python
total_trades = 0
rois = []
for _ in range(cfg.n_eval_episodes):
    obs, _ = env.reset()
    ...
    total_trades += getattr(env, "trades_count", 0)
    rois.append(roi_for_this_episode)
return {
    "gross_roi": sum(rois) / len(rois),
    "trade_count": total_trades,
}
```

---

## Finding 5: dust_buy_pending 例外時のクリア漏れ (低リスク) [INFORMATIONAL]

**ファイル**: `scripts/v460/lib/orchestrator_mid_cycle.py` L406-426

### 現状

```python
except Exception as e:
    ...
    self._balance_checker.restore_lot_after_dust_sweep()  # lot 復元
    # ❌ clear_dust_buy_pending() なし → pending 状態が継続
    return

self._balance_checker.restore_lot_after_dust_sweep()
# 正常パスでのみ clear
if self._balance_checker.dust_buy_pending and next_side == "buy":
    self._balance_checker.clear_dust_buy_pending()
```

### 影響

- 例外発生 → pending 維持 → 次ループで再試行
- これは実質的に自動リトライとして機能しており、**設計上は問題ない**
- ただし retry limit / timeout がないため、極端なケースでは永続ループの可能性

### 推奨アクション

- 現状は許容範囲（balance check が無限オーダーを防止）
- 将来的には `_dust_buy_retry_count` + 上限追加を検討

---

## Finding 6: TODO/FIXME 残留 [INFORMATIONAL]

### scripts/v460/ 内

| ファイル | 行 | 内容 | 重要度 |
|---------|-----|------|-------|
| `scripts/v460/ml/skip_gate.py` | L6 | `ztb/models/` への移行検討 | 低 (アーキテクチャ) |
| `scripts/v460/analysis/vg_and_trend.py` | L134 | regex パースは脆い | 低 (分析ツール) |

### ztb/ 内

| ファイル | 行 | 内容 | 重要度 |
|---------|-----|------|-------|
| `ztb/trading/live/order_state.py` | L404,412,419 | Coincheck 固有処理の TODO 3件 | 中 |
| `ztb/trading/live/core/reconciliation.py` | L568,576,584 | Coincheck 固有照合の TODO 3件 | 中 |
| `ztb/trading/environment/bridge.py` | L624,687,719 | Zaif API 実装の TODO 3件 | 低 (bridge 未使用?) |
| `ztb/trading/live/simulation/paper_trader.py` | L147,335 | 実装未完了の TODO 2件 | 低 |

### 推奨アクション

- コアトレーディングパス上の TODO (`order_state.py`, `reconciliation.py`) は優先的に対処
- 分析ツール/bridge の TODO は低優先

---

## Finding 7: YAML サンプル設定未反映 [INFORMATIONAL]

**ディレクトリ**: `configs/`

### 現状

`configs/` 内の YAML ファイルに `confidence_roi_full` / `min_trade_count` キーが存在しない。
Finding 2 の修正に伴い、YAML サンプルファイルにもデフォルト値付きでキーを追記すべき。

---

## Finding 8: テストカバレッジギャップ [INFORMATIONAL]

### 未カバレッジの重要パス

| パス | テスト状況 | 重要度 |
|-----|-----------|-------|
| `from_yaml_dict()` + `confidence_roi_full` / `min_trade_count` | ❌ パース自体が未実装 | CRITICAL (Finding 2 と連動) |
| `_evaluate_model()` 複数エピソードでの trade_count 累積 | ❌ | IMPORTANT (Finding 4 と連動) |
| sidecar signal stale 時の fill_test フォールバック挙動 | ⚠ 間接テストのみ | 低 |
| dust_buy_pending + buy 例外 → lot 状態の整合性 | ❌ | 低 |
| `retrain_once()` model deploy 成功 + signal 更新失敗 | ❌ | 低 |

### 既存テストの良い点

- `test_sidecar_sac_integration.py`: 15 テストクラス、types/IO/gate/FillRecord を網羅的にカバー
- `test_dust_sweep.py`: micro-dust 検出、pending フラグ、auto-clear を網羅
- `test_sac_retrain_scheduler.py`: trigger/config/deploy/history を網羅

---

## 推奨対応優先度

1. **即時対応** (CRITICAL): Finding 2 — `from_yaml_dict` に 2 フィールド追加
2. **今週中** (IMPORTANT): Finding 1 — sidecar_bias round-trip 修正
3. **今週中** (IMPORTANT): Finding 4 — `_evaluate_model` 複数エピソード対応
4. **設計検討** (IMPORTANT): Finding 3 — signal TTL vs retrain interval の不整合解消
5. **次スプリント**: Finding 5-8 — INFORMATIONAL 項目
