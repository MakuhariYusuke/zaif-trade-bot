# 596# Primary Model 連続 Skip 安全弁 — Death Spiral 防止

## 背景: 5:58 JST 以降の取引全停止

2026-03-24 05:56:18 JST の buy fill (SHA 8e37cf96) を最後に、約12時間以上取引が完全停止。
hot_swap_restart (06:08:59 JST, PID 3012, SHA 29fe26ed1) 後も回復せず。

## 根本原因分析

### Death Spiral メカニズム

```
BTC=0 → sell preflight_insufficient → side freeze
  → buy 選択 → skip_gate primary model が reject (ev_normal_skip)
  → fill なし → BTC 回復不能 → ∞ ループ
```

**3つの設計ギャップが重なった結果:**

| # | 問題 | 詳細 |
|---|------|------|
| 1 | 190# A 安全弁の死コード化 | `ev_as_offset_enabled=true` (193#) により `_try_ev_weighted_decision` が offset 分岐を取り、190# A 連続 skip カウンタが常時リセット |
| 2 | 190# B threshold 緩和の死コード化 | `one_sided_balance` フラグが同じく offset 分岐で完全に無視 |
| 3 | Primary model 側に安全弁なし | ev_weighted 安全弁は ev_weighted 判定パスのみ。Primary model の skip/pass 決定には安全弁が存在しない |

### ログ証拠

- **Sell 側** (96 records): 全て `preflight_insufficient`, `mid_at_order=None` (market data 取得前に abort)
- **Buy 側** (14 records): 全て `skip_gate` / `ev_normal_skip`, sg_score: -0.37 ~ -3.88, threshold: 0.15 ~ -1.356
- **API 正常**: buy records に `spread=4212`, `order_price=11248636` 等の market data あり
- **EV positive でも block**: ev_score=+2.25 でも sg_score=-0.499 < threshold=-0.353

### コードパス (offset モード)

```
active_gate.evaluate()     ← Primary model: should_skip=True (score=-2.5)
  ↓
_try_ev_weighted_decision()
  ↓ ev_as_offset_enabled=True
  ↓ _ev_weighted_as_offset()
  ↓   ev_score > -5.0 (not toxic) → should_skip=False, offset 修飾のみ
  ↓   _ev_consecutive_skip_count = 0  ← 常時リセット！
  ↓
_ev_combined.should_skip = False → primary decision 維持
  ↓
decision.should_skip = True (primary の判断がそのまま最終結果)
  ↓ ← ここに安全弁が無かった
_apply_decision_to_result() → SKIP
```

## 修正内容

### 596# Primary model 連続 skip 安全弁 (evaluator-level)

190# A が ev_weighted パス内で offset モード時に無効化される問題を、
**evaluator-level** に mode 非依存の安全弁を追加して根本修正。

| ファイル | 変更 |
|----------|------|
| `scripts/v460/lib/fill_config.py` | `skip_gate_primary_max_consecutive_skip: int = 0` 追加 |
| `scripts/v460/lib/fill_config_parser.py` | `primary_max_consecutive_skip` → config field マッピング追加 |
| `scripts/v460/lib/skip_gate_evaluator.py` | `_primary_consecutive_skip_count` カウンタ + 安全弁ロジック追加 |
| `configs/v460/fill_test.yaml` | `primary_max_consecutive_skip: 10` 追加 |
| `tests/unit/v460/test_596_primary_consecutive_skip_safety.py` | 8 テスト (config, counter, YAML 統合) |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | `KNOWN_YAML_OVERRIDES` にフィールド追加 |

### 安全弁ロジック

```python
# skip_gate_evaluator.py: ev_weighted 処理後、_apply_decision_to_result 前
_primary_max = config.skip_gate_primary_max_consecutive_skip
if decision.should_skip and _primary_max > 0:
    self._primary_consecutive_skip_count += 1
    if self._primary_consecutive_skip_count >= _primary_max:
        # 強制 PASS: decision を上書き
        decision = SkipDecision(should_skip=False, reason="primary_safety_valve_pass", ...)
        self._primary_consecutive_skip_count = 0
elif not decision.should_skip:
    self._primary_consecutive_skip_count = 0
```

### 設定値

- `primary_max_consecutive_skip: 10` (YAML)
- 10 サイクル連続 skip → 11 サイクル目で強制 PASS
- interval 10s 想定で約 100s 後に 1 回取引を強行

## テスト結果

- 596# 新規テスト: 8/8 passed
- 190# 既存テスト: 28/28 passed
- 193# 既存テスト: 23/23 passed
- 336# ドリフト防止: 4/4 passed
- 593# toxic skip: 12/12 passed
- v460 全体: 4079 passed, 9 skipped

## 残課題

1. **即時対応**: fill test プロセス (PID 3012) の hot_swap_restart が必要。安全弁コードは次回デプロイで有効化。
2. **`one_sided_balance` の配線**: CycleContext で `one_sided_balance=True` が設定されない (190# 設計のみ、実装漏れ)。今後の拡張で検討。
3. **Primary model の偏り調査**: buy 側 sg_score が恒常的に -2 ~ -3.8 と極端に負 → モデル再訓練 or 特徴量診断が必要。
