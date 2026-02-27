# 187# Chase 方向制御 + guard_trace 記録 + clamp YAML外部化

> **種別**: impl  
> **フェーズ**: ph2  
> **日付**: 2026-02-28  
> **前提**: 186# Phase A (ヒステリシス + clamp) 完了  
> **対応計画**: 186# Phase B + 追加改善

---

## 0. 実施内容サマリ

| # | 施策 | 根拠 | 状態 |
|---|------|------|------|
| **B-1** | Chase 方向制限 | 185#-§5.4, 178#-U2 | ✅ 完了 |
| **B-2** | guard_trace 全レコード記録 | 185#-所見6, 178#-U6 | ✅ 完了 |
| **追加** | Clamp 値 YAML 外部化 + hot-reload | 186# 186#-§5 A-2 発展 | ✅ 完了 |

---

## 1. B-1: Chase 方向制限

### 問題
現行 Chase は `is_drifting_away` (注文価格からの乖離) のみで方向判定。
trending_up で sell 注文を chase すると、マクロトレンドに逆行して逆選択リスクが増大。

### 実装

`CycleStrategy.is_chase_enabled()` に `side` パラメータを追加:

```python
def is_chase_enabled(self, regime: str | None, side: str | None = None) -> bool:
```

方向フィルタリングルール:
- `trending_up`: buy chase ✅ / sell chase ❌ (cancel-only)
- `trending_down`: sell chase ✅ / buy chase ❌ (cancel-only)
- `trending` (方向不明): 両方許可 (後方互換)
- `side=None`: フィルタ非適用 (後方互換)

### 変更ファイル
- `regime_policy.py`: Protocol 定義 + DefaultCycleStrategy 実装
- `fill_cycle_executor.py`: `_monitor_fill_polling()` 内の呼び出しに `side` 追加

---

## 2. B-2: guard_trace 記録

### 問題
`gated_regime()` によるヒステリシス判定後の実効 regime が FillRecord に記録されず、
事後分析でヒステリシスの効果を検証できない。

### 実装

`FillRecord` に 2 フィールドを追加:
- `gated_regime: Optional[str]` — ヒステリシス適用後の実効 regime
- `effective_cycle_interval: Optional[float]` — 使用されたサイクル間隔 (秒)

`fill_cycle_executor.py` の FillRecord 構築時に設定:
```python
gated_regime=self._cycle_strategy.gated_regime(regime_str, regime_conf),
effective_cycle_interval=self._cycle_strategy.effective_interval(regime_str),
```

### 後方互換性
- 既存 JSONL からの `from_dict()` は未知フィールドを無視 → `None` のまま
- `to_dict()` は `dataclasses.asdict()` で自動対応

---

## 3. 追加改善: Clamp YAML 外部化

### 問題
186# で導入した strictness clamp の上下限値 `[-0.3, 0.5]` がハードコードされており、
hot-reload でチューニングできない。

### 実装

`FillTestConfig` に新フィールド追加:
```python
skip_gate_offset_floor: float = -0.3   # 最大緩和
skip_gate_offset_ceil: float = 0.5     # 最大厳格化
```

- YAML `skip_gate.offset_floor` / `skip_gate.offset_ceil` でパース
- `_HOT_RELOADABLE_FIELDS` に追加 → プロセス再起動なしで変更可能
- `skip_gate_evaluator.py` はハードコード定数を `self._config` 参照に変更

---

## 4. 保守性改善

| 項目 | 対応 |
|------|------|
| `regime_policy.py` MAX LINES | 250 → 400 (186# ヒステリシス + 187# chase 方向) |
| `fill_cycle_executor.py` MAX LINES | 700 → 720 (187# guard_trace 8行追加) |
| `test_113` line count | 510 → 520 (guard_trace 分) |

---

## 5. テスト結果

### 新規テスト (22 件)
- `TestChaseDirectionControl` (9件): 方向別の chase 有効/無効
- `TestGuardTraceFillRecord` (8件): 新フィールドの存在/デフォルト/シリアライゼーション
- `TestClampYAMLExternalization` (4件): config フィールド/hot-reload/YAML パース
- `TestCycleStrategyProtocol` (1件): Protocol 互換性

### リグレッション
- **2373 passed, 0 failed** (全 v460 テスト)

---

## 6. 178# 未達事項の進捗

| # | 項目 | 186# | 187# |
|---|------|------|------|
| U1 | Trend Mode ヒステリシス | ✅ | — |
| U2 | Chase 方向制限 | — | ✅ **本セッション** |
| U3 | IOC 調査 | — | — (Phase D) |
| U4 | Buy model horizon | — | — (Phase C) |
| U5 | Strictness clamp | ✅ | ✅ YAML外部化 |
| U6 | guard_trace 記録 | — | ✅ **本セッション** |
