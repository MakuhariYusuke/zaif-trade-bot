# 230# FFD deadzone/streak + MCB/SAD guard + hasattr 排除

> **種別**: fix (pre-deployment comprehensive bug hunt)
> **日付**: 2026-03-04
> **テスト**: `test_230_ffd_deadzone_streak_guards.py` — 34 テスト

---

## 概要

本番投入前の包括的セルフレビューにより特定された高優先度バグ (H-1〜H-4) と
中優先度コード品質問題 (M-1) を一括修正。市場理論に基づく FFD 改善 (AS理論:
H-1, Kyle 1985: H-2) により、false positive 防御の削減と true positive
検出の精度向上を実現。

## 変更一覧

### H-1: FFD Layer 2 deadzone (AS理論)

**問題**: `post_fill_pnl_bps < 0` の判定で、正常なスプレッドコスト (~2-3 bps)
が Layer 2 adverse selection として誤検知 → 不要な offset 拡大 → fill rate
低下 → 収益機会損失。

**修正**: `pnl < -l2_deadzone_bps` に変更。デフォルト `3.0 bps`。マーケット
メイカーのスプレッドコストは取引の正常コストであり、これを超える損失のみが
真の adverse selection を示す。

**ファイル**: `fast_fill_defense.py`, `fill_config.py`, `fill_test.yaml`

### H-2: FFD boost gradual release (Kyle 1985)

**問題**: 1回の正常約定で即座にブースト解除。Kyle (1985) の情報漸次伝播
モデルでは、情報トレーダーは複数の連続した fills にわたって取引するため、
1回の正常 fill で安全宣言するのは尚早。

**修正**: `boost_release_streak` パラメータ (デフォルト 3) を追加。N 回
連続で正常 fill が観測されるまでブースト維持。途中で再度 adverse fill が
検出されたら streak リセット。

**ファイル**: `fast_fill_defense.py`, `fill_config.py`, `fill_test.yaml`

### H-3: MCB/SAD None guard

**問題**: `self._mcb.config.enabled` / `self._sad.config.enabled` が
`_mcb`/`_sad` が `None` の場合に `AttributeError` を発生。4箇所。

**修正**: `self._mcb is not None and self._mcb.config.enabled` パターンで
全4箇所をガード。

**ファイル**: `fill_loop_orchestrator.py`

### H-4: regime_detector hasattr→init

**問題**: `_last_result` と `_last_velocity_pct` が `__init__` で宣言されず、
`hasattr`/`getattr` でアクセス。型安全低下。

**修正**: `__init__` で明示的に `None`/`0.0` 初期化。`hasattr` → `is not None`,
`getattr(self, "_last_velocity_pct", 0.0)` → `self._last_velocity_pct` に変換。

**ファイル**: `regime_detector.py`

### M-1: fill_cycle_executor hasattr 排除

**問題**: 10箇所の `hasattr(self, ...)` がプロパティアクセスの代替として使用。

**修正**: 8箇所を `is not None` に変換。2箇所は mixin method 存在確認
(`_current_regime_value`) として正当なため据え置き。

**ファイル**: `fill_cycle_executor.py`

### 追加: orchestrator diff 修正

ツール操作中に orchestrator の SAD halt 処理ブロック本体と
`await self._effective_sleep(multiplier=5.0)` が脱落、MCB main-loop の
`update()`/`check()` 呼び出しが消失するデグレーションを検出・修正。

## 設定

```yaml
fast_fill_defense:
  l2_deadzone_bps: 3.0       # Layer 2 deadzone (bps)
  boost_release_streak: 3     # 解除に必要な連続正常 fill 数
```

## テスト

| クラス | テスト数 | 対象 |
|---|---|---|
| TestFFDL2Deadzone | 6 | H-1: deadzone 境界テスト |
| TestFFDBoostGradualRelease | 8 | H-2: streak logic |
| TestFFDStreakStatePersistence | 3 | H-2: export/import |
| TestMCBSADNoneGuard | 4 | H-3: None guard source check |
| TestRegimeDetectorInit | 5 | H-4: init + properties |
| TestFillCycleExecutorHasattr | 4 | M-1: hasattr 排除確認 |
| TestConfigValidation230 | 6 | Config バリデーション |
| TestFFDConfigDefaults | 3 | FFDConfig デフォルト値 |
| **合計** | **39** | |

## 231# Self-review 指摘修正 (same commit)

| # | 重大度 | 内容 |
|---|--------|------|
| R1 | HIGH | TTL 期限切れ時に streak 未リセット → stale 値永続化リスク |
| R2 | HIGH | Slow fill + negative PnL が streak インクリメント → adverse 中に boost 早期解除 |
| R3 | HIGH | Adverse fill 継続時に `boost_activated_at` 非更新 → TTL 窓内で防御解除 |
| R4 | MEDIUM | `import_state` で JSON null → `int(None)` TypeError |
| R5 | MEDIUM | Config バリデーションに上限なし → サイレント無効化 |
| R8 | MEDIUM | L1+L2 同時発火時のログが L1 のみ |

### R2 詳細: Kyle 1985 の正しい適用

元の実装では `is_fast` が `False` でも `elif state.boost_active:` 分岐で
`normal_fill_streak += 1` されていた。つまり slow fill で PnL が -20bps
（L2 deadzone 超え）であっても「正常」と数えられ、boost 解除が加速していた。

修正後:
```python
elif state.boost_active:
    if has_negative_edge:
        state.normal_fill_streak = 0  # adverse PnL → streak リセット
    else:
        state.normal_fill_streak += 1
```

### R3 詳細: TTL リフレッシュ

情報トレーダーが断続的に攻撃を続ける場合、元の TTL は初回起点
からのみ計時され、600s 後に攻撃中でも防御が切れる窓が生まれた。
修正後は adverse fill のたびに `boost_activated_at = time.time()`
がリフレッシュされ、攻撃が続く限り防御も続く。

## 既存テスト修正

- `test_100_fast_fill_defense.py`:
  - `TestTwoLayerNegativeEdge.test_layer2_post_fill_pnl_negative`: deadzone=2.0 を指定 (H-1 対応)
  - `TestBoostDeactivation.test_normal_fill_deactivates`: streak=1 を指定 (H-2 対応)
