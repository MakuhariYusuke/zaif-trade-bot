# 231# FFD ロジック強化 + import_state None 安全

> **種別**: fix (230# セルフレビュー指摘対応)
> **日付**: 2026-03-03
> **コミット**: `23ebd2eb7`
> **テスト**: `test_230_ffd_deadzone_streak_guards.py` — 60 テスト (230# 54 + 231# 6 追加)

---

## 概要

230# で実装した FFD deadzone/streak ロジックに対する破壊的セルフレビューにより
発見された HIGH 3 件 + MEDIUM 3 件を修正。外部 AI レビューで指摘されうる問題を
先回りして解消し、本番投入前のロバスト性を確保。

## 変更一覧

### R1 (HIGH): TTL 期限切れ時の streak 未リセット

**問題**: `get_boost_multiplier()` の TTL expired 分岐で `boost_active` と
`boost_multiplier` はリセットされるが `normal_fill_streak` はそのまま。
`export_state()` → JSON → `import_state()` 経由で stale な streak 値が
永続化されるリスク。

**修正**: TTL expired 分岐に `state.normal_fill_streak = 0` を追加。

**ファイル**: `fast_fill_defense.py` L111

```python
# Before:
state.boost_active = False
state.boost_multiplier = 1.0
# state.normal_fill_streak は stale のまま

# After:
state.boost_active = False
state.boost_multiplier = 1.0
state.normal_fill_streak = 0  # 231# R1
```

### R2 (HIGH): Slow fill + negative PnL が streak にカウント

**問題**: `elif state.boost_active:` 分岐で `is_fast=False` かつ
`has_negative_edge=True` のケースでも `normal_fill_streak += 1` されていた。
つまり情報トレーダーが速度を落として (slow fill) 攻撃を継続しても
「正常」としてカウントされ、boost が早期解除される。

**市場理論根拠**: Kyle (1985) — 情報トレーダーは検出回避のため注文ペースを
調整する。fill 速度だけでなく PnL 方向が adverse かを判定すべき。

**修正**: `elif state.boost_active:` 分岐で `has_negative_edge` を先にチェック。
True なら streak=0 にリセット、False の場合のみインクリメント。

**ファイル**: `fast_fill_defense.py` L240-244

```python
# Before:
elif state.boost_active:
    state.normal_fill_streak += 1  # 常にインクリメント

# After:
elif state.boost_active:
    if has_negative_edge:
        state.normal_fill_streak = 0  # adverse → streak リセット
    else:
        state.normal_fill_streak += 1
```

### R3 (HIGH): Adverse fill 継続時の TTL 非リフレッシュ

**問題**: `boost_activated_at = time.time()` が `if not state.boost_active:`
(初回起動) ブロック内にのみ配置。2回目以降の adverse fill で TTL がリフレッシュ
されず、600s 後に攻撃中でも防御が自動解除される窓が発生。

**市場理論根拠**: 情報トレーダーの攻撃は断続的に数十分にわたることがある。
防御 TTL は攻撃の「最終検出時刻」を起点にすべき。

**修正**: `state.boost_activated_at = time.time()` を `if not state.boost_active:`
ブロックの外側に移動。新規・継続に関わらず全ての adverse fast fill で リフレッシュ。

**ファイル**: `fast_fill_defense.py` L234-235

```python
# Before:
if is_fast and has_negative_edge:
    if not state.boost_active:
        state.boost_active = True
        state.boost_activated_at = time.time()  # ← 初回のみ
        ...

# After:
if is_fast and has_negative_edge:
    if not state.boost_active:
        state.boost_active = True
        ...
    state.boost_activated_at = time.time()  # ← 毎回リフレッシュ
    state.normal_fill_streak = 0
```

### R4 (MEDIUM): import_state で JSON null → TypeError

**問題**: `state.get("key", default)` パターンでは、JSON の `null` が
Python の `None` としてデシリアライズされた場合、キーが存在するため
`.get()` は `None` を返す (デフォルト値は使われない)。
結果: `int(None)` → `TypeError` でクラッシュ。

**修正**: 全 8 フィールドを `state.get("key") or default` パターンに変更。

**ファイル**: `fast_fill_defense.py` L299-306

```python
# Before:
self._state_buy.boost_multiplier = float(state.get("buy_boost_multiplier", 1.0))

# After:
self._state_buy.boost_multiplier = float(state.get("buy_boost_multiplier") or 1.0)
```

**注意点**: `or default` パターンは `0` や `False` も falsy とみなす。
本モジュールでは:
- `boost_active=False` → `or False` で正しい (False が期待値)
- `boost_multiplier=0.0` → `or 1.0` で `1.0` になる (0 は無効値なので正しい)
- `boost_activated_at=0.0` → `or 0.0` で `0.0` のまま (0 → falsy → `or 0.0` = `0.0`)
- `normal_fill_streak=0` → `or 0` で `0` のまま (0 → falsy → `or 0` = `0`)

全フィールドで意味的に正しい挙動であることを確認済み。

### R5 (MEDIUM): Config バリデーション上限なし

**問題**: `ffd_l2_deadzone_bps >= 0` / `ffd_boost_release_streak >= 1` のみで
上限チェックがない。`deadzone=99999` のようなサイレント無効化を許容。

**修正**: 上限追加。

**ファイル**: `fill_config.py`

```python
# Before:
if self.ffd_l2_deadzone_bps < 0:
    raise ValueError(...)
if self.ffd_boost_release_streak < 1:
    raise ValueError(...)

# After:
if not (0.0 <= self.ffd_l2_deadzone_bps <= 100.0):
    raise ValueError(...)
if not (1 <= self.ffd_boost_release_streak <= 20):
    raise ValueError(...)
```

**根拠**:
- `l2_deadzone_bps ≤ 100`: BTC/JPY の typical bid-ask spread は 5-15 bps。
  100 bps (= 1%) を超える deadzone は事実上 L2 を無効化する。
- `boost_release_streak ≤ 20`: 約定間隔 30-120s × 20 = 10-40 分。
  これ以上の streak は合理的な攻撃持続時間を超える。

### R8 (MEDIUM): L1+L2 同時発火ログが L1 のみ

**問題**: `layer_info = "L1" if has_negative_edge_l1 else "L2(pnl)"` では
L1 と L2 が同時に検出されたケースで常に "L1" と表示。

**修正**: 3 分岐のログラベルに変更。

**ファイル**: `fast_fill_defense.py` L225-228

```python
_layer = (
    "L1+L2" if (has_negative_edge_l1 and has_negative_edge_l2)
    else ("L1" if has_negative_edge_l1 else "L2(pnl)")
)
```

## テスト

### 新規追加 (6 テスト)

| テスト名 | 対象 |
|---|---|
| `test_slow_adverse_pnl_resets_streak` | R2: slow fill + negative PnL → streak=0 |
| `test_ttl_expiry_resets_streak` | R1: TTL 期限切れ → streak=0 |
| `test_adverse_refreshes_ttl` | R3: 再 adverse → `boost_activated_at` 更新 |
| `test_import_none_value_streak` | R4: JSON null 全フィールド → デフォルト復元 |
| `test_ffd_l2_deadzone_bps_over_100_raises` | R5: deadzone > 100 → ValueError |
| `test_ffd_boost_release_streak_over_20_raises` | R5: streak > 20 → ValueError |

### テスト結果

```
test_230_ffd_deadzone_streak_guards.py — 60 passed
test_100_fast_fill_defense.py — 14 passed
v460 全体 — 3154 passed, 0 failed
```

## 未対応 (LOW — リファクタリングスコープ)

| # | 内容 | 理由 |
|---|------|------|
| R6 | `get_boost_multiplier` の getter 副作用 (TTL check) | 現行で実害なし。分離は API 変更を伴う |
| R7 | `_cycle_strategy is not None` dead branch | 常に初期化されるが防御的チェックとして無害 |
| R9/10 | MCB/SAD テストがソース文字列ベースのみ | 統合テストは別スコープ |
| R11 | `export_state` の return type `dict[str, object]` | TypedDict 化は互換性リスク |
| R12 | `max(1, boost_release_streak)` が冗長 | config validation で保証済み、防御的冗長 |
| R14 | test_100 docstring に streak=1 説明なし | ドキュメント品質のみ |
| R15 | `boost_ttl_sec` が YAML/Config 非公開 | 600s デフォルトで実用上問題なし |

## 状態遷移図

```
                          ┌─────────────────────────────┐
                          │                             │
                          ▼                             │
    ┌──────────┐   fast+adverse   ┌──────────────┐      │
    │ INACTIVE │ ────────────────→│   ACTIVE     │──────┘
    │ mult=1.0 │                  │ mult=N.N     │  fast+adverse
    │ streak=0 │                  │ streak=0     │  (TTL refresh)
    └──────────┘                  │ ttl=time()   │
         ▲                        └──────┬───────┘
         │                               │
         │                    ┌──────────┼──────────┐
         │                    │          │          │
         │              slow+adverse  normal    TTL expire
         │              streak=0    streak++   streak=0
         │                    │          │     mult=1.0
         │                    │          │          │
         │                    ▼          ▼          │
         │              ┌────────┐  streak≥N?      │
         │              │ ACTIVE │  ───Yes──→───────┤
         │              │ (stay) │                  │
         │              └────────┘                  │
         │                                          │
         │◄─────────── deactivate ◄─────────────────┘
         │              (streak≥N or TTL or unfilled)
         │
    ┌────┴─────┐
    │ unfilled │ → reset (mult=1.0, streak=0)
    └──────────┘
```

## evaluate_fill 決定表

| is_fast | has_negative_edge | boost_active | アクション |
|---------|-------------------|--------------|------------|
| T | T | F | **activate**: mult=boost, streak=0, ttl=now |
| T | T | T | **refresh**: ttl=now, streak=0 |
| T | F | F | (no-op) |
| T | F | T | streak++, check deactivate |
| F | T | F | (no-op) |
| F | T | T | streak=0 (adverse but slow) |
| F | F | F | (no-op) |
| F | F | T | streak++, check deactivate |
