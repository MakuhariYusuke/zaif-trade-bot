# 110# 086# time_filter デッドロック修正 — 49% アイドル解消

| key | value |
|-----|-------|
| type | fix |
| phase | ph3 (fill_test 実戦改善) |
| status | committed |
| parent | 086#, 107# |
| tests | 835 passed (v460 unit tests, +8 new) |

---

## §1 背景

fill_test 7 日間連続稼働ログ分析 (02/13–02/19) で、**稼働時間の 49% がアイドル状態**という重大なバグを発見。

### 発見経緯

| ステップ | 内容 |
|-----------|------|
| ログ末尾確認 | `[time_filter] ... treating as both-filtered (086# 片側蓄積防止)` が 2 分間隔で反復 |
| 全日集計 | 527 回の both-filtered 反復 → 約 12.1 時間のアイドル (24.4 時間中) |
| 時間帯分析 | 特定 UTC 時間帯で 50–60 分間の完全停止パターン |

### 日別パフォーマンス (最新)

| 日付 | cycles | filled | skip | fill率 | AS% | 30s PnL mean | sum |
|------|--------|--------|------|--------|-----|-------------|-----|
| 02/13 | 211 | 163 | 0 | 77% | 48% | -0.44bps | -71.8 |
| 02/14 | 220 | 161 | 0 | 73% | 31% | -0.72bps | -116.5 |
| 02/15 | 60 | 49 | 0 | 82% | 35% | -0.88bps | -42.9 |
| 02/16 | 21 | 14 | 0 | 67% | 36% | -1.12bps | -15.7 |
| 02/17 | 205 | 137 | 24 | 76% | 28% | **+0.45** | **+61.5** |
| 02/18 | 277 | 149 | 53 | 67% | 19% | **+0.35** | **+52.6** |
| 02/19 | 16 | 10 | 0 | 62% | 0% | +0.47 | +4.7 |

### 追加パターン分析

| 分析軸 | 結果 |
|---------|------|
| side別 PnL | buy: +103.0bps (n=346), **sell: -231.2bps (n=337)** |
| 待機時間別 | <10s: -0.09bps, 10-30s: -0.53bps, **30-60s: +0.02bps**, 60-120s: +0.05bps |
| AS率推移 | 02/13: 48% → 02/18: 19% (107# skip_gate + volatility_guard 効果) |

## §2 根本原因

### デッドロックメカニズム

```
_last_side = "sell" (前回の取引)
  ↓
_next_side() → "buy" (交互ロジック)
  ↓
_is_time_filtered("buy") → True (例: UTC 1h)
  ↓
alt_side = "sell"
_is_time_filtered("sell") → False
  ↓
086# チェック: alt_side("sell") == _last_side("sell")
  → 両方ブロック扱い → 120s スリープ
  ↓
_last_side は変更されない (取引未実行)
  → 次ループも同じ判定 → 無限ループ (UTC時間変更まで最大60分)
```

### 影響範囲

| 指標 | 値 |
|------|-----|
| 総 both-filtered 回数 | 527 回 |
| アイドル時間 | 12.1 時間 / 24.4 時間 (**49%**) |
| 理論的影響 UTC 時間 | 1,2,4,8,12,13,14,16,18,21 (10h/24h) |
| 損失機会 | ~500 サイクル分の取引機会喪失 |

### デッドロック発生パターン (日別・時間帯別)

| 日付 | UTC 時間帯 | ブロック時間 |
|------|-----------|-------------|
| 02/17 | 11-13h, 21-23h | 各 40-62 分 |
| 02/18 | 2-3h, 6h, 10-11h, 13h, 21-23h | 各 50-60 分 |
| 02/19 | 3h, 6h, 10-11h (進行中) | 各 ~60 分 |

## §3 修正内容

### 設計方針

086# の片側蓄積防止は正当な意図があるため、完全廃止ではなく**上限付き待機**を導入:

- 短期 (最初 3 サイクル = ~6 分): 片側蓄積を防止 (086# 元来の意図を保持)
- 長期 (4 サイクル目以降): デッドロック解除、alt_side を許可

### 変更箇所

#### 1. `FillTestConfig` に新パラメータ追加

```python
# 110# 086# デッドロック修正: 連続 both-filtered 上限
max_086_consecutive_wait: int = 3  # 0=無制限(旧動作), >0でN回超過後alt_side許可
```

#### 2. `FillTestRunner.__init__` にカウンタ追加

```python
self._consecutive_086_wait: int = 0
```

#### 3. メインループ 086# ブロック修正 (L2200-2235)

```
旧: alt_side == _last_side → 常に both-filtered 扱い → 無限スリープ
新: alt_side == _last_side → カウンタ増加
    → max_wait 超過時: alt_side 許可 (デッドロック解除)
    → max_wait 以下時: 従来通りスリープ (086# 保持)
```

#### 4. YAML パーサー (`from_yaml`)

```python
if "max_086_consecutive_wait" in tf:
    kwargs["max_086_consecutive_wait"] = tf["max_086_consecutive_wait"]
```

#### 5. time_filter 離脱時にカウンタリセット

```python
if self._in_time_filter:
    self._in_time_filter = False
    self._consecutive_086_wait = 0  # 110# カウンタリセット
```

#### 6. YAML 設定 (`configs/v460/fill_test.yaml`)

```yaml
max_086_consecutive_wait: 3  # 3回×120s=6分待機後に解除
```

## §4 期待効果

| 指標 | Before | After (予測) |
|------|--------|-------------|
| アイドル率 | 49% | ~8% (genuine both-filtered のみ) |
| 有効取引時間 | 12.3h/24h | ~22h/24h |
| 取引サイクル数 | ~200/日 | ~370/日 (理論上 +85%) |
| 連続同 side リスク | なし (完全遮断) | 6 分遅延後のペア不均衡 (次サイクルで反転) |

### リスク評価

- **片側蓄積**: 最大 1 回の連続同 side → 次サイクルの `_next_side()` で反転される
- **AS リスク**: time_filter 下で alt_side を実行するが、そもそも alt_side は time_filter non-blocked
- **設定安全弁**: `max_086_consecutive_wait: 0` で旧動作に即座に戻せる

## §5 テスト

| テスト | 内容 | 結果 |
|--------|------|------|
| test_config_default_max_086 | デフォルト値 = 3 | ✅ |
| test_config_from_yaml_max_086 | YAML 読み込み | ✅ |
| test_config_from_yaml_no_086_uses_default | 未指定時デフォルト | ✅ |
| test_runner_has_consecutive_086_counter | カウンタ初期化 | ✅ |
| test_consecutive_086_wait_zero_means_unlimited | 0 = 無制限 | ✅ |
| test_deadlock_break_logic_in_source | ソースコード検証 | ✅ |
| test_is_time_filtered_unchanged | 非干渉確認 | ✅ |
| test_yaml_roundtrip_max_086 | YAML ラウンドトリップ | ✅ |
| 全 v460 テスト | 835 passed | ✅ リグレッションなし |

## §6 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `scripts/v460/run_fill_test.py` | Config 新パラメータ、カウンタ、デッドロック解除ロジック、YAML パーサー |
| `configs/v460/fill_test.yaml` | `max_086_consecutive_wait: 3` 追加 |
| `tests/unit/v460/test_fill_test_config.py` | `Test110DeadlockBreak` 8 テスト追加 |
| `docs/v460/110_086_time_filter_deadlock_fix.md` | 本ドキュメント |
