# 453# Micro-timeouts (TIF Emulation) 実装結果

**種別**: result  
**日付**: 2026-03-16  
**関連**: 452# (設計), 447# (提案B), 450#  
**ステータス**: 実装完了 (disabled by default)

---

## 1. 概要

452# で設計された **Micro-timeouts (TIF Emulation)** のサブサイクル再クオート方式を実装した。
従来の `order_timeout_sec=90s` 単一監視ではなく、15秒間隔でキャンセル→最新 mid で再発注を繰り返す。

**目的**: Adverse Selection リスク低減 — HFT/大口が長時間固定板をターゲットにする問題への構造的防御。

## 2. 実装内容

### 2.1 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/fill_config.py` | `micro_timeout_*` config フィールド 6 件追加 |
| `scripts/v460/lib/fill_config_parser.py` | `micro_timeout:` YAML セクション解析追加 |
| `scripts/v460/lib/fill_config_results.py` | `FillMonitorResult` に `requote_attempts`, `partial_filled_qty` 追加 |
| `scripts/v460/lib/fill_cycle_executor.py` | `run_single_cycle()` にサブサイクルループ実装 |
| `ztb/metrics/fill_quality.py` | `FillRecord` に `requote_attempts`, `micro_timeout_partial_filled_qty` 追加 |
| `configs/v460/fill_test.yaml` | `micro_timeout:` セクション追加 (disabled) |
| `tests/unit/v460/test_micro_timeout.py` | 単体テスト 12 件 (全 PASSED) |

### 2.2 新しい設定項目

```yaml
micro_timeout:
  enabled: false                   # サブサイクル re-quote の有効/無効
  wait_sec: 15.0                   # 1 回あたりの最大配置時間 (秒)
  wait_sec_sell: 10.0              # sell 側の配置時間 (既存 sell timeout 優遇を踏襲)
  max_requote_per_cycle: 4         # 1 サイクル内の最大 re-quote 回数
  requote_cooloff_sec: 5.0         # キャンセル→再発注の冷却期間 (秒)
  cancel_on_cross_venue_flip: true # Cross-Venue 反転時の即キャンセル (将来拡張用)
```

### 2.3 アーキテクチャ

452# で推奨された **案2 (サブサイクル・ポーリング型)** を採用。

```
┌─────────────────────────────────────────────────────────┐
│  Policy Phase (1サイクル1回)                              │
│  features → price → offset → lot → skip gate            │
├─────────────────────────────────────────────────────────┤
│  Execution Sub-cycle Phase (最大 N 回ループ)              │
│  ┌───────────────────────────────────────────────┐      │
│  │ Attempt 1: place_order → monitor(15s) → ✗     │      │
│  │   ↓ cancel + cooloff(5s)                      │      │
│  │ Attempt 2: fetch_mid → reprice → monitor(15s) │      │
│  │   ↓ cancel + cooloff(5s)                      │      │
│  │ Attempt 3: fetch_mid → reprice → monitor(15s) │      │
│  │   ↓ ... (最大 max_requote_per_cycle 回)        │      │
│  └───────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────┤
│  Post-fill Phase (PnL 計測、レジーム更新、FillRecord)      │
└─────────────────────────────────────────────────────────┘
```

**設計判断**:
- **Policy/Execution 分離**: 特徴量抽出・offset 計算は 1 回のみ。re-quote は mid 基準を更新するだけ。
- **OrderMonitor の timeout 一時差替**: `object.__setattr__` で `order_timeout_sec` を micro_timeout_wait_sec に一時差替え → 復元。OrderMonitor 内部ロジックの改修は最小限。
- **整数丸め**: JPY ペアのため `round(order_price)` で tick 丸め。
- **デフォルト無効**: 本番稼働中のシステムに影響なし。有効化は YAML の `enabled: true` のみ。

### 2.4 FillRecord 記録

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `requote_attempts` | `int \| None` | re-quote 回数 (0=初回で約定, None=micro_timeout 無効) |
| `micro_timeout_partial_filled_qty` | `float \| None` | re-quote ループ中の部分約定合計 |

## 3. テスト結果

```
21 passed in 23.26s
```

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| `TestMicroTimeoutConfigDefaults` | 2 | config デフォルト値テスト |
| `TestMicroTimeoutYamlParsing` | 4 | YAML パース (有効/無効/なし/部分) |
| `TestFillMonitorResultRequoteFields` | 2 | FillMonitorResult 新フィールド |
| `TestFillRecordRequoteFields` | 2 | FillRecord 新フィールド + to_dict |
| `TestProductionYamlMicroTimeout` | 2 | 本番 YAML セクション存在確認 |
| `TestMicroTimeoutValidation` | 9 | 値域バリデーション + 構造的整合性警告 |

既存テスト回帰なし: `test_fill_test_config.py` 83 テスト全 PASSED。

## 4. 有効化手順

```yaml
# configs/v460/fill_test.yaml の micro_timeout セクションを変更:
micro_timeout:
  enabled: true    # ← false → true
```

hot-reload 対応: YAML 変更 → 次サイクルで自動反映。

### 推奨段階的ロールアウト

1. **Step 1**: `enabled: true`, `wait_sec: 30.0`, `max_requote_per_cycle: 2` (保守的)
2. **Step 2**: `wait_sec: 15.0`, `max_requote_per_cycle: 3` (標準)
3. **Step 3**: `wait_sec: 15.0`, `max_requote_per_cycle: 4`, `wait_sec_sell: 10.0` (攻撃的)

各ステップで fill_rate、AS 率、requote_attempts の分布を監視。

## 5. 将来拡張 (452# 残件)

- **`cancel_on_cross_venue_flip`**: config フィールドは追加済みだが、Cross-Venue シグナル監視ロジックは未実装 (449# cross-venue confidence と統合予定)。
- **部分約定 (Partial Fill) 再発注**: 現実装ではフル cancel → フル re-quote。部分約定の残量減算ロジックは基盤のみ (`partial_filled_qty` フィールド)。API の partial fill 検出が Coincheck 側で安定確認できた時点で拡張。
- **stale order reprice との競合**: micro_timeout 有効時に stale_order_enabled=true だと二重制御になる。将来は micro_timeout 有効時に stale_order を自動無効化するか、相互排他にする検討が必要。

## 6. リスク評価

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| API Rate Limit | re-quote が 4 回 × cancel/place = 8 API calls 追加 | cooloff_sec=5.0 + max_requote=4 で 120s 内に収まる |
| 二重発注 | cancel 失敗時の phantom 注文 | 既存の phantom guard (237#) が検出・照合 |
| Offset stale | re-quote 時は mid のみ更新、offset pipeline 再計算なし | Offset は Policy Phase で計算済み。15s 程度なら stale にならない |
| 既存挙動への影響 | 無効化時 (デフォルト) は一切のコードパスが変わらない | if-else 分岐で従来パスを完全に保持 |

## 7. セルフレビュー (453# review)

### 7.1 修正事項

| # | 種別 | 内容 | 修正 |
|---|------|------|------|
| R1 | BUG | `cancel_failed_likely_filled` が `fill_price is not None` 内にネストされており、fill_price=None の場合に到達不能 | ネストを解除、独立した if 文に変更 |
| R2 | BUG | `cancel_reason` が micro-timeout の短 timeout by design の cancel でも `"timeout"` のまま。通常 timeout と区別不能 | 未約定 + micro_timeout 有効時は `"micro_timeout"` に上書き |
| R3 | BUG | `queue_wait` が re-quote 後の `t_submit` 基準 (最後の注文の待ち時間) になる。初回発注→最終約定の通算が正しい | `_first_t_submit` を保持し、約定時は `time.time() - _first_t_submit` を使用 |
| R4 | MISSING | `micro_timeout_*` フィールドにバリデーションなし。無効値 (wait=0, max_requote=0, 負値) が素通り | `fill_config_validation.py` に 5 件の値域チェック追加 |
| R5 | MISSING | サブサイクル合計時間が `cycle_interval_sec` を超過する設定への警告なし | 合計 > cycle_interval_sec 時に `warnings.warn` |
| R6 | MISSING | `stale_order_enabled` 同時有効時の二重価格制御の警告なし | `stale_order_enabled=True` + `micro_timeout_enabled=True` 時に `warnings.warn` |
| R7 | DEAD_CODE | `_remaining_lot`, `_micro_partial_qty` が初期化のみで更新なし | 将来の Partial Fill 対応基盤として意図的に残置 (453# §5 で記載済み) |

### 7.2 追加バリデーションルール (fill_config_validation.py)

```python
# 452# Micro-timeout バリデーション
micro_timeout_wait_sec > 0
micro_timeout_wait_sec_sell > 0 or None
micro_timeout_max_requote >= 1
micro_timeout_requote_cooloff_sec >= 0
# 構造的整合性警告
max_requote * (wait_sec + cooloff) <= cycle_interval_sec  # 超過時 warn
stale_order_enabled + micro_timeout_enabled 同時有効 → warn
```

### 7.3 テスト追加

| テストクラス | テスト数 | 内容 |
|-------------|---------|------|
| `TestMicroTimeoutValidation` | 9 | 値域バリデーション (negative/zero/warn 等) |

合計: **21 passed** (12 既存 + 9 新規)

### 7.4 残存リスク (許容判断)

| リスク | 判断 | 理由 |
|--------|------|------|
| `_remaining_lot` 未更新 | 許容 | Partial Fill API が Coincheck 側で未安定。基盤変数のみ先行配置 |
| `object.__setattr__` config mutation | 許容 | シングルスレッド async。try/finally で復元保証。代替は Context Manager 化 (over-engineering) |
| stale_order との競合 | 許容 (警告追加) | デフォルト `stale_order_enabled=False`。同時有効は `warnings.warn` で検出可能 |
