# Codex Task: 688# Timeout Side/Regime 別短縮 (638# P1)

## 目的
order timeout をマクロレジーム × side で動的に決定し、強トレンド時の逆選択コストを低減する。
現在は sell_timeout のみ個別設定だが、regime 条件による buy/sell 両側の timeout 最適化が未実装。

## 背景

### 638# で特定された課題
- strong_up 時の sell は AS 率が高い → timeout を短縮して被弾時間を減らすべき
- strong_down 時の buy は逆に timeout を短縮すべき
- 現状: `order_timeout_sec_sell` で sell だけ固定短縮、regime 別の分岐はマクロ sell timeout のみ

### 現状のアーキテクチャ (fill_cycle_executor.py L1137-1190)
- `order_timeout_sec=90.0` がベース
- `order_timeout_sec_sell` で sell 側のみオーバーライド
- `macro_sell_timeout_strong_up` / `macro_sell_timeout_weak_up` が既存 (sell×macro のみ)
- `micro_timeout_wait_sec` (requote 間隔) もマクロ aware

### 既存実装の確認ポイント
- `scripts/v460/lib/fill_cycle_executor.py` L1137-1190: timeout 決定ロジック
- `scripts/v460/lib/fill_config.py`: `FillTestConfig` の timeout 関連属性
- `configs/v460/fill_test.yaml`: `order_timeout_sec`, `order_timeout_sec_sell`, `macro_sell_timeout_*`
- `scripts/v460/lib/macro_regime.py`: `MacroTrend` enum (STRONG_UP, WEAK_UP, NEUTRAL, etc.)

## タスク

### Task 1: YAML 設定拡張

**対象**: `configs/v460/fill_test.yaml`

既存の `macro_sell_timeout_*` を一般化した regime×side timeout マトリクスを追加:
```yaml
# 既存 (変更なし、後方互換)
order_timeout_sec: 90
order_timeout_sec_sell: 45

# 新規: regime×side override (optional、未設定なら既存ロジック)
regime_timeout_overrides:
  strong_up:
    sell: 20        # 急騰中の sell は AS 被弾を最小化
    buy: 120        # 急騰中の buy は約定機会を延長
  strong_down:
    sell: 90        # 暴落中の sell は約定機会を延長
    buy: 30         # 暴落中の buy は AS 被弾を最小化
  # weak_up/weak_down/neutral: 未設定 → order_timeout_sec (/ _sell) にフォールバック
```

### Task 2: FillConfig 拡張

**対象**: `scripts/v460/lib/fill_config.py`

1. `regime_timeout_overrides: dict[str, dict[str, float]]` を追加 (default: `{}`)
2. YAML ローディングで型安全にパース
3. `get_timeout(side: str, regime: str) -> float` ヘルパーメソッド追加:
   - `regime_timeout_overrides[regime][side]` → あれば使用
   - なければ既存の `order_timeout_sec_sell` / `order_timeout_sec` にフォールバック

### Task 3: fill_cycle_executor.py のロジック統合

**対象**: `scripts/v460/lib/fill_cycle_executor.py` L1137-1190

1. timeout 決定箇所で `config.get_timeout(side, macro_trend)` を呼ぶ
2. 既存の `macro_sell_timeout_strong_up` / `macro_sell_timeout_weak_up` ロジックは regime_timeout_overrides が空の場合のフォールバックとして残す
3. ログに選択理由を記録: `"timeout={value}s (regime={regime}, side={side})"`

### Task 4: FillRecord 記録

**対象**: `scripts/v460/lib/fill_record_builder.py`, `ztb/metrics/fill_quality.py`

1. `timeout_applied_sec: float | None` フィールド追加
2. `timeout_reason: str | None` フィールド追加 (例: "regime_strong_up_sell")

### Task 5: テスト

**対象**: `tests/unit/v460/`

1. `regime_timeout_overrides` 設定あり → strong_up/sell で短縮 timeout が使われる
2. `regime_timeout_overrides` 設定なし → 既存の `macro_sell_timeout_*` フォールバック
3. neutral regime → base timeout にフォールバック
4. buy 側 regime override の動作確認
5. FillRecord に timeout_applied_sec が記録される
6. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 受け入れ基準

- [ ] regime×side でタイムアウトが動的に決定される
- [ ] 既存の `macro_sell_timeout_*` は後方互換で動作
- [ ] FillRecord に `timeout_applied_sec`, `timeout_reason` が記録
- [ ] 新規テスト 5 件以上、全テスト pass
- [ ] YAML hot-reload で regime_timeout_overrides が反映される

## リスク評価

- **低リスク**: timeout はサイクル開始時に1回だけ参照。副作用なし
- **ロールバック**: `regime_timeout_overrides: {}` で旧動作に完全復帰
- **検証**: FillRecord の timeout_applied_sec で A/B 比較可能
