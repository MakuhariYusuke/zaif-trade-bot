# 349# EWMA バグ修正 — コードレビュー依頼

## レビュー目的

DynamicKillManager (売買の動的キル判定) の EWMA (指数加重移動平均) に関する
3 つのバグを修正した。本レビューでは **正確性・エッジケース・設計判断** を検証する。

## 背景

zaif-trade-bot は暗号通貨の自動マーケットメイキングボットで、
`DynamicKillManager` が各サイド (sell/buy) の直近約定 PnL を EWMA で追跡し、
パフォーマンス悪化時に自動的に売買を一時停止 (kill) する機構を持つ。

### 問題の発生経緯

再起動後に sell EWMA が -10.710bps に毒化 (threshold は -0.5bps) し、
TIME LIMIT (30分) → 解除 → 即再 kill → TIME LIMIT ... の無限ループが 24 時間継続。
結果、5 fills / -18.54bps / WR 20% という壊滅的パフォーマンスに。

### 修正後の結果

41 fills / -7.08bps / WR 46.3% / avg PnL -0.17bps/fill に改善。
EWMA rebuild ログ、TIME LIMIT decay ログも正常動作を確認。

## 修正内容 (3件)

### P0 (Critical): EWMA 状態永続化

`export_state()` / `import_state()` に `_ewma_value` が含まれていなかった。
再起動時に EWMA が `None` にリセットされ、最初の fill の PnL がそのまま seed に。

**修正:**
- `export_state()` に `ewma_value` フィールドを追加
- `import_state()` で `ewma_value` を復元、欠落時は `_rebuild_ewma_from_history()` で再構築
- `reset()` に `self._ewma_value = None` を追加

### P1: EWMA シード安定化

初回 EWMA シードが単一観測値 (`pnl_bps`) で行われていた。
α=0.05 の場合、外れ値が ~45 fills まで支配的に残留し、実質回復不能。

**修正:**
- `track()` の初回 EWMA 初期化時、`pnl_history` に複数データがあれば算術平均でシード
- 単一データの場合は従来通りその値を使用

### P2: TIME LIMIT 解除時の EWMA decay

TIME LIMIT 解除が EWMA をリセットしないため、解除後も即座に再 kill が発生。

**修正:**
- TIME LIMIT 解除時に EWMA を `threshold * 0.8` にリセット
- レジーム別閾値がある場合はそれを使用
- `ewma_alpha = 0` (EWMA 無効) の場合はスキップ

### 横展開: orchestrator warmup 非対称バグ

`_warmup_kill_managers_from_records()` の発火条件が sell 側のみチェックしていた。

**修正:**
- sell/buy 両方の `pnl_history` を OR でチェック
- 既に `pnl_history` がある側は warmup をスキップ (二重 track 防止)
- restore ログに `ewma=` 値を追加

## レビュー観点

以下の点について批判的にレビューしてください:

### 1. 正確性
- `_rebuild_ewma_from_history()` の再構築ロジックは正しいか？ 先頭の平均シード → 全要素 replay は EWMA の数学的定義と合致するか？
- P1 の `sum(self._pnl_history) / len(self._pnl_history)` は `track()` 後に呼ばれるため、`pnl_bps` が既に追加された状態。これは意図通りか？
- P2 の `threshold * 0.8` は regime_thresholds の符号 (負値) を考慮しているか？ 例: threshold=-0.5 → reset=-0.4 (kill 閾値の「上」) は正しい方向か？

### 2. エッジケース
- `import_state()` で `ewma_value` が `0.0` の場合、`if _ewma is not None` で拾えるか？ (`0.0` は falsy だが `is not None` なので OK のはず)
- `_rebuild_ewma_from_history()` で `pnl_history` が 1 要素のみの場合、平均 → 1 回 replay で二重カウントにならないか？
- warmup で `sell_needs_warmup` を先に計算した後、`import_state()` が呼ばれた場合のタイミング問題はないか？

### 3. 設計判断
- `threshold * 0.8` の係数 `0.8` はハードコードされている。これを `DynamicKillConfig` のパラメータにすべきか？
- `_rebuild_ewma_from_history()` は「平均でシード → 全要素 replay」だが、「最後の N 要素のみ replay」の方が最近のトレンドを反映しないか？
- TIME LIMIT decay は「リセット」方式と「指数減衰」方式がありえる。現在のリセット方式で十分か？

### 4. テストカバレッジ
- 13 テストケースで不足しているシナリオはあるか？
- `_rebuild_ewma_from_history()` が実際に呼ばれる条件 (旧フォーマット state) のテストは十分か？
- concurrent access / thread safety の考慮は必要か？

### 5. パフォーマンス
- `_rebuild_ewma_from_history()` は `pnl_history` 全要素を走査する。`window * 3` (= 150 程度) なので問題ないはずだが、確認。
- `track()` で毎回 `len(self._pnl_history)` をチェックするオーバーヘッドは無視可能か？

## Diff

### ztb/risk/sell_dynamic_kill.py

```python
# track() — P1: 初回シードを history 平均から算出
 alpha = self._config.ewma_alpha
 if alpha > 0:
     if self._ewma_value is None:
-        self._ewma_value = pnl_bps  # 初回: seed
+        # 初回: history があれば平均でシード、なければ当該値
+        if len(self._pnl_history) > 1:
+            self._ewma_value = sum(self._pnl_history) / len(self._pnl_history)
+        else:
+            self._ewma_value = pnl_bps
     else:
         self._ewma_value = alpha * pnl_bps + (1.0 - alpha) * self._ewma_value

# 新規メソッド — P0: history からの EWMA 再構築
+def _rebuild_ewma_from_history(self) -> None:
+    """349# P0: pnl_history から EWMA 値を再構築."""
+    alpha = self._config.ewma_alpha
+    if alpha <= 0 or not self._pnl_history:
+        self._ewma_value = None
+        return
+    ewma = sum(self._pnl_history) / len(self._pnl_history)  # 平均でシード
+    for v in self._pnl_history:
+        ewma = alpha * v + (1.0 - alpha) * ewma
+    self._ewma_value = ewma
+    logger.info(
+        f"[349# P0] {self._side} EWMA rebuilt from {len(self._pnl_history)} records: "
+        f"ewma={ewma:.4f}"
+    )

# check_kill() — P2: TIME LIMIT 解除時の EWMA decay
+old_ewma = self._ewma_value
+if self._config.ewma_alpha > 0 and self._ewma_value is not None:
+    threshold = self._config.threshold_bps
+    if regime and regime in self._config.regime_thresholds:
+        threshold = self._config.regime_thresholds[regime]
+    reset_target = threshold * 0.8
+    self._ewma_value = reset_target
+    logger.info(
+        f"[349# P2] {self._side} EWMA decay on TIME LIMIT: "
+        f"{old_ewma:.3f} → {reset_target:.3f}bps "
+        f"(threshold={threshold})"
+    )

# reset() — P0: EWMA 初期化追加
 self._kill_activated_at = None  # 273#
+self._ewma_value = None  # 349# P0

# export_state() — P0: EWMA 状態永続化
 "kill_activated_at": self._kill_activated_at,  # 273#
+"ewma_value": self._ewma_value,  # 349# P0

# import_state() — P0: EWMA 状態復元 + フォールバック再構築
+_ewma = state.get("ewma_value")
+if _ewma is not None:
+    self._ewma_value = float(_ewma)
+elif self._config.ewma_alpha > 0 and self._pnl_history:
+    self._rebuild_ewma_from_history()
+else:
+    self._ewma_value = None
```

### scripts/v460/lib/orchestrator_lifecycle.py

```python
# _warmup_kill_managers_from_records() — 349#: 二重 track 防止
+sell_needs_warmup = len(self._sell_kill_mgr._pnl_history) == 0
+buy_needs_warmup = len(self._buy_kill_mgr._pnl_history) == 0
 for r in records:
     ...
-    if r.side == "sell":
+    if r.side == "sell" and sell_needs_warmup:
         self._sell_kill_mgr.track(pnl)
-    elif r.side == "buy":
+    elif r.side == "buy" and buy_needs_warmup:
         self._buy_kill_mgr.track(pnl)

# warmup 発火条件 — 349#: 両側独立チェック
-if existing_records and len(self._sell_kill_mgr._pnl_history) == 0:
+if existing_records and (
+    len(self._sell_kill_mgr._pnl_history) == 0
+    or len(self._buy_kill_mgr._pnl_history) == 0
+):
     self._warmup_kill_managers_from_records(existing_records)

# restore ログ — ewma 追加
-f"kills={self._sell_kill_mgr._total_kills}"
+f"kills={self._sell_kill_mgr._total_kills}, "
+f"ewma={self._sell_kill_mgr._ewma_value}"
```

### tests/unit/v460/test_349_ewma_fixes.py (新規, 230行)

4 テストクラス、13 テストケース:

| クラス | テスト数 | 検証内容 |
|--------|---------|----------|
| `TestEwmaStatePersistence` | 5 | export/import roundtrip, 旧フォーマット対応, 空 history, alpha=0 |
| `TestEwmaSeedStability` | 3 | 単一値 seed, history 平均 seed, 外れ値耐性 |
| `TestTimeLimitEwmaDecay` | 4 | decay 動作, regime 別閾値, alpha=0 スキップ, 再 kill 回避 |
| `TestResetIncludesEwma` | 1 | reset() 後の EWMA = None |

## トレード分析データ (修正の効果検証)

### 修正前 vs 修正後

| 指標 | 修正前 (5h) | 修正後 (6h) | 改善 |
|------|------------|------------|------|
| Fills | 5 | 41 | ×8.2 |
| WR | 20% | 46.3% | +26.3pp |
| Total PnL | -18.54 bps | -7.08 bps | +11.46 bps |
| Avg PnL | -3.71 bps/fill | -0.17 bps/fill | +3.54 bps |

### 残存損失パターン (次の改善対象)

1. **即約定 + 逆選択** (sell 集中, wait<7s): VPIN=0.82 でも offset 未連動
2. **レジーム遷移境界**: trending→ranging 切替時の offset 不整合
3. **長待ち buy 逆選択** (wait>30s): トレンド追従型の遅延約定リスク

これらは 349# の scope 外だが、次の改善候補として認識している。
