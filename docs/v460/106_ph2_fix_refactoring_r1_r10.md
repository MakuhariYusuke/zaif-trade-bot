# 106# リファクタリング調査 + 即時修正

- **parent**: `105_ph2_fix_sell_offset_balance.md`
- **status**: committed
- **commit**: (後記)

## 背景

前ターンのコードレビューで R1～R10 のリファクタリング余地を特定。
本ドキュメントは全項目の記録と、再起動前に安全に実行可能な修正の実施報告。

---

## R1～R10 全項目一覧

### 高優先度 (収益直結)

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R1** | `run_single_cycle` 分割 (~750行) | **後日** | 注文・ポーリング・stale order・E3計測・レジーム更新が混在。分割は大規模リファクタで再起動前にやるにはリスキー |
| **R2** | `BPS_FACTOR = 10_000` 定数化 | **✅ 実施済** | `run_fill_test.py` 14箇所 + `lot_sizer.py` 1箇所 |
| **R3** | SkipGate `evaluate()`/`warm_start` 単体テスト不足 | **後日** | 既存テスト: 38件 (test_enricher_skip_gate.py + test_088_features.py)。warm_start較正のunit testは有用だが再起動に影響なし |

### 中優先度 (保守性)

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R4** | ドキュメント命名違反 28件 | **後日** | 060-098番台に `phX`/`type` 欠落多数。大量リネームは運用に影響なし |
| **R5** | lib → ztb 移動検討 | **後日** | `fast_fill_defense`, `param_adapter`, `lot_sizer`, `regime_detector` の再利用性は高いが v461 移行時に対応 |
| **R6** | utils 70+ファイルの分割 | **後日** | God package。safety.py, rate_limiter.py が埋もれている。大規模リファクタ |
| **R7** | `config/` vs `configs/`, `reporting/` vs `reports/` の重複ディレクトリ整理 | **後日** | 影響範囲が広い |

### 低優先度

| # | 項目 | 判定 | 理由 |
|---|------|------|------|
| **R8** | `# type: ignore` 3箇所の解消 | **✅ 部分実施** | 1/3 解消 (`regime_detector.update` → assert ガード)。残り2箇所は正当 (psutil=untyped, SIGBREAK=Windows固有attr) |
| **R9** | インライン import 移動 | **✅ 実施済** | `import random as _rng` をトップレベルに移動。psutil は lazy import 維持 (3rd party、未インストール環境対応) |
| **R10** | 100番の番号重複解消 | **✅ 105#で解消済** | 100→101→102→103→104 cascade rename |

---

## 実施済み変更の詳細

### §1 R2: `_BPS_FACTOR` 定数化

**Before**: `* 10000` / `* 1e-4` がファイル内に散在 (14+1箇所)

**After**: クラス定数 `_BPS_FACTOR: int = 10_000` を定義し、全箇所を統一

```python
# run_fill_test.py — クラス定数
_BPS_FACTOR: int = 10_000

# 使用例 (ratio → bps)
mid_trend_bps = (mid_price - prev) / prev * self._BPS_FACTOR

# 使用例 (bps → ratio)
cumulative_pnl_jpy += pnl_bps / self._BPS_FACTOR * price * qty
```

**対象ファイル**:
- `scripts/v460/run_fill_test.py`: 14箇所 (全 `* 10000` → `* self._BPS_FACTOR`, 全 `* 1e-4` → `/ self._BPS_FACTOR`)
- `scripts/v460/lib/lot_sizer.py`: 1箇所 (`* 1e-4` → `/ 10_000`)

### §2 R8: `# type: ignore` 解消

| 箇所 | Before | After | 判定 |
|------|--------|-------|------|
| L2081 `regime_detector.update()` | `# type: ignore[arg-type]` | `assert r.mid_at_fill is not None` ガード | **解消** |
| L1147 `import psutil` | `# type: ignore[import-untyped]` | 維持 | 正当: psutil にスタブなし |
| L3033 `signal.SIGBREAK` | `# type: ignore[attr-defined]` | 維持 | 正当: Windows 固有属性 |

### §3 R9: インライン import 整理

| import | Before | After |
|--------|--------|-------|
| `random` | L1867 インライン `import random as _rng` (毎サイクル実行) | トップレベル stdlib import に移動 |
| `psutil` | L1147, L2123 インライン lazy import | 維持: 3rd party、未インストール環境対応 |

---

## 未実施項目の優先順位付け (次回以降)

| 優先 | # | 推奨タイミング |
|------|---|---|
| 1 | R1 | v461 移行時 or 大規模リファクタフェーズ |
| 2 | R3 | 次回 SkipGate 再訓練時 |
| 3 | R5 | v461 設計時 |
| 4 | R4 | ドキュメント整理一括作業時 |
| 5 | R6/R7 | リポジトリ構造整理フェーズ |

## テスト結果

- 811 passed, 0 failed (v460 unit tests)
