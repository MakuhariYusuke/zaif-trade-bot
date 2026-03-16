# 077# ph2 impl: 076# レビュー指摘対応

| key | value |
|---|---|
| 番号 | 077 |
| フェーズ | ph2 |
| 種別 | impl |
| 参照 | `076_ph2_rev_075.md`, `scripts/v460/ml/run_075_verification.py` |
| 作成日 | 2026-02-16 |
| テスト | 666 passed (side_override ×4 新規含む) |
| 目的 | 075# 外部レビュー (076#, 8 指摘) への対応 — 検証コード品質の担保 |

---

## §0 エグゼクティブサマリ

**076# 外部レビュー (8 件: CRITICAL ×2, HIGH ×3, MEDIUM ×2, LOW ×1) すべてに対応。**
**最重要修正: win_pct 計算バグと MC +0.2bps 人為バイアスの除去。**
**修正後も side filter の改善効果は維持されるが、全戦略で Holm-Bonferroni ゲート不通過。**

### 対応ステータス

| # | 重要度 | 指摘 | 対応 | 状態 |
|---|---|---|---|---|
| 1 | CRITICAL | `win_pct` 計算バグ: `np.mean([1 for x...])` ≈ 1/n | `sum(1 for ...) / len(...)` に修正 | ✅ 修正済 |
| 2 | CRITICAL | MC +0.2bps 人為バイアス → 自明な結果 | 手動加算廃止、raw data のみ使用 | ✅ 修正済 |
| 3 | HIGH | Holm/Cliff/p-mean 主張 → 実装は Wilcoxon のみ | `holm_bonferroni_gate()` + `p_mean_gate()` 実装 | ✅ 修正済 |
| 4 | HIGH | artifact path 不一致 (074 vs 075) | `verification_077/` に統一 | ✅ 修正済 |
| 5 | HIGH | `side_override` 単体テストなし | 4件のユニットテスト追加 | ✅ 修正済 |
| 6 | MEDIUM | before/after が PnL のみ | fill_rate, cancel_ratio, AS_ratio 追加 | ✅ 修正済 |
| 7 | MEDIUM | S13 queue_wait 事後依存 | 認識済、補助扱い継続 (075# 踏襲) | ✅ 認識 |
| 8 | LOW | ドキュメント "11/24h" 実際は 12/24h | "12/24h" に修正 | ✅ 修正済 |

---

## §1 CRITICAL#1: win_pct 計算バグ

### 旧コード (バグ)
```python
win_pct = np.mean([1 for x in test_pnl if x > 0]) / len(test_pnl)
```
`np.mean([1, 1, 1, ...])` は常に 1.0 → `1.0 / len()` ≈ 1-2%。

### 新コード
```python
win_pct = sum(1 for x in test_pnl if x > 0) / len(test_pnl)
```

### 修正後 結果比較
| Strategy | 旧 win_pct (バグ) | 新 win_pct (正) |
|---|---|---|
| S0_baseline | ~1.7% | 46.1% |
| S1_side_time | ~1.8% | 47.5% |
| S12_offset_sim_fix | ~1.8% | 58.4% |

---

## §2 CRITICAL#2: MC +0.2bps バイアス除去

### 旧コード
```python
after_sell_adj = after_sell + 0.2  # 人為的に +0.2bps 加算
pool_after = np.concatenate([after_buy, after_sell_adj])
```

### 新コード
```python
pool_after = np.concatenate([after_buy, after_sell])  # raw data only
```

### MC 結果 (修正後)
| | Before (global) | After (side) | 差分 |
|---|---|---|---|
| 累積PnL mean (50K steps) | +16,130 bps | +43,955 bps | +27,825 |
| Per-step mean | +0.323 bps | +0.879 bps | +0.556 |
| 正の確率 | 100% | 100% | — |
| JPY (BTC=¥15M, 0.001 BTC) | ¥+24,195 | ¥+65,932 | ¥+41,737 |

**考察**: バイアス除去後も After が Before を大幅に上回る。
ただし両方ともプラスになるのは、filter が負 PnL 時間帯をカットしている効果。
G1.1 全体 mean=-0.459 bps は依然として負であり、filter 適用後の条件付き期待値のみが正。

---

## §3 HIGH#3: Holm-Bonferroni + p_mean_gate 実装

`ztb/metrics/gate_checks.py` の既存実装を使用:

```
holm_bonferroni_gate(results, alpha=0.05, min_effect=0.10)
p_mean_gate(fold_p_values, alpha=0.05)
```

### 結果
| Strategy | p_raw | p_holm | Cliff d | PASS |
|---|---|---|---|---|
| S12_offset_sim_fix | 0.0166 | 0.0665 | +0.115 | ❌ |
| S1_side_time | 0.3122 | 0.9366 | +0.027 | ❌ |
| S13_sell_offset | 0.3745 | 0.7490 | +0.018 | ❌ |
| S9_conservative | 0.4214 | 0.4214 | +0.011 | ❌ |

**S12 が p_raw=0.0166 で惜しいが、Holm 補正後 p=0.0665 で α=0.05 を超える。**
サンプル数 284 では統計的検出力が不足。

---

## §4 MEDIUM#6: マルチメトリクス before/after

| メトリクス | Before (global) | After (side) | Δ |
|---|---|---|---|
| mean PnL | +0.322 bps | +0.917 bps | +0.595 |
| fill_rate | 83.4% | 86.8% | +3.3pp |
| cancel_ratio | 16.6% | 13.2% | -3.3pp |
| AS_ratio | 28.6% | 24.4% | -4.1pp |

**全指標で After (side filter) が改善。** PnL 以外のメトリクスでも filter 効果が一貫。

---

## §5 HIGH#5: side_override テスト

`tests/unit/v460/test_fill_test_config.py` に `TestSideOverride` クラス追加:

| テスト名 | 検証内容 |
|---|---|
| `test_side_override_skips_next_side` | override 指定時に _next_side() を呼ばない |
| `test_side_override_none_falls_through` | None 時は通常フロー |
| `test_side_override_updates_tracking` | 連続 side カウントが正しい |
| `test_run_continuous_passes_side_override` | ソースコードに side_override= パスが存在 |

---

## §6 結論と次ステップ

### 厳しい現実
1. **全戦略で Holm-Bonferroni ゲート不通過** — 284 サンプルでは統計的に有意な edge を示せない
2. **G1.1 全体 mean PnL = -0.459 bps** — filter 適用前は負
3. **S12 (offset+sim_fix) のみ win=58.4%, p_raw=0.0166** だが Holm 補正で有意水準未達

### filter 効果の一貫性
- side filter 適用後 mean = +0.917 bps (before +0.322 → +0.595 改善)
- fill_rate +3.3pp、cancel_ratio -3.3pp、AS_ratio -4.1pp と全指標改善
- **ただし条件付き期待値であり、時間制限で取引機会は 12/24h**

### 推奨
- **ph2 G1.1 ゲートはデータ蓄積待ち** — 現在のサンプル数 (1.4 日, 284 件) では判定保留
- 現行 YAML の side filter は継続稼働 (悪化要素なし)
- 追加 500+ サンプル (3-4 日分) 後に Holm-Bonferroni 再検定

---

## §7 ファイル変更一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/ml/run_075_verification.py` | CRITICAL#1/#2, HIGH#3/#4 修正, MEDIUM#6 追加 |
| `tests/unit/v460/test_fill_test_config.py` | HIGH#5: TestSideOverride (4件) 追加 |
| `docs/v460/075_ph2_impl_review_response.md` | LOW#8: "11/24h"→"12/24h", artifact path 修正 |
| `results/v460/verification_077/` | 修正後の JSON artifact |
