# 031# Fill Test 分析 & 改善

**日付**: 2026-02-14
**前提**: 030# (review response), 028# (gap analysis), 009# (fill test 設計)
**対象コミット**: (本ドキュメント作成時点)

---

## §1 Fill Test 現状分析

### §1.1 データ概要

- **期間**: 2026-02-13 〜 02-14 (約 18.1 時間)
- **総サイクル**: 275 (15.2 cycles/hour)
- **データファイル**: `fill_records_20260213.jsonl` (211L), `fill_records_20260214.jsonl` (64L)

### §1.2 G1.1 Gate 指標 vs 閾値

| 指標 | 実測値 | G1.1 閾値 | 結果 |
|------|--------|-----------|------|
| fill_rate | 75.3% (207/275) | ≥ 90% (p90) | **NG** |
| cancel_ratio | 24.7% | ≤ 30% | OK |
| queue_wait (median) | 11.8s | ≤ 60s | OK |
| post_fill_30s_pnl (mean) | -0.28 bps | ≥ 0 | **NG** |
| AS_ratio (deadzone) | 45.4% (94/207) | ≤ 20% | **NG** |

### §1.3 詳細分析

**AS の buy/sell 対称性**:
- buy AS: 52/109 = 47.7%
- sell AS: 42/98 = 42.9%
- → 片側バイアスなし。構造的問題（オフセット設定）が原因

**PnL 分布**:
- negative: 103 件 (mean = -4.07 bps)
- positive: 104 件 (mean = +3.47 bps)
- → ほぼ対称だが負側テールが重い → AS の影響

**キャンセル理由内訳**:
| 理由 | 件数 | 備考 |
|------|------|------|
| (理由なし) | 26 | 取引所 cancel/reject 未分類 |
| api_error | 24 | エラーメッセージ未取得 |
| timeout | 18 | 300s 超過 |

**時間帯別特徴**:
- 最低 fill rate: 12h (42%) — 昼間の低流動性
- 最高 fill rate: 05h (100%) — 早朝の安定期
- AS は時間帯を問わず 20-67% と高水準

### §1.4 根本原因

| 問題 | 原因 | 影響 |
|------|------|------|
| AS 45.4% | `spread_offset_ratio=0.2` がスプレッド内部に入り過ぎ | 約定は速いが逆選択多発 |
| fill_rate 75.3% | timeout + api_error + 未分類キャンセル | 25% のサイクルが無駄 |
| PnL -0.28 bps | AS の結果として負のリターン | メイカー優位性なし |
| データ品質 | spread_at_order / error_message 未記録 | 分析ボトルネック |

---

## §2 改善施策

### §2.1 AS 低減 (P0: 最重要)

**変更**: `spread_offset_ratio` デフォルト 0.2 → **0.05** に引き下げ

- **根拠**: 0.2 はスプレッドの 20% を内側にオフセット → 注文が攻撃的すぎて逆選択されやすい
- **0.05 の効果**: スプレッド 200 JPY の場合、best_bid + 10 JPY (従来は +40 JPY)
- **トレードオフ**: fill rate 低下の可能性あるが、AS 改善を優先
- **CLI 制御**: `--spread-offset-ratio` パラメータを追加、実験的に調整可能

### §2.2 スプレッドフィルター

**変更**: `min_spread_jpy` パラメータ追加 (デフォルト 0 = フィルタなし)

- 狭スプレッド時はメイカー利益余地なし → スキップ可能に
- CLI: `--min-spread-jpy`

### §2.3 データ品質改善

**FillRecord 拡張** (3 フィールド追加):
| フィールド | 型 | 目的 |
|------------|-----|------|
| `spread_at_order` | `Optional[float]` | 発注時スプレッド記録 → AS 相関分析 |
| `error_message` | `Optional[str]` | api_error 詳細 → 原因特定 |
| `spread_offset_ratio` | `Optional[float]` | 使用パラメータ記録 → A/B 比較 |

**後方互換**: 全フィールドは `Optional` + デフォルト `None` → 旧データ読み込みに影響なし

### §2.4 キャンセル理由の明確化

**変更**: 取引所からの `cancelled` / `rejected` ステータスを `exchange_cancelled` / `exchange_rejected` として明示記録

- 従来: 26 件が理由不明 (cancel_reason = None)
- 改善後: `exchange_cancelled` / `exchange_rejected` として分類

---

## §3 変更ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `ztb/metrics/fill_quality.py` | FillRecord に 3 フィールド追加 |
| `scripts/v460/run_fill_test.py` | §2.1-§2.4 全施策実装 |
| `tests/unit/v460/test_fill_quality.py` | 031# フィールドテスト 3 件追加 |

---

## §4 テスト結果

- v460 テストスイート: **318 passed** (前回 315 + 3 新規)
- 旧データ後方互換テスト: PASSED

---

## §5 次のアクション

1. **fill test 再起動**: `--spread-offset-ratio 0.05` で再実験
2. **比較分析**: 0.2 vs 0.05 の AS ratio / fill rate を比較
3. **段階的チューニング**: 結果次第で 0.03 / 0.10 なども試行
4. **G1.1 Gate 再判定**: 十分なサンプル (n≥200, 3暦日) 後に再評価
