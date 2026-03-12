# 264# Kelly Criterion ロットサイジング

| 項目 | 値 |
|---|---|
| Issue | 264# |
| 種別 | impl (市場理論統合) |
| フェーズ | phg (横断品質改善) |
| Commit | `93d7d549a` |
| テスト | 3640 passed, 32 skipped (+24 新規) |
| 元チケット | 259# 市場理論「未実装 — 拡張候補」 |

---

## 背景

既存のロットサイジング (方策 B) は step-based の段階調整 (`±lot_step`) のみで、
理論的な最適ポジションサイズを考慮していなかった。

Kelly Criterion を「天井」として統合し、過剰なロット増量を理論的に制約する。

## Kelly Criterion の公式

$$f^* = \frac{p \cdot b - q}{b}$$

| 変数 | 意味 |
|---|---|
| $p$ | 勝率 (`post_fill_30s_pnl > 0` の約定割合) |
| $q$ | $1 - p$ |
| $b$ | 平均勝ち幅 / 平均負け幅 (bps 絶対値比) |
| $f^*$ | 最適ベット比率 (bankroll の何%) |

### Fractional Kelly

Full Kelly はボラティリティが高すぎるため、**Half-Kelly** ($f^*/2$) をデフォルトとする。
さらに `max_fraction = 0.25` でキャップし、過剰リスクを防止。

## アーキテクチャ

```
FillRecord[] → compute_kelly_fraction() → KellyEstimate
                                              ↓
                                    kelly_recommended_lot()
                                              ↓
                                    kelly_estimate.recommended_lot
                                              ↓
                            compute_lot_size(kelly_estimate=...)
                                    → Kelly 天井適用
```

### 統合方式: 天井モデル

- 既存の step-based 増量ロジックはそのまま維持
- Kelly 推奨ロットを「天井」として使用
- `step_based_new > kelly_lot` の場合 → `max(current, kelly_lot)` に制限
- 減量には介入しない (天井のみ)
- `kelly_estimate = None` or `recommended_lot ≤ 0` の場合 → 天井なし

## 新規コード

### `lot_sizer.py` — 追加分

| 要素 | 説明 |
|---|---|
| `KellyEstimate` | dataclass: win_rate, win_loss_ratio, kelly_fraction, fractional_kelly, recommended_lot, sample_count, reason |
| `LotSizingConfig` 拡張 | `kelly_enabled`, `kelly_fraction` (0.5), `kelly_min_win_samples` (30), `kelly_max_fraction` (0.25), `kelly_equity_btc` |
| `compute_kelly_fraction()` | FillRecord → KellyEstimate \| None |
| `kelly_recommended_lot()` | KellyEstimate × equity_btc → BTC ロット (lot_step 丸め) |
| `compute_lot_size()` | `kelly_estimate` パラメータ追加、増量分岐に Kelly 天井 |

### `adaptation_engine.py` — 統合

`try_auto_lot_size()` 内で:
1. YAML `kelly:` セクションから設定を読み込み
2. `compute_kelly_fraction(records)` を呼び出し
3. `kelly_recommended_lot()` で BTC ロットに変換
4. `compute_lot_size(kelly_estimate=...)` に渡す

## YAML 設定

```yaml
kelly:
  enabled: true
  fraction: 0.5          # half-Kelly
  min_win_samples: 30    # Kelly 推定に必要な最小サンプル
  max_fraction: 0.25     # f* 上限
  equity_btc: 0.01       # 口座残高 (BTC 建て)
```

デフォルト: `kelly.enabled = false` → 既存動作に影響なし。

## テスト (24 件)

| クラス | テスト数 | 内容 |
|---|---|---|
| `TestComputeKellyFraction` | 11 | 基本計算, half-Kelly, サンプル不足, no edge, max cap, 全勝/全敗, 未約定除外, PnL=0 除外 |
| `TestKellyRecommendedLot` | 6 | BTC ロット変換, min/max clamp, step 丸め, zero equity, 負 fraction |
| `TestComputeLotSizeKellyCeiling` | 7 | 天井ブロック, 部分増量, 通常増量, ゼロロット, 減量非介入, 損失キャップ優先 |

## リスク評価

| リスク | 対策 |
|---|---|
| Kelly 過大推定 → 過剰ロット | Fractional Kelly (50%) + max_fraction (25%) + 既存 max_lot |
| サンプル不足 → 不安定な推定 | `min_win_samples = 30` → サンプル不足なら None (天井なし) |
| PnL 分布の非定常性 | 直近レコードのみ使用 (adaptation_engine の window) |
| 既存動作への影響 | `kelly_enabled = false` がデフォルト → 明示的に有効化しないと動作しない |

## 今後の拡張候補

- Kelly × レジーム連動: レジーム別に Kelly を算出 (trending/ranging で異なるエッジ)
- ログ離散: Kelly f* を Prometheus メトリクスとして公開
- equity_btc の自動取得: BalanceAdapter から BTC 残高を自動計算
