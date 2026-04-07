# 712# Post-Restart Performance Analysis & Ceiling/Offset Recalibration

| 項目 | 内容 |
|------|------|
| 日付 | 2026-04-08 |
| SHA | (pre-commit) |
| 前提 | 750cd71 (CX4-CX6 + 710# deep analysis) で 4/7 12:37 再起動後の 191 レコード (97 filled) |

## 1. 実績サマリ (SHA 750cd71, 4/7)

| 指標 | Buy (48) | Sell (49) | 全体 (97) |
|------|----------|-----------|-----------|
| spread_capture avg | -0.716 | -0.615 | -0.665 |
| spread_capture sum | -34.4 | -30.2 | -64.5 |
| AS rate | 22.9% | 32.7% | 27.8% |
| 平均 spread (bps) | 2.65 | 2.26 | — |

### SHA 比較 (同日)
- **OLD (6e193a6)**: 36 fills, sc_avg=-0.975 **(悪い)**
- **NEW (750cd71)**: 97 fills, sc_avg=-0.665 **(改善)**
- 4/4-4/6 ベースライン: -0.320 〜 -0.481

→ 新コードは同日比では改善。ただし 4/7 の市場環境が 4/4-6 より厳しかった。

## 2. Root Cause Analysis

### RC1: Final Clamp Over-Restriction (最大要因)

| 指標 | Buy | Sell |
|------|-----|------|
| Pre-clamp avg offset | 0.5613 | 0.7325 |
| Final offset (clamped) | 0.2638 | 0.4131 |
| **Clamp rate** | **100%** | **100%** |
| Clamped fills sc_avg | -0.853 | — |
| Unclamped fills sc_avg | -0.569 | — |

Pipeline は適切なリスク反映 offset を計算しているが、ceiling が低すぎて全てカット。

### RC2: Entry Gate 無駄ブロック
- cancelled 94 件中、88 件が `entry_gate_blocked` (stale CalibrationMap で全て SUPPRESSED)
- CPU + ログ浪費のみ、実質 0 件の有効ブロック

### RC3: Skip Gate Model 逆転
- organic pass (10 fills): sc_avg=-0.951
- bypassed (28 fills): sc_avg=-0.624
- → ML 通過判定がむしろ悪い fill を選択 (サンプル少数、断定不可だが傾向注視)

### RC4: Trending Cross-Fill 損失集中

| Regime/Side | fills | sc_avg | sc_sum |
|-------------|-------|--------|--------|
| trending_down/buy | 9 | -1.55 | -13.9 |
| trending_up/sell | 3 | -1.52 | -4.6 |
| trending_up/buy | 4 | -1.15 | -4.6 |
| trending_down/sell | 9 | -0.95 | -8.6 |

→ 全 trending 組は負。Pipeline が高 offset を計算するも clamp で切り詰め。

### RC5: Cancel 内訳
| Reason | Count |
|--------|-------|
| entry_gate_ev_negative | 5 |
| spread_too_narrow | 24 |
| timeout | 44 |
| postonly_crossing_skip | 5 |
| final_clamp_hard_skip | 7 |
| その他 (kill/mcb/drift) | 9 |

## 3. 修正内容

### F1: Offset Ceiling 引上げ (P0)
```yaml
offset_ceiling_ratio_buy: 0.35 → 0.50    # pipeline avg 0.56 の 89% を通過
offset_ceiling_ratio_sell: 0.50 → 0.65   # pipeline avg 0.73 の 89% を通過
```
**根拠**: 100% clamp 率。Pipeline のリスク判断を ceiling が体系的に無視。
ceiling を引き上げることで中央値 (p50) 程度の pipeline 出力が通過可能に。

### F2: Entry Gate 無効化 (P0)
```yaml
entry_gate_enabled: true → false
```
**根拠**: 88 blocked/日が全て stale_calibration_map で抑制。CalibrationMap 再学習まで無効化。

### F3: Side Offset 引上げ (P1)
```yaml
side_offset:
  buy: 0.08 → 0.10    # buy AS=22.9%, unclamped fills の base offset 強化
  sell: 0.14 → 0.18   # sell AS=32.7% (688# 時点 30.4% から悪化)
```
**根拠**: AS 率上昇。ceiling 引上げ後も unclamped fills のベースが不足。

## 4. Hot-Reload 対応

| パラメータ | Hot-Reload |
|-----------|-----------|
| offset_ceiling_ratio_buy/sell | ✅ (MakerPriceCalculator sync) |
| entry_gate_enabled | ✅ (EntryGateGuard reset) |
| side_offset → spread_offset_ratio_buy/sell | ✅ (MakerPriceCalculator sync) |

→ **全変更が hot-reload 対応**。コールドリスタート不要。

## 5. テスト

| テスト | 結果 |
|--------|------|
| test_fill_quality.py + test_093 + test_169 | 262 passed |
| ceiling/clamp/entry_gate 関連 | 270 passed |
| v460 全体 | 4552 passed, 8 skipped |
| 唯一の failure | test_260 line count (pre-existing, 無関係) |

## 6. 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | F1 ceiling + F2 entry_gate + F3 side_offset |
| `tests/unit/v460/test_fill_quality.py` | sell offset assertion 0.14→0.18 |
| `tests/unit/v460/test_336_yaml_code_drift_prevention.py` | entry_gate_enabled を KNOWN_YAML_OVERRIDES から除去 |
| `docs/v460/712_cplt_post_restart_ceiling_recalibration.md` | 本ドキュメント |

## 7. 期待効果

- **Ceiling 引上げ**: clamp 率 100% → 推定 40-60% まで低下。Pipeline のリスクシグナルがより反映。
- **Entry gate 無効化**: 88 cycles/日の無駄ブロック解消。ログノイズ削減。
- **Side offset 引上げ**: unclamped fills Δoffset ≈ +25% (buy), +28% (sell)。
- **Trending fills**: ceiling 引上げにより pipeline が計算した高 offset がより通過 (特に trending_down/buy の 0.975 → 0.50 vs 旧 0.35)。

## 8. 監視ポイント (次回確認)

1. **Clamp 率**: 100% → 目標 50% 以下へ
2. **spread_capture avg**: -0.665 → 目標 -0.3 以下 (4/4-6 ベースライン並)
3. **Hard skip 率**: 7/94 → ceiling 引上げで減少見込み
4. **AS rate**: sell 32.7% の推移 (side_offset 引上げで低下を期待)
5. **Skip gate**: organic pass vs bypass の sc 比較 (逆転続けば model 再学習要検討)
