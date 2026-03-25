# 630# P1 閾値チューニング (626#/629# アクション消化)

## 概要

629# で整理した P1 アクション 3 件を実施。config-only 変更で AS 防御層を強化。

---

## 変更内容

### P1a: sell velocity skip 閾値引き下げ

| 項目 | 変更前 | 変更後 |
|------|:------:|:------:|
| `sell_velocity_skip_threshold_bps` | 6.0 | **4.0** |

- 626#/629# 根拠: vel≥4bps 帯の sell AS 率 60% 超
- buy 側 (-4.0) と対称化で sell/buy 均等防御
- `velocity_skip_as_offset_enabled=true` のため hard skip ではなく proportional soft boost (1.5x–4.0x)

### P1b: regime trending 閾値引き下げ

| 項目 | 変更前 | 変更後 |
|------|:------:|:------:|
| `trend_threshold_pct` | 0.5 | **0.20** |

- 629# §2.2 根拠: 40min 窓で 0.5% は鈍感すぎて trending が発火しない
- 0.20% @ 40min ≈ 20bps → BTC 日中ボラの実用域
- 下流: `trending_up_sell_offset_boost=1.8` の発火頻度増加（意図した防御効果）
- 窓短縮 (20→5 obs) は P2 据え置き

### P1c: VG velocity 閾値引き下げ

| 項目 | 変更前 | 変更後 |
|------|:------:|:------:|
| `velocity_threshold_bps` | 12.0 | **6.0** |

- 626# §3 根拠: VG vel = 逆選択最強因子。12bps では事実上発火しない
- vel_skip 4bps + VG 6bps の 2 段防御構造:
  - 4bps≤vel<6bps: velocity proportional boost のみ
  - vel≥6bps: velocity boost + VG `offset_boost_factor=2.0` の複合防御

---

## セルフレビュー

| チェック項目 | 結果 |
|-------------|------|
| sell/buy velocity 対称性 | ✅ sell=4.0, buy=-4.0 で対称 |
| VG と vel_skip の棲み分け | ✅ 4bps (soft) < 6bps (VG 2x) の階層 |
| regime 下流暴走リスク | ⚠ trending_up_sell_offset_boost=1.8 発火増。fill rate 低下の可能性あり |
| clamp 飽和との相互作用 | ⚠ ceiling sell=0.40 で offset boost が頭打ちになる可能性（P2 課題） |
| テスト | ✅ 2237 passed, 127 skipped |
| ロールバック容易性 | ✅ YAML 3 値変更のみ |

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `configs/v460/fill_test.yaml` | 3 閾値変更 (velocity 6→4, regime 0.5→0.20, VG 12→6) |

## 残 P2/P3 課題

| 優先度 | アクション | 状態 |
|:------:|-----------|:----:|
| P2 | clamp ceiling 引き上げ検討 (627# §5.3) | 未着手 |
| P2 | Velocity Zスコア動的化 R&D (628# §2.1) | 未着手 |
| P2 | regime 窓短縮 (20→5 obs) 検討 | 未着手 |
| P3 | SG 予測ホライズン 30s→60s 検討 | 未着手 |
