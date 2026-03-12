# 397# Regime Confidence [0.7,0.9) 構造的問題対策 + JST 23h ガード強化

> **作成日**: 2026-03-13
> **前提**: 395# SHA-fenced 実証評価
> **変更種別**: offset パイプライン拡張 + YAML チューニング

---

## 0. 背景

395# で SHA-fenced 実証を行った結果、391# の因果主張 5 本中 4 本が current SHA で再現しなかった。
唯一 **SHA 横断で再現した構造的問題** が以下の 2 点:

1. **Regime confidence [0.7,0.9)**: 全 SHA で paradoxical underperformance (−0.734 bps, WR=46%)
2. **JST 23h (UTC 14h)**: current SHA で n=6, WR=0%, −6.996 bps — 全敗

本 397# はこれら 2 点に対する最小限の修正を実装する。

---

## 1. 変更内容

### 1.1 Regime Confidence [0.7,0.9) Offset Boost (P1-1)

**問題**: レジーム判定の confidence が [0.7,0.9) の「中程度に自信がある」帯域で、
paradoxically にパフォーマンスが最悪。[0.5,0.7) や [0.9,1.0) よりも悪い。

**仮説**: confidence が中程度のとき、レジーム分類器は「あるレジームと認識しているが、
実際は遷移中 or 別のレジーム」という状態が多い。結果、offset パイプラインが
不適切なレジーム前提でブースト/ディスカウントを適用してしまう。

**対策**: confidence [0.7,0.9) 帯域で offset を ×1.2 に拡大し、保守化する。

#### 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/fill_config.py` | `regime_mid_confidence_offset_boost`, `regime_mid_confidence_lo`, `regime_mid_confidence_hi` 3 フィールド追加 |
| `scripts/v460/lib/maker_regime_boost.py` | `_regime_boost_mid_confidence()` メソッド追加 (6 番目のステージ) |
| `scripts/v460/lib/regime_detector.py` | `RegimeDetectorLike` Protocol に `current_confidence` プロパティ追加 |
| `scripts/v460/lib/fill_config_parser.py` | YAML regime セクションの `mid_confidence_*` キーマッピング追加 |
| `configs/v460/fill_test.yaml` | `mid_confidence_offset_boost: 1.2`, `mid_confidence_lo: 0.7`, `mid_confidence_hi: 0.9` |

#### 設計判断

- **offset ×1.2 (20% 拡大)**: 控えめなスタート。high_vol_offset_boost (1.2) と同等
- **帯域 [0.7,0.9)**: 395# SHA-fenced データで明確に negative な唯一の帯域
- **RegimeDetectorLike Protocol 拡張**: `current_confidence` は既に `FillTestRegimeDetector` に存在したが Protocol 未定義だった。型安全性向上
- **6 ステージ化**: dispatcher `_apply_regime_boosts()` は trending → high_vol → ranging → low_vol → unknown_buy → mid_confidence の順序。最終段で confidence-based 補正をかけることで、他のレジーム boost との干渉を最小化

### 1.2 JST 23h (UTC 14h) ガード強化

**問題**: current SHA で JST 23h (UTC 14h) が n=6, WR=0%, −6.996 bps。

**対策**: 2 つのガードを強化:

| パラメータ | 旧値 | 新値 | 根拠 |
|-----------|------|------|------|
| `sell_hour_offset_boost[14]` | 1.3 | **1.5** | sell offset ×1.5 で逆選択コスト低減 |
| `skip_gate.hour_offsets[14]` | 0.3 | **0.5** | SkipGate 閾値厳格化 (UTC 16h と同等に引上げ) |

#### 設計判断

- n=6 のためサンプル極小 → hard_skip には至らず
- 既存の 2 つの防御レイヤー（offset 拡大 + skip_gate 厳格化）の強度を上げる
- 3/13 以降のデータで効果を検証し、必要に応じて再調整

---

## 2. テスト影響

| テストファイル | 変更内容 |
|--------------|---------|
| `test_260_compute_extract_regime_split.py` | dispatcher が 6 sub-method を呼ぶことを検証、`_regime_boost_mid_confidence` メソッド存在確認追加 |
| `test_258_as_reservation_vpin_continuous_protocol.py` | Protocol mock に `current_confidence` 追加 |
| `test_259_as_vol_ratio_adaptation_hasattr.py` | mock detector に `current_confidence` 追加 |
| `test_336_yaml_code_drift_prevention.py` | `regime_mid_confidence_offset_boost` を KNOWN_YAML_OVERRIDES に追加 |

全テスト通過確認済み (60 + 149 + 198 = 407 tests passed)。

---

## 3. リスク評価

### 低リスク

- **offset ×1.2 は控えめ**: fill rate への影響は限定的（confidence [0.7,0.9) 自体の出現頻度が限定的）
- **JST 23h 強化も穏当**: hard_skip ではなく既存ガードの閾値強化のみ
- **既存テスト全通過**: 既存のレジーム検知・offset パイプライン・YAML パーサーの動作に影響なし

### 監視項目

1. **confidence [0.7,0.9) の fill rate 変化**: offset 拡大で fill が減る可能性 → 3 日間監視
2. **JST 23h の PnL 改善**: WR=0% からの改善を確認
3. **confidence [0.7,0.9) の regime_at_order vs regime_30s_later 一致率**: P2 として将来のデータ収集

---

## 4. 395# 行動計画との対応

| 395# 項目 | 本 397# での対応 | 状態 |
|-----------|----------------|------|
| P0-1: EV方向ガード不採用 | — (変更不要: 実装禁止確定済) | ✅ |
| P0-2: 5分制限不採用 | — (変更不要: 実装禁止確定済) | ✅ |
| P1-1: confidence [0.7,0.9) 対策 | offset ×1.2 実装 | ✅ |
| P1-2: JST 23h 点検 | sell_hour_boost 1.3→1.5, skip_gate 0.3→0.5 | ✅ |
| P1-3: Buy 側劣化監視 | — (データ蓄積待ち、3日後再評価) | ⏳ |
| P2-1: 既存ガード効果測定 | — (将来タスク) | ⏳ |
| P2-2: Sell tail 傾向確認 | — (改善傾向、現時点で変更不要) | ⏳ |
