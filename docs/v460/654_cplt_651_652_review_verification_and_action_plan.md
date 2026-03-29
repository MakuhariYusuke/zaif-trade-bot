# 654# 651-652 レビュー検証 — 利益保全型提案の実装状況と見落とし分析

- **日付**: 2026-03-30
- **目的**: 651#(利益保全型レビュー)・652#(セカンドオピニオン)の提案を、現行コードベース・設定と照合し、実装状況・見落とし・追加改善点を特定する。605#/606# の「渙」第三段階—「守って切る」を実行フェーズに移行する。
- **入力**: 651#, 652#, 650# (実測データ), 653# (561-562検証)
- **注**: 653# は 561-562 の検証文書だが、提案の多くが 651-652 と収束しており相互補完的。

---

## §0 総括

651# と 652# は共に「全体が負け」のフェーズを脱した現状 (avg_pnl30=+0.428bps, PF 1.20) を正しく認識した上で、**利益保全型の局所改善** を提案している。652# は 651# に対する独立レビューとして、Spoofing 耐性 (price_velocity) や Regime-Drift Exit 等の補強を加えており、両文書の品質は高い。

以下は現行コードベースとの突合結果である。

---

## §1 P0 提案の実装状況

### 1.1 P0-1: Inventory Skew 再調整 — ❌ 未実装

**提案**: `neutral_band: 0.10 → 0.05`, `decay_tau_sec: 1800 → 3600`

**現状** (`configs/v460/fill_test.yaml` L918-925):
```yaml
inventory_skewing:
  enabled: true
  window: 100
  max_factor: 0.4
  neutral_band: 0.1       # ← 未変更
  decay_tau_sec: 1800.0    # ← 未変更
  regime_gate_enabled: true
```

**検証**: 650# で 25/27 fills の skew_factor = 0.0000 を確認済み。triple-disable (neutral_band 0.1 + regime_gate_enabled + decay_tau 短すぎ) により inv_skew は事実上の死んだコード。651# と 652# の指摘は完全に正当。

**652# 追加懸念**: min_spread_bps フロア vs skew の適用順序 → maker_price.py で skew 適用後に min_spread_bps が最終防衛線として機能することを確認済み。**問題なし**。

**アクション**: 設定変更のみ。即時実行可能。

### 1.2 P0-2: Toxic Low-Spread Sell Veto — ❌ 未実装

**651# 提案**: `sell` + `spread_bps < 2.3` + `ob_imbalance > 0.25` + `vpin > 0.65`
**652# 補強**: `price_velocity_bps > 0` を追加条件に (spoofing 耐性)

**現状**:
- `velocity_skip`: `sell_velocity_skip_enabled: true`, `threshold: 4.0bps` — 速度ベースのスキップは存在
- `velocity_skip_as_offset_enabled: true` — hard skip ではなく offset boost に変換済み
- `price_velocity_bps`: fill_cycle_executor L881/L904 で skip_gate に渡されている → **パイプラインで利用可能**
- `ob_imbalance`, `vpin`: build_features.py で計算済み → **特徴量として利用可能**
- **しかし**: spread + OBI + VPIN + velocity の **複合条件** での sell veto は未実装

**見落とし分析**:
- 現行の `narrow_spread_pause` (fill_cycle_executor L824) は side 非依存。651# の「sell 限定の低 spread 条件」は別機構が必要。
- skip_gate_evaluator にも low-spread+sell の compound guard は存在しない。

**アクション**: skip_gate_evaluator または fill_cycle_executor に compound sell veto 追加が必要。

### 1.3 P0-3: Long Hold Sell-Entry Escape — ⚠️ 部分実装

**651# 提案**: sell-entry の長時間 hold に対する優先的 close モード
**652# 補強**: hold > N 分 + regime=trending_up → Soft Stop Loss (即時撤退)

**現状**:
- `micro_timeout`: 有効 (`enabled: true`, sell: 10s, buy: 15s, max_requote: 4) — **オーダー単位** の timeout であり、**RT 単位** の hold time 監視ではない
- `macro_sell_timeout_strong_up`: 6s, `weak_up`: 12s — macro regime に応じた sell timeout 短縮あり
- **RT 単位の hold time 超過 → 積極的 close** の機構は存在しない

**見落とし分析**:
- micro_timeout は「発注→約定」の短期 timeout。651# が言及する「RT の entry→exit の hold time 69.7 分」とは全く異なるスケール。
- 652# の regime-drift exit は理論的に正しいが、現行アーキテクチャに RT-level position tracking がない可能性あり → 要調査。

**アクション**: 中期課題。RT ベースの position tracking と regime-aware exit trigger の設計が必要。

---

## §2 652# 独自提案の検証

### 2.1 P1-X: 非対称 RT Exit Tolerance — ❌ 未実装

**提案**: Sell-entry 決済 (buy back) は利益要求を妥協して早期クローズ、Buy-entry 決済は利益追求。

**現状**:
- `execution_final_clamp_hard_skip_mult_overrides` で side/regime 別の clamp 設定は可能
- `sell_asymmetric_high_vol_enabled` が存在するが、これは **entry** の抑制であり **exit tolerance** ではない
- Buy-entry vs Sell-entry の **決済** 時の offset/clamp 非対称設定は未実装

**評価**: 構造的に正しい提案。現物 BTC/JPY の上方ドリフトバイアスを考慮すると、sell-entry の hold 延長は本質的に不利。実装にはエントリー方向を記憶した close-side の挙動分岐が必要。

### 2.2 652# Self-Audit の評価

652# の自己点検は鋭い。特に:

1. **サンプルサイズ問題**: 13 RT は統計的に不十分。ダウントレンドでは sell-entry が有利に反転する可能性 → 固定値ではなく regime 条件付き非対称にすべき
2. **フィルター追加 → skip 率上昇リスク**: P0-2 の veto 追加が max_skip_rate (0.4) に当たる可能性 → P0-1 (inv_skew 改善) が前提条件として必須

---

## §3 651#/652# が *見落としている* 点

### 3.1 eDRC (動的 ceiling) の α=β=0 無効化

fill_config.py に eDRC が実装済みだが `alpha=0.0, beta=0.0` で完全無効。sell ceiling 0.40 で 92% 飽和している現状、DRC/eDRC の有効化は ceiling 問題の根治策になりうるが、**両文書とも eDRC に言及していない**。

576# インシデント (α/β 推定ミス) の教訓から慎重に再推定する必要はあるが、eDRC の存在自体を認識しておくべき。

### 3.2 VG boost + ceiling 吸収の矛盾

650# 問題4: VG (Volatility Guard) が boost > 1.0 で sell offset を引き上げようとしても、ceiling 0.40 で頭打ち → VG の防御が形骸化。651# は「sweet spot は 2.7-3.1bps」と指摘しているが、VG+ceiling の矛盾構造には踏み込んでいない。

### 3.3 sidecar stale 93% の構造的意味

651# は「sidecar stale は P2」と判定。current data (3/29) ではプラスなので優先度は低いが、**inv_skew 改善で buy fill が増えると sidecar prediction の重要度が上がる** 点を見落としている。P0-1 実装後の secondary effect として注視すべき。

### 3.4 sell model 停止 (model_path_sell: null) の長期リスク

645# で degenerate sell model を停止したのは正しい止血だが、sell 側の ML 学習パイプライン自体が放棄されている。651# は「今やるべきでない」としているが、retrain 経由での sell model 再構築の条件を明文化しておくべき。

---

## §4 渙 (風水渙) 第三段階アセスメント

605# で「渙 = 散じて再構成する」流れが始まり、606# で SAD/MCB 解凍、607# で hot-reload、641#+以降で個別止血が進んだ。

**第一段階** (605-607): 固着の解体 → SAD/MCB enabled, hot-reload
**第二段階** (641-650): 個別止血 → sell model停止、σ stale fix、RT分析
**第三段階** (651-652→今): 利益保全型局所改善 → 勝ち筋を守りつつ負け筋を切る

第三段階は「散じきった後の再凝集」に相当する。651# が正しく指摘する通り、もう全面的な guard 増設ではなく、**精密な条件付き遮断** が必要なフェーズ。

---

## §5 実行計画

### T0: 即時 (設定変更のみ)
1. **inv_skew 調整**: `neutral_band: 0.10 → 0.05`, `decay_tau_sec: 1800 → 3600`
   - 651#/652# 共に最優先。config 変更のみ、リスク最小

### T1: 短期 (コード変更あり)
2. **Toxic low-spread sell veto**: 複合条件 `sell + spread<2.3 + OBI>0.25 + VPIN>0.65 + velocity>0`
   - 651# P0-2 + 652# velocity 補強
   - skip_gate_evaluator に compound guard 追加
3. **RT 主 KPI 化**: 651# P1 — pnl30 単独から RT PnL 主体の評価基盤へ

### T2: 中期
4. **Regime-drift exit for sell-entry hold**: 652# P0-3 補強版
5. **Asymmetric RT exit tolerance**: 652# P1-X — sell-entry close の利益妥協
6. **eDRC α/β 再推定**: 653# T2-1 — ceiling 飽和の根治策

### T3: 長期
7. **Sell model 再構築条件の明文化**
8. **Sidecar retrain 成功率改善**

---

## §6 自己批判

- 651#/652# の分析精度は高く、特に 651# の empirical grounding (Q1-Q4 split, RT entry-type analysis) は actionable
- 652# の price_velocity 追加は理論的に正当だが、スプーフィング検知としては限定的 (高速 HFT の見せ板は velocity にも現れる)。VPIN が本来その役割を担うため、velocity 条件は「補助」として位置づけるのが妥当
- P0-3 (long hold escape) は現行アーキテクチャへの影響が大きく、micro_timeout 拡張では不十分。RT tracking 基盤の整備が前提
- 13 RT でのサンプルサイズ限界は 652# が正しく指摘。少なくとも 50+ RT の蓄積まで、非対称戦略の固定パラメータ化は避けるべき

---

## §7 結論

651#/652# の P0 提案は正当であり、**T0 (inv_skew 設定変更) は即時実行可能**。T1 (toxic sell veto) はコード変更を伴うが、既存パイプラインに必要な特徴量 (VPIN, OBI, velocity) は揃っており、実装障壁は低い。

653# (561-562 検証) とのクロスリファレンスでは、**inv_skew triple-disable** と **ceiling 飽和** が 561-562 と 651-652 の両方で収束する最重要課題。渙の第三段階として、まず「血流再開」(inv_skew) → 次に「毒の遮断」(toxic sell veto) の順序で進める。
