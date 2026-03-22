# 546# レビュー依頼: Phase 4 以降の施策選定

> **目的**: 540-545# で実施した Phase 1-4b の成果を踏まえ、次の施策について別 AI のレビューを受ける  
> **日付**: 2026-03-23  
> **背景**: fill_test は旧 SHA で稼働中。540#-545# の変更はまだ本番未反映。

---

## §1 完了済み施策 (540-545#)

| Commit | 内容 | 理論基盤 |
|--------|------|----------|
| 540# | Pipeline 実態検証 + Phase 1 YAML (kill 600s, composite 1.0) | データ駆動検証 |
| 541# | Lazy import 8箇所 + disabled stage 5段 skip | パフォーマンス |
| 542# | Ceiling 0.25→0.30 (pre-clamp median=0.2996) | 実測分布適合 |
| 543# | OFI-Lite (CKS 2014) + Toxicity Budget 独立化 (GM 1985) + A-S δ* 計測 | 学術理論実装 |
| 544# | δ*→spread_adapt 動的閾値 + OFI rolling (50cyc) + sidecar 0.20 | 理論→制御接続 |
| 545# | OFI→boost 変調 + Toxicity→sidecar confidence + δ*→sidecar ceiling | 固定値の全面動的化 |
| 546# | sidecar shaping: quadratic + OFI k config化 | チューニング準備 |

---

## §2 今回のレビュー対象: 保留施策 5 件

### A) Sidecar max_boost_bps 0.20→0.30 (固定引上げ)

**現状**: max_boost_bps=0.20, shaping=quadratic (546# で変更)  
**545# で δ* dynamic ceiling (×2.0 cap) が入っている** → δ*>1.0 のとき自動拡大済み

**検討ポイント**:
1. 固定 0.30 引上げは δ* dynamic ceiling と重複しないか？
   - δ*=1.5 のとき: 固定 0.30 × 1.5 = **0.45bps**、δ* ceiling なしなら 0.30
   - 現行 0.20 × 1.5 = **0.30bps** (δ* ceiling のみで 0.30 に到達)
2. quadratic shaping で中間帯の出力は抑制されるため、max_boost の天井引上げは安全性が高い
3. **SAC 学習が 0.20 前提で行われている場合**、0.30 に引上げると分布が変わり SAC の calibration が崩れるリスク
4. fill_test データなしで判断すべきか

**提案された選択肢**:
- (a) 0.30 に引上げ（SAC 影響力拡大）
- (b) δ* dynamic ceiling に完全委任（理論が正当化するときのみ拡大）← 現状
- (c) データ確認後に段階的引上げ (0.25 → 0.30)

---

### B) CalibrationMap → sidecar confidence 統合 (538# §6「第三の道」)

**CalibrationMap とは**: `ztb/trading/signal/calibration_map.py`  
- EWMA decay (tau=100) + Beta CI で regime×action 別の `p_win_lcb` (勝率下限信頼区間) を算出
- Training 環境 (fast_intraday_env) でのみ使用中、v460 fill_test では**未使用**

**検討ポイント**:
1. **コールドスタート問題**: CalibrationMap は最低 50-100 fills で有意な n_eff に到達。fill_test 開始直後は信頼できない
2. **リアルタイム vs オフライン**: fill_test JSONL からバッチ構築 → 起動時ロード vs サイクル毎 update
3. **入力次元の選択**: regime だけか、regime×side か、regime×action×spread_band か
4. **Toxicity confidence との競合**: 545# で ToxicityLevel → confidence attenuation を入れた。CalibrationMap の p_win_lcb も confidence に影響するなら **二重減衰** のリスク
5. **538# の「第三の道」**: SAC のオンライン学習の代替として提案。full RL ではなく低次元の calibration table

**提案された選択肢**:
- (a) fill_test_runner にリアルタイム update 統合
- (b) オフラインバッチ構築 → JSON/YAML エクスポート → 起動時ロード ← 推奨
- (c) Regime 別固定テーブル（手動、CalibrationMap 不使用）
- (d) 保留（fill_test 1 週間分のデータ蓄積後に再検討）

---

### C) δ* → executor pipeline 伝搬

**現状**: δ* は pre-order (maker_price) の spread_adapt + sidecar ceiling でのみ使用。executor の 9 段チェーンには到達しない。

**executor pipeline 構造** (9段):
1. EV offset (skip_gate 由来)
2. Velocity offset
3. Trending offset (sell-side only)
4. **Toxicity offset** ← δ* floor 注入候補
5. VG supplement (発火率 0%)
6. Macro boost (発火率 0%)
7. Alert mode
8. Sidecar offset (BPS→price)
9. Final clamp (ceiling 0.30)

**検討ポイント**:
1. **二重適用リスク**: pre-order で δ* が spread_adapt の narrow_bps を引上げ済み → executor でも floor を入れると**同じ情報が 2 回効く**
2. 540# で「executor は既に tame」と評価 — VG/Macro 死亡、実効 5 段
3. Toxicity 段 (240#) に floor を入れる案: `final_mult = max(toxicity_mult, delta_star_floor)` → toxicity が「縮小」と言うのを δ* が「いやキープ」と上書き → **理論間衝突**
4. Final clamp 段に δ* を参照する案: 「理論的に妥当なら hard_skip しない」→ ceiling 0.30 の根拠 (542#) と矛盾
5. 新段追加は pipeline の複雑性を増す

**提案された選択肢**:
- (a) Toxicity 段に δ* floor 注入
- (b) Final clamp に δ* 参照
- (c) 新段として δ* floor を追加
- (d) 保留（二重適用リスクが高い、executor は既に安定）← 推奨

---

### D) Drift Detection (OFI / Toxicity 分布監視)

**既存実装**: `ztb/utils/drift_detection.py` — PSI (Population Stability Index) + KS テスト  
**現在**: v460 未使用。archived/scripts/feature_drift_report.py で過去使用。

**検討ポイント**:
1. **baseline 不在**: OFI-Lite は 543# で初実装。「正常分布」が未確立 → drift 検知の前提がない
2. **Toxicity 分布は簡易監視可能**: GREEN/YELLOW/ORANGE/KILL の割合カウンタ → ORANGE+KILL > 30% 持続でアラート
3. **OFI drift → 何をするか**: PSI > 0.2 で「市場構造が変化した」と検知できても、**アクション定義**がない。boost 増加？一時停止？
4. **計算コスト**: PSI は O(N) だがバッファ管理が必要。100 cycle buffer × 毎サイクル比較は軽量

**提案された選択肢**:
- (a) Toxicity 分布カウンタ（簡易、即実装可能）
- (b) OFI rolling PSI + アラート（中工数、baseline 蓄積後）
- (c) EvaluationManager 統合（大工数、既存フレームワーク活用）
- (d) 保留（baseline 蓄積優先）← 推奨

---

### E) その他の検討事項

**VG/Macro executor 死亡段**: コードは残存、発火率 0%。削除候補だがマクロトレンド再接続時に活用可能。

**sell_hour 時間帯ルール**: 536# が「ハードコード」と指摘したが、実際は YAML 辞書ベースで動的。問題なし。

**pre-order identity 段 (11/14)**: 541# で disabled 5段 skip 済み。残り 6 段は稀少発火（vol_guard, cross_venue, sell_hour 等）で防御的に保持。

---

## §3 レビューで確認したい論点

1. **Phase 4 の動的化アプローチ（OFI→boost, Toxicity→confidence, δ*→ceiling）は理論的に健全か？**
   - 特に: 3 つの独立制御チャネルが同時に作用したとき、過剰に保守的にならないか
   - 例: Toxicity=ORANGE (×0.3) + adverse OFI (boost ×1.5) + δ*>1.0 → 実質的に sidecar 無効化 + boost 増大 → 約定率が極端に低下しないか

2. **CalibrationMap の統合タイミング**: fill_test データなしで構築開始すべきか、1 週間待つべきか

3. **executor への δ* 伝搬**: 二重適用リスクは実際にどの程度か？pre-order と executor で同じ理論値を使うことの是非

4. **sidecar quadratic shaping の妥当性**: linear→quadratic で中間帯の SAC 影響力が低下。SAC の学習精度が中間帯で高い場合、これは逆効果ではないか

5. **全体アーキテクチャ**: 540-545# で pre-order pipeline に理論ベースの動的化を入れたが、executor pipeline はほぼ手付かず。この非対称性は問題か

---

## §4 参考データ

### 現在の動的制御フロー

```
Pre-order Pipeline (maker_price.py, 14段):
  ├─ spread_adapt: narrow_bps = max(config, δ*_bps) [544#]
  ├─ spread_adapt: boost = base × OFI_scalar [545#]
  └─ ceiling: 0.30 [542#]

CycleGateAggregator.evaluate():
  └─ sidecar:
      ├─ confidence × Toxicity_attenuation [545#]
      ├─ max_boost × δ*_ceiling_scalar [545#]
      └─ shaping: quadratic [546#]

Executor Pipeline (offset_pipeline.py, 9段):
  ├─ EV, Velocity, Trending, Toxicity, Alert: 通常動作
  ├─ VG, Macro: 死亡 (発火率 0%)
  ├─ Sidecar: sidecar_offset_bps 変換
  └─ Final clamp: 0.30 + hard_skip
```

### OFI boost 感度値 (reference)

```yaml
# 546# config 化済み — 現在値 0.5
ofi_boost_sensitivity: 0.5  # 0=無効, 0.5=最大50%増, 1.0=最大100%増
```

### 計測テレメトリ (offset_stages JSON)

```json
{
  "ofi_lite": -0.3421,
  "ofi_mean": -0.1205,
  "as_delta_star": 0.8734,
  "delta_star_bps": 2.15,
  "spread_bps": 2.43
}
```
