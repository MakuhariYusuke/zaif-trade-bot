# 105# Phase D 計画: Gate 2 突破に向けた次フェーズ戦略

**Date**: 2026-02-10  
**前提ドキュメント**: 0# (プロジェクト計画), 66# (整合性レビュー), 91# (方針転換), 101# (再利用提案), 103# (Phase Cレビュー), 104# (Phase C結果)  
**目的**: Phase C (C0'→C3) の実験結果を踏まえ、Gate 2 突破に向けた Phase D の具体的実行計画を策定する  
**提出先**: Codexレビュー

---

## §0 結論（先に要点）

1. Phase C で **DrawdownController ラッチバグ** を特定・修正し、eval_trades=0 問題を解決した
2. C3 最良条件 (ent001+thr70+nodd) で **WinRate=49.2% (✅PASS)、MaxDD=0.06% (✅PASS)** を達成
3. しかし **PF=0.990、Sharpe=-1.59、ROI=-4.38%** は Gate 2 未達。最大ボトルネックは **手数料支配** (fees/gross=4.2x)
4. 91# の仮説 H1 (γ=0.80) は **C3 で否定済み** — γ=0.80 は γ=0.99 より劣化。91# Phase A は完了
5. 91# の仮説 H2 (コスト支配) は **Phase C 全体で strong confirm** — gross PnL はプラスだが fees で消滅
6. 次フェーズ (Phase D) は **予測精度の本質的向上** と **コスト構造の最適化** を同時に攻める
7. 101# §4 の C0→C1→C2→C3→C4 マップに対し、C0~C1 は実質完了、C2 (約定モデル現実化) は未着手、C3 (報酬/ハイパラ) は部分完了、C4 (OOS) は次段階

---

## §1 Phase C 実績の要約

### §1.1 Gate 2 基準との距離

| KPI | Gate 2 基準 (0# §5.2) | Phase C 最良 (C3 thr70_nodd) | Gap | 状態 |
|-----|----------------------|------------------------------|-----|------|
| ROI | >5% | -4.38% | 9.4pp | ❌ |
| PF | >1.20 | 0.990 | 0.21 | ❌ |
| Sharpe | >1.0 | -1.59 | 2.59 | ❌ |
| WinRate | >35% | 49.2% | - | ✅ |
| MaxDD | <15% | 0.06% | - | ✅ |

### §1.2 91# 仮説検証結果

| 仮説 | 当初優先度 | Phase C 結果 | 判定 |
|------|-----------|-------------|------|
| H1: γ=0.80 が最適 | ⭐⭐⭐ | C3: γ=0.80 でPF=0.939、γ=0.99 でPF=0.990 | **否定** (γ=0.99 が優勢) |
| H2: コストが利益を食う | ⭐⭐⭐ | C1': gross_pnl=+5,833 vs fees=20,882 (3.6x) | **強く確認** |
| H3: ペナルティ過多 | ⭐⭐ | P1 で否定済み (91#記載), Phase C はPnL-only | 否定済み |
| H4: 行動空間が複雑 | ⭐⭐ | 1D action は維持。threshold 制御が有効 | 部分確認 |
| H5: 強シグナル不信頼 | ⭐ | threshold 引上げで WinRate 改善 → 高確度シグナル選別が有効 | **部分確認** |

### §1.3 101# 実行マップとの対応

| 101# Phase | 計画内容 | Phase C での達成度 | 残課題 |
|------------|---------|-------------------|--------|
| C0: 計測統一 | KPI算出統一、Gross/Net分離 | ✅ Gate2 メトリクス実装済み | window分割ベースライン比較未実施 |
| C1: コスト圧縮 | threshold/holding最適化 | ✅ threshold sweep 完了 (0.33→0.70) | 動的閾値 (z-score) 未検証 |
| C2: 約定モデル現実化 | realistic.py, pseudo_hft.py | ❌ 未着手 | 約定遅延/スリッページ感度未評価 |
| C3: 報酬/ハイパラ再現 | γ, loss_weight, V457Reward | △ γ=0.80 検証済、loss_weight/V457Reward未検証 | 非対称報酬 (loss×1.2) 未実験 |
| C4: OOS最終判定 | 4split walk-forward | ❌ 未着手 | IS 条件が Gate 2 未達のため未移行 |

### §1.4 66# 指摘事項との対応

| 66# 指摘 | 現状 | Phase D 対応 |
|----------|------|-------------|
| Gate 2 の全 KPI を計測していない | ✅ Phase C で全 KPI 計測実装 | 継続 |
| 統計検定が不十分 | △ seed=42 のみでスクリーニング | D3 で 4-seed 拡張 |
| seed数が 1 では信頼不十分 | △ スクリーニング段階 | D3 で対応 |
| コスト計上の二重/曖昧さ | ✅ Gross/Net 分離済み | 継続 |
| OOS 評価の欠如 | ❌ phase C は IS のみ | D4 で walk-forward 導入 |

---

## §2 構造的問題の診断

### §2.1 手数料支配の構造

```
C3 best (thr70_nodd):
  309 trades / 50,000 steps
  avg_gross_pnl_per_trade = +4.44 JPY
  avg_fee_per_trade       = 18.86 JPY
  cost_ratio              = 424%
  → 1取引あたり gross edge は正だが fee の 1/4 以下
```

**Gate 2 PF=1.20 の逆算**:

$$PF = \frac{\sum W}{\sum L + \sum F} \geq 1.20$$

現在の fee 構造 (0.1% 往復) で PF≥1.20 を達成するには:
- **avg_gross_pnl_per_trade ≥ avg_fee × (PF/(PF-1)) ≈ 113 JPY** (現在 +4.44)
- つまり現在の **約25倍** の per-trade edge が必要
- あるいは fee を 1/25 に削減 (0.004% 往復 = 非現実的)

**結論**: threshold/holding 調整だけでは到達不可能。**予測精度 (per-trade edge) の本質的向上** が必須。

### §2.2 WinRate ~49% の意味

- 方向予測精度がコイン投げ同等
- PF=0.990 → 勝ちと負けのサイズもほぼ同等
- **SACは現在の観測空間 (RSI×7 + ReturnStdDev) から有意なアルファを抽出できていない可能性**

### §2.3 50K timesteps の学習量

- 0# §2.1 は v451 の成功を γ=0.80 に帰属 → C3 で否定
- 50K steps で 309 eval trades → 学習データとしての取引経験が少ない
- 200K-500K への拡大で precision が向上する余地あり（ただし構造問題の解決が先）

---

## §3 Phase D 計画

### §3.1 設計原則

1. **0# §3.1 の "No New Features" 原則を継続**: 既存実装の再利用を最優先
2. **101# §4 の C2→C3→C4 順序を踏襲**: 未着手の約定モデル現実化 → 報酬微調整 → OOS
3. **66# の統計検定・seed数指摘に対応**: 改善条件のみ 4-seed + walk-forward へ拡張
4. **91# H2 (コスト支配) を最重要仮説として維持**: per-trade edge の拡大が最大レバー
5. **猜疑心を以て** 各段階で「Random超過」を検証（101# §1.1 統計独立性に注意）

### §3.2 Phase D ロードマップ

```
Phase D0: 学習量拡大 (50K→200K)
         → per-trade edge の変化を観察
         → 予測精度向上の余地を判定 (CRITICAL GATE)

Phase D1: 特徴量強化
         → 現 8特徴 (RSI偏重) → 情報量追加
         → 既存 FeatureRegistry の未使用特徴を選択的に投入

Phase D2: 報酬関数微調整
         → v451 非対称報酬 (loss×1.2) の検証
         → 学習時 transaction_cost 感度の検証

Phase D3: Multi-seed 検証
         → 勝ち筋条件のみ 4-seed 展開
         → Mann-Whitney exact test で Random 超過を判定

Phase D4: OOS 最終判定
         → walk-forward 4-split
         → Gate 2 全 KPI で GO/NO-GO 判定
```

### §3.3 Phase D0: 学習量拡大 (CRITICAL GATE)

**仮説**: 50K steps では学習量が不足 → 200K で per-trade edge が有意に改善する

**実験定義**:

| 実験名 | timesteps | ent_coef | threshold | eval_dd | 目的 |
|--------|-----------|----------|-----------|---------|------|
| d0_200k_thr70 | 200,000 | 0.01 | 0.70 | 1.0 (無効) | 学習量 4x の効果 |
| d0_200k_thr80 | 200,000 | 0.01 | 0.80 | 1.0 | + 閾値引上げ |
| d0_500k_thr70 | 500,000 | 0.01 | 0.70 | 1.0 | 学習量 10x の限界 |
| d0_50k_thr70 (ref) | 50,000 | 0.01 | 0.70 | 1.0 | C3 baseline (既存) |

**評価 KPI**: `avg_gross_pnl_per_trade`, `PF`, `WinRate`, `Sharpe`

**Gate 判定**:
- **GO**: 200K で avg_gross_pnl_per_trade が 50K の 2x 以上 → D1 へ進む
- **CONTINUE**: 改善あるが 2x 未満 → 500K も含めて D1 並行
- **NO-GO**: 200K でも改善なし → §4 代替戦略に分岐

**根拠**: C3 で gross_pnl は +1,373 (309 trades)。学習量拡大で direction prediction accuracy が 49.2%→55% に改善すれば、per-trade edge は大幅に拡大し PF>1.0 が射程に入る。

### §3.4 Phase D1: 特徴量強化

**仮説**: RSI×7 + ReturnStdDev の 8特徴では情報量が不足 → 追加特徴で予測精度向上

**実施手順** (101# §B の既存資産活用):
1. `FeatureRegistry.list()` で利用可能な全特徴を棚卸し
2. `ztb/benchmarks/ablate_features.py` で特徴量アブレーション (101# §2.B)
3. 候補: ボリンジャーバンド幅、ATR、MACD、Volume変化率 等
4. 相関フィルタ (correlation_threshold=0.95) で冗長排除
5. 12-16 特徴の候補セットで D0 best 条件を再実験

**Gate 判定**:
- PF 改善 > 0.05 → 採用して D2 へ
- 改善なし → D0 best のまま D2 へ

### §3.5 Phase D2: 報酬関数微調整

**実験定義** (91# Phase C + 101# §3.1):

| 実験名 | reward 変更 | 目的 |
|--------|------------|------|
| d2_asymmetric | loss × 1.2 | v451 非対称報酬の効果 (91# H1 の報酬側面) |
| d2_cost_aware | 学習時 cost=0.005, eval時 cost=0.001 | コスト感度を明示学習 |
| d2_combo | loss×1.2 + cost=0.005 | コンボ |

**注意** (101# §3.4, 0# §3.1):
- v456 の教訓: penalty 項を増やさない。PnL-only の微調整に留める
- `V457RewardCalculator` は 101# §2.B で条件付き採用 → Phase D2 で検証素案としては残すが、優先度は低

### §3.6 Phase D3: Multi-seed 検証

**実施条件**: D0-D2 で **PF ≥ 1.05** かつ **ROI ≥ 0%** の条件が 1 つ以上存在
- 4 seeds: [42, 123, 456, 789]
- 101# §1.1 の指摘に従い、deterministic baseline は seed ではなく **window 分割** で比較
- 統計検定: Mann-Whitney (exact) + Cliff's delta + Holm 補正 (gate_c3_comparison.py 活用)

### §3.7 Phase D4: OOS 最終判定

**実施条件**: D3 で 4-seed 中 **3/4 以上が PF>1.0** かつ Random 超過が有意 (p<0.10)
- `ztb/evaluation/walk_forward/splitter.py` で 4-split (embargo 付き)
- 4 seeds × 4 splits = 16 条件
- Gate 2 全 KPI で最終 GO/NO-GO

---

## §4 代替戦略 (D0 NO-GO 時の分岐)

Phase D0 で学習量拡大が効果なしの場合、**SACの観測空間→行動変換能力の限界** と判断し、以下に分岐する。

### §4.1 分岐 A: 時間軸変更

0# §2 の "1分足" 前提を見直し:
- **5分足**: コスト/edge 比が改善 (取引あたり利益幅が拡大)
- **15分足**: 更にノイズ削減 → 方向予測精度向上の可能性
- 既存データから `resample()` で生成可能 (新規データ取得不要)

### §4.2 分岐 B: 離散行動空間

103# §3.3 の提案:
- 連続行動 (SAC) → 離散 (DQN-like: BUY/SELL/HOLD)
- 閾値変換のヒューリスティクスを排除 → よりクリアな学習信号
- `sb3_contrib.RecurrentPPO` 等の選択肢も

### §4.3 分岐 C: ハイブリッド戦略

103# §3.4:
- ML による方向予測 (分類モデル) + ルールベース執行
- RL の連続行動空間の難しさを回避
- 既存 `ztb/features/` をそのまま流用

### §4.4 分岐判定基準

| 条件 | 判定 |
|------|------|
| D0 で avg_gross_pnl 改善なし + 特徴量拡大でも改善なし | → 分岐 A (時間軸変更) |
| D0 で gross は改善するが threshold 変換がボトルネック | → 分岐 B (離散行動) |
| 上記いずれでも Gate 2 未達 | → 分岐 C (ハイブリッド) |

---

## §5 リスクと制約

### §5.1 計算資源

| Phase | 見込み計算時間 | 備考 |
|-------|--------------|------|
| D0 | 200K: ~40分, 500K: ~100分 | GPU なし (CPU SAC) |
| D1 | ~30分 (特徴量計算 + 再学習) | parquet 再生成含む |
| D2 | ~90分 (3実験 × 200K) | D0 best timesteps 準拠 |
| D3 | ~6時間 (4-seed × best条件) | 並列化で短縮可能 |
| D4 | ~24時間 (4-seed × 4-split) | batch実行 |

### §5.2 前提条件

- 学習データ: `data/btc_jpy_1m_v451_optimized_features.parquet` (2026-02-10 更新済み, 1,216,930行)
- 環境: HeavyTradingEnv (既存のまま、eval_dd_threshold=1.0 で DD 停止無効化)
- テスト: `tests/unit/scripts/test_run_phase_c.py` (38テスト PASS 維持)

### §5.3 既知リスク

| リスク | 影響度 | 緩和策 |
|--------|-------|--------|
| 200K でも予測精度が改善しない | 高 | §4 代替戦略を事前に設計済み |
| 過学習 (IS のみで判断) | 高 | D3-D4 で multi-seed + OOS |
| 1分足の情報密度限界 | 中 | 分岐 A で時間軸変更 |
| 特徴量追加が noise を増幅 | 中 | アブレーション + 相関フィルタで管理 |

---

## §6 成功基準のまとめ

### §6.1 Phase D 完了条件 (Gate 2)

0# §5.2 の基準を再掲:
- **ROI > 5%** (net, 手数料込み)
- **PF > 1.20**
- **Sharpe > 1.0**
- **MaxDD < 15%** (✅ Phase C で達成済み)
- **WinRate > 35%** (✅ Phase C で達成済み)

### §6.2 Phase D 中間 Gate

| Gate | 条件 | 判断 |
|------|------|------|
| D0-Gate | 200K avg_gross_pnl ≥ 50K の 2x | GO / CONTINUE / NO-GO |
| D1-Gate | 特徴量拡大で PF 改善 > 0.05 | 採用/不採用 |
| D2-Gate | PF ≥ 1.05 かつ ROI ≥ 0% | D3 multi-seed へ |
| D3-Gate | 4-seed 中 3/4 で PF>1.0 + Random有意超過 | D4 OOS へ |
| D4-Gate | OOS で Gate 2 全 KPI PASS | **本番移行 GO** |

---

## §7 Codex へのレビュー依頼事項

1. **Phase D0 の Gate 判定基準** (2x 改善) は合理的か？より厳しい/緩い基準がありうるか？
2. **91# H1 否定** (γ=0.80 < γ=0.99) の解釈 — v451 との環境差を考慮すべきか？
3. **§2.1 の PF≥1.20 逆算** — この条件下で 1分足 SAC が到達可能と判断するか？
4. **特徴量強化** (D1) vs **学習量拡大** (D0) の優先順位は適切か？
5. **代替戦略** (§4) の分岐条件は論理的に十分か？
6. **101# の "C2: 約定モデル現実化" を Phase D に含めなかった** — これは現段階では早すぎるという判断に同意するか？
7. **本プロジェクトの大義 (短期間での高収益性)** に対して、Phase D のタイムラインは許容範囲か？

---

## §8 参照ドキュメント一覧

| Doc# | タイトル | Phase D での参照箇所 |
|------|---------|---------------------|
| 0# | Project Proposal v459 | Gate 2 基準 (§5.2), 設計原則 (§3.1), vXXX 教訓 (§2) |
| 66# | 00# 整合性チェック | 統計検定不足、seed数不足、コスト計上曖昧さ |
| 91# | 方針転換計画 | H1-H5 仮説マトリクス、Phase A/B/C 構造 |
| 101# | Phase 4.5 フォローアップ | 再利用資産マップ (A/B/C), C0-C4 実行順 |
| 102# | Phase C 実験ログ | C0+C1 の 14実験 NO-GO 結果 |
| 103# | Phase C レビュー | VecNormalize + DD ラッチ根本原因 |
| 104# | Phase C 総合レポート | C0'→C3 全結果、Gate 2 距離分析 |
