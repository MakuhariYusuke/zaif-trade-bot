# 392# 389/390 レビュー妥当性評価

## 概要

389# (Codex rev) と 390# (Gemini second opinion) の主張を、コード・artifact の実物に基づいて妥当性を判定する。

---

## 評価基準

| 記号 | 意味 |
|:----:|------|
| ✅ | 主張が実証され、対応が必要 |
| ⚠️ | 主張は概ね正しいが、重要度または解釈に留意点あり |
| ❌ | 主張が事実と異なる、または対応不要 |

---

## 389# (Codex rev) の評価

### P0-1: 保存 artifact が FAIL のまま → ✅ 妥当

**検証結果**:

```
results/v460/v460_g2train_seed42_20260312_073155.json
└─ g2_judgment_cache.gate_result = "FAIL"
└─ g2_judgment_cache.checks.worst_seed_roi.threshold = -0.02  ← 旧閾値
└─ g2_judgment_cache.checks.worst_seed_roi.pass = false
```

一方、`configs/v460/gate_thresholds.yaml` では `worst_seed_min_roi: -0.035` に更新済み。
`run_gate_check.py --gate G2` で再判定すると **PASS** になることも確認。

**評価**: artifact と docs の不整合は事実。
「G2 PASS」の主張自体は current policy 上は成立するが、保存 artifact が FAIL のままでは
後続ツールや外部レビューで混乱する。

**対応**: reward-tuned の G2 判定を current thresholds で再保存する。

### P0-2: reward-tuned は最良候補ではなく Gate 適合候補 → ✅ 妥当

**検証結果**:

| 実験 | Mean ROI | G2 |
|------|:--------:|:--:|
| baseline | **+1.39%** | FAIL |
| γ=0.95 | +0.74% | FAIL |
| reward-tuned | +0.32% | PASS |
| warm-start | -2.65% | FAIL |

**評価**: 完全に妥当。平均 ROI で最高はベースライン (+1.39%)。
reward-tuned が PASS したのは seed 間分散が収まり E4 緩和閾値内に入ったからであって、
収益最大化の結果ではない。

388# が reward-tuned を「確定モデル」として扱っていた点は修正すべき。
G3 候補は reward-tuned 一本に固定せず、baseline/γ=0.95 も同一指標で比較すべき。

### P0-3: reward と PnL の相関が reward-tuned で反転 → ✅ 妥当 (最重要)

**検証結果** (artifact から直接計算):

| Seed | gross_pnl | mean_reward | trades |
|------|----------:|------------:|-------:|
| 42 | +1,226 | +335,640 | 187,303 |
| 123 | +39,345 | +81,719 | 120,215 |
| 456 | -320,273 | +103,157 | 116,815 |
| 789 | +409,345 | -9,243 | 312,925 |

```
corr(gross_pnl, mean_reward) = -0.3783
```

他実験との比較:

| 実験 | corr(PnL, reward) | avg trades |
|------|:-----------------:|-----------:|
| baseline (163707) | +0.91 | 1,242 |
| baseline (202003) | +0.89 | 44,254 |
| **reward-tuned** | **-0.38** | **184,314** |
| warm-start | +0.95 | 232,462 |

**評価**: これは389# で最も重要かつ正確な指摘。
reward-tuned だけが負相関に反転している。つまり「reward を最大化してもPnLが増えない」
構造になっている。ペナルティ削減が reward を膨張させたが、それは PnL と無関係な
行動（バランス調整、エントロピーボーナス等）に由来する可能性が高い。

同時に、trade 数が baseline の **4.2 倍** に急増している点も重要。
fee=0 環境では取引増加がコスト化されず、ROI は見かけ上プラスになりうるが、
これは実環境では friction (queue miss, adverse selection) で崩壊する。

**対応**: G3 前に `reward_profit_alignment` 指標を必須化する。
最低条件: `corr(mean_reward, gross_pnl) > 0`。

### P1-1: `g3_gate_check.py` 新設は二重化 → ✅ 妥当

**検証結果**: `scripts/v460/run_gate_check.py` L329-404 に `run_g3_judgment()` が既に存在。

期待入力: `seed_metrics: [{pf, sharpe_annual, max_drawdown, avg_gross_per_trade, avg_fee_per_trade}, ...]`

一方、`scripts/v460/lib/sac_common.py` の `evaluate_model_oos()` は
`{gross_roi, mean_reward, trade_count, n_episodes, gross_pnl}` のみ出力。

**評価**: 388# が提案した `g3_gate_check.py` 新設は不要。
必要なのは `evaluate_model_oos()` の出力拡張（PF, Sharpe, MaxDD, equity curve の追加）であり、
それを既存 `run_g3_judgment()` に渡す接続コード。

388# のこの部分は撤回し、389# の提案に従う。

### P1-2: cost model で実利益を過大評価 → ⚠️ 概ね妥当だが優先度に疑問

**評価**: 理論的には正しい。maker 0% でも queue miss, adverse selection, 約定遅延は実コスト。
しかし現時点では G3 Gate 自体に到達していない。
stress 条件（slippage 1tick 等）の追加は重要だが、**まず基本 G3 を通してから**が現実的。

**対応**: G3 本判定を先に実施。PASS 後に stress 条件を追加検証する（G3.1 枝番として）。

### P1-3: gamma=0.99 / 500K は今の最優先ではない → ✅ 妥当

**評価**: 389# の優先順位は正しい。固めるべきは:
1. reward-profit alignment
2. G3 計測出力の一本化
3. artifact 判定の整合

gamma=0.99 / curriculum / 500K は、これらが揃った後に same-metric で比較する。

### P2-1: SB3 cleanup は完了扱い早い → ⚠️ 注意だが低優先

**評価**: 事実として `_sb3_test_stub/`, `sb3_compat.py`, `conftest.py` の
SB3 fallback コードは残存している。ただしこれは機能に影響しておらず、
G3 進行のブロッカーではない。conftest detox は別セッションで。

### P2-2: 文書・設定の追跡性に drift → ✅ 妥当

**検証結果**:

```
configs/v460/experiments/g2_sac_gamma095_reward_tuned.yaml
├─ L13: "根拠: docs/v460/386_reward_analysis.md"  ← 旧名
├─ L59: "386# 分析: PnL reward O(1-50 JPY)..."   ← 旧番号
└─ L2:  "387# (予定)"                             ← 387# だが小文字で「予定」
```

**対応**: YAML コメントの `386` 参照を `387` に更新する。

### P2-3: 既存実装の selective reuse → ✅ 妥当

**評価**: `reward_function_evaluator` は deprecated 寄りで、そこに全面依存すべきではない。
型・指標関数の再利用 + gate 判定は既存 `run_gate_check.py` に統一が正解。

---

## 390# (Gemini second opinion) の評価

### 2.1 評価環境 (checkpoint_eval_env) の分離 → ✅ 事実確認

**評価**: Ghost State バグの修正は `sac_train.py` で明示的に確認可能。異論なし。

### 2.2 設定監査と手数料修正 → ✅ 事実確認

**評価**: `transaction_cost: 0.0` + `reward_scaling: 1.0` は YAML 上で確認済み。
「トレードしないのが最善と学習する根因が取り除かれた」という表現は的確。

### 2.3 reward ペナルティチューニング → ⚠️ 表現に注意

**評価**: 390# は「エージェントが利益追求ではなくペナルティ回避行動に走る構造を是正した」と
肯定的に記述しつつ、「G2 通過は制約緩和の結果であり、エッジの証拠ではない」と警告している。
この二面性は正しい。ただし 389# P0-3 の corr=-0.38 問題には触れていない。

### 2.4 G3 移行計画 → ⚠️ 方向は正しいが具体性不足

**評価**: 390# は「独自の集計コードを新設するな」と明確に警告しているが、
具体的にどの既存コードを使うべきかの指示は 389# のほうが詳しい。

### 「実証テストの決行」提案 → ❌ タイミングが早い

**評価**: 390# は「フルステップ 500K〜1M で走らせてください」「gamma=0.99 以上での収束を見届けてください」と提案しているが、
389# P1-3 が正しく指摘する通り、**現時点で先に固めるべきは reward-profit alignment と G3 計測の一本化**。
reward と PnL が負相関のまま 1M steps を回しても、「long training but wrong objective」になるだけ。

### 易占について

雷火豊（☳ 雷 / ☲ 火）: 「明るさと動き、豊かさの極み。ただし盛りの後には陰りが来る」。
変爻なしは **純卦** = 卦意そのまま。

プロジェクトの現在地に対する解釈:
- **明察** = パイプラインバグの発見と修正 (384#-385#)
- **決断** = 修正後の reward-tuning と G2 PASS 達成 (387#)
- **豊の極み** = G2 PASS を起点に G3 に進もうとしている現在
- **盛りの後の陰り** = reward-profit alignment が壊れている事実 (389# P0-3)

易の示唆は「今は動ける時だが、明察（問題の正確な把握）を失うと衰退する」。
389# の P0-3 指摘はまさにこの「明察」に相当する。
**reward が profit を向いていない構造のまま突き進めば、「豊」は「幻の豊」に終わる。**

---

## 総合判定

### 389# (Codex)

| 指摘 | 妥当性 | 対応要否 |
|------|:------:|:-------:|
| P0-1 artifact FAIL | ✅ | 要: 再保存 |
| P0-2 最良候補ではない | ✅ | 要: 388# 前提修正 |
| P0-3 reward-PnL 負相関 | ✅ | **最優先** |
| P1-1 G3 checker 二重化 | ✅ | 要: 388# 撤回 |
| P1-2 cost model 楽観 | ⚠️ | G3 PASS後に対処 |
| P1-3 gamma/500K は後回し | ✅ | 同意 |
| P2-1 SB3 cleanup | ⚠️ | 低優先 |
| P2-2 drift | ✅ | 要: YAML修正 |
| P2-3 selective reuse | ✅ | 同意 |

### 390# (Gemini)

| 指摘 | 妥当性 | 対応要否 |
|------|:------:|:-------:|
| 2.1 eval env 分離 | ✅ | 確認済み |
| 2.2 手数料修正 | ✅ | 確認済み |
| 2.3 reward ペナルティ | ⚠️ | P0-3 補足要 |
| 2.4 G3 移行 | ⚠️ | 389# のほうが具体的 |
| 500K/gamma=0.99 提案 | ❌ | 時期尚早 |
| DRY 原則警告 | ✅ | 同意 |

### 389# vs 390# の質的比較

| 観点 | 389# (Codex) | 390# (Gemini) |
|------|:------------:|:-------------:|
| コード実証の深さ | ◎ artifact 直接検証 | ○ コード片引用 |
| 構造的問題の発見 | ◎ P0-3 (corr=-0.38) | △ 未指摘 |
| 実行可能性 | ◎ 既存コード流用を推奨 | △ 「走らせてください」止まり |
| リスク認識 | ◎ reward hacking を具体化 | ○ 「エッジの証拠ではない」 |
| 鼓舞・士気面 | △ 冷静な分析に徹する | ◎ 「完全に正しい道」 |

**結論**: 389# のほうが技術的に深く正確。390# は方向性の確認とモチベーション面で補完的。
ただし 390# の「500K/γ=0.99 を今すぐ」は 389# P1-3 と矛盾しており、389# を採用する。

---

## 次アクション（389# 準拠で更新）

### 即座 (P0)

1. **reward-profit alignment 計測**: `evaluate_model_oos()` に `reward_profit_correlation` を追加
2. **artifact 再保存**: reward-tuned の G2 判定を current thresholds で再保存
3. **YAML drift 修正**: `g2_sac_gamma095_reward_tuned.yaml` の `386#` 参照を `387#` に更新

### 短期 (P1)

4. **evaluate_model_oos 拡張**: PF, Sharpe, MaxDD, equity_curve を出力に追加
5. **G3 接続**: 拡張した evaluator → 既存 `run_g3_judgment()` への接続
6. **横並び比較**: baseline / γ=0.95 / reward-tuned を同一 G3 指標で評価

### 中期 (P2)

7. **stress 条件追加**: slippage 1tick, maker miss penalty の感度分析
8. **gamma=0.99 / 500K**: alignment が正の状態で拡大訓練
9. **SB3 cleanup / conftest detox**: 別セッション
