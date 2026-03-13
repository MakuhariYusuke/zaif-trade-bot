# 411# Seed 感度の構造的原因分析

**Date**: 2026-03-14
**Phase**: phg (フェーズ横断)
**Type**: rpt (レポート)
**前提**: 410# G3 PASS 深堀り分析

---

## §1 問題定義

reward-clean 4-seed 訓練 (20K steps) で seed456 のみ `reward_profit_corr = -0.20` と逆相関した。
他 3 seed は corr > +0.5 であり、seed 値による結果の散らばりは「偶然」に帰属される。

**本文書の目的**: 偶然要素がどこに存在するかを特定し、構造的に排除 or 緩和する方策を提案する。

---

## §2 乱数注入箇所の網羅的マッピング

SAC 訓練パイプラインにおける全ての乱数消費箇所を特定した。

### §2.1 乱数フロー図

```
seed (e.g. 456)
  │
  ├─[SB3 set_random_seed(seed)]
  │   ├─ random.seed(seed)         ← Python stdlib
  │   ├─ np.random.seed(seed)      ← NumPy global RNG
  │   └─ torch.manual_seed(seed)   ← PyTorch global RNG
  │
  ├─[R1] Policy ネットワーク初期化 (torch RNG)
  │   ├─ Actor: 20→256→256→1×2 (mu, log_std)     71,682 params
  │   ├─ Critic: 21→256→256→1 × 2 (twin Q)       71,681 params × 2
  │   └─ Critic Target: clone of Critic            71,681 params × 2
  │   → torch RNG state: ~215,044 回消費
  │
  ├─[R2] Warm-up 探索 (steps 0 ~ learning_starts-1)
  │   └─ action_space.sample() × 1,000 回        ← np.random (※)
  │   → np.random.uniform(-1, 1) × 1,000 回
  │   → 初期 Replay Buffer の内容を決定
  │
  ├─[R3] Policy Action Sampling (steps 1,000 ~ 19,999)
  │   └─ SAC reparameterization trick             ← torch RNG
  │   → μ(s) + σ(s) × ε, ε ~ N(0,1) (torch.randn)
  │   → 行動選択 → 環境遷移 → Replay Buffer 追加
  │
  ├─[R4] Replay Buffer バッチサンプリング
  │   └─ np.random.choice(buffer_size, batch_size) × 38,000 回
  │   → 勾配更新のデータ構成を決定
  │
  └─[R5] 環境リセット (random_start=False → 影響なし ✅)
      └─ 常に step=0 から開始 → 環境側の乱数消費ゼロ
```

**※ 重要**: SB3 は `set_random_seed(seed)` で Python/NumPy/PyTorch の全グローバル RNG を
**同一の seed 値** でリセットする。R1 (ネットワーク初期化) は torch RNG のみ消費するため、
R2 (warm-up) 開始時の np.random 状態は seed 値にのみ依存する。
つまり **R1 と R2 は独立した乱数ソース** であり、相互干渉しない。

### §2.2 各注入箇所の影響度評価

| # | 箇所 | RNG | 影響 | 影響度 |
|---|------|-----|------|--------|
| R1 | ネットワーク初期化 | torch | 初期重みが学習の出発点を決定。loss landscape 上の位置が seed ごとに異なる | ⭐⭐⭐ **最大** |
| R2 | Warm-up 探索 (1,000 steps) | numpy | 初期 Replay Buffer の内容。最初の勾配更新の質を決定 | ⭐⭐ |
| R3 | Policy Action Sampling | torch | 探索軌道の多様性。R1 からの torch state 継続 | ⭐⭐ |
| R4 | バッチサンプリング | numpy | 勾配方向のバリアンス。38,000 回のサンプリング順序 | ⭐ |
| R5 | 環境リセット | — | random_start=False のため影響なし | — |

---

## §3 過パラメータ化問題（根本原因）

### §3.1 データ対パラメータ比の異常

| 項目 | 値 | 健全基準 |
|------|-----|---------|
| 訓練可能パラメータ | **215,044** | — |
| ユニーク遷移数 (20K steps) | **20,000** | — |
| パラメータ / データ比 | **10.8x** | **< 1.0x** が望ましい |
| 勾配更新回数 | 38,000 | — |
| 各遷移の再利用回数 | ~243 回 | 10-50 回が一般的 |

**結論**: 現在のネットワークは **データに対して 10.8 倍の過パラメータ化**。
これは画像認識における ImageNet 規模のデータがない状態で ResNet-50 を訓練するのに匹敵する。

### §3.2 過パラメータ化がもたらす問題

```
loss landscape (概念図)

215K params, 20K data → 多数の sharp local minima

           ╱╲
     ╱╲   ╱  ╲  ╱╲           ← sharp minima (seed依存)
    ╱  ╲ ╱    ╲╱  ╲ ╱╲
   ╱    ╲      ╲   ╱  ╲
──╱──────╲──────╲─╱────╲──
  seed42  seed456  seed789

各seed → 異なるlocalに落ちる → generalization性能が大きく異なる
```

1. **多数の局所最適**: パラメータ空間が広大で、データが少ないため、同等の訓練損失を達成する重み構成が無数に存在する
2. **鋭い最小値**: 正則化なし (weight_decay=0) のため、鋭い谷に収束しやすい → 汎化性能が不安定
3. **初期値依存性**: 出発点 (R1) が異なると到達する局所最適が異なる
4. **バッチ構成依存性**: 同一データでも再サンプリング順序 (R4) により勾配方向が変わり、異なる谷に導かれる

### §3.3 20K steps の問題

| ステップ数 | 過パラメータ比 | 意味 |
|-----------|--------------|------|
| **20K (現在)** | **10.8x** | データ飢餓。ネットワークは訓練データを丸暗記可能 |
| 50K | 4.3x | 改善だが依然過大 |
| 100K | **2.2x** | 許容範囲に近づくが、[256,256] には不十分 |
| 200K | 1.1x | ようやく均衡 |

---

## §4 seed456 の固有挙動分析

### §4.1 seed456 vs 他 seed のメトリクス比較

| メトリクス | seed42 | seed123 | **seed456** | seed789 | seed456の乖離 |
|-----------|--------|---------|-------------|---------|---------------|
| ROI | +0.69% | +0.64% | **+0.33%** | +0.55% | 中央値の55% |
| PF | 1.198 | 1.116 | **1.089** | 1.174 | 最低だがPASS |
| Sharpe | 5.76 | 6.00 | **3.02** | 5.64 | 中央値の53% |
| corr | +0.54 | +0.56 | **-0.20** | +0.61 | **唯一の負値** |
| mean_reward | 126 | 72 | **-58** | 176 | **唯一の負値** |
| trade_count | 10,749 | 22,481 | **10,699** | 13,071 | 最少 |
| avg_gross | 12.87 | 5.69 | **6.22** | 8.54 | 中程度 |
| best_ckpt | 20K | 5K | **10K** | 10K | 中間 |
| training_time | 2,087s | 1,157s | **2,021s** | 2,275s | 標準的 |

### §4.2 seed456 固有の問題パターン

1. **mean_reward = -58 (唯一の負)**: 報酬の平均が負であるにもかかわらず PnL は正。
   → Actor が「報酬を下げる方向」の行動で利益を出すパターンを学習した
   → **reward signal が PnL と逆方向に機能している**

2. **corr = -0.20**: 報酬が上がると利益が下がる（逆相関）。
   → これは M1 (過パラメータ化) の典型的症状：
   ネットワークが reward の特定の局所パターンを暗記し、
   PnL 方向との整合性なく行動する局所最適に陥った

3. **trade_count = 10,699 (seed42 とほぼ同じ)**: 取引数は正常。
   → 問題は「いつ取引するか」ではなく「どの方向に取引するか」

### §4.3 OOS チェックポイント推移の異常

| Timesteps | seed456 OOS | 他seed平均OOS |
|-----------|-------------|---------------|
| 5K | +0.45% | +0.08% |
| 10K | **+0.47%** ★ | +0.21% |
| 15K | +0.37% | -0.04% |
| 20K | +0.44% | +0.18% |

**意外な発見**: seed456 は OOS ROI が全チェックポイントで正であり、他 seed の平均より良い。
にもかかわらず corr=-0.20 で mean_reward=-58 → **PnL は出ているが学習が正しい方向に進んでいない**。

**解釈**: seed456 のネットワーク初期化が、報酬信号を無視しつつ市場の統計的エッジ（モメンタム等）を
偶然拾う重み構成に到達した。学習は reward に沿っていないが、初期重みの偶然により PnL は正。
これは **再現性のない運** であり、100K 訓練で reward 方向に矯正される可能性と崩壊する可能性の両方がある。

---

## §5 構造的対策の提案

### §5.1 即効性のある対策 (100K 訓練前に実施可能)

#### M1: ネットワーク縮小 (`net_arch` パラメータ化)

| アーキテクチャ | パラメータ数 | @20K 比率 | @100K 比率 | 推奨 |
|--------------|------------|----------|----------|------|
| [256,256] (現行) | 215,044 | 10.8x | 2.15x | — |
| **[128,128]** | **58,372** | **2.9x** | **0.58x** | ✅ **100K なら最適** |
| [64,64] | 16,900 | 0.8x | 0.17x | ✅ 20K 短期実験向け |
| [32,32] | 5,380 | 0.3x | 0.05x | ⚠️ capacity 不足リスク |

**実装**:
1. `g2_sac_reward_clean.yaml` に `policy_kwargs.net_arch` を追加
2. `_create_sac_model()` で `policy_kwargs` を SAC コンストラクタに転送

```yaml
# g2_sac_reward_clean.yaml に追加
sac_hyperparameters:
  policy_kwargs:
    net_arch: [128, 128]  # 215K → 58K params
```

#### M2: Weight Decay 追加

```yaml
sac_hyperparameters:
  policy_kwargs:
    optimizer_kwargs:
      weight_decay: 1.0e-4  # L2 正則化 → 平坦な最小値への誘導
```

**効果**: 鋭い局所最適を回避し、seed 間の収束先を近づける。
**リスク**: 制約が強すぎると表現力低下。1e-4 は保守的な値。

#### M3: `learning_starts` 増加 (1,000 → 5,000)

**効果**: 初期 Replay Buffer に 5 倍のデータが蓄積されてから学習開始。
最初の勾配更新がより多様なデータに基づく → seed 依存性の軽減。

**コスト**: 20K 訓練では学習可能ステップが 19K → 15K に減少 (21% 削減)。
100K 訓練では 99K → 95K (5% 削減) で影響軽微。

### §5.2 100K 訓練時の対策

#### M4: Seed 数増加 (4 → 8)

```yaml
seeds: [42, 123, 456, 789, 0, 1, 2, 3]
```

**効果**: 異常 seed の統計的検出力向上。8 seed 中 1 seed 異常は 12.5% (現在は 25%)。
**コスト**: 訓練時間 2 倍。1 seed ≈ 2,000 秒 → 8 seed ≈ 16,000 秒 (4.4 時間)。

#### M5: Checkpoint Ensemble

単一 best checkpoint ではなく、OOS ROI が正の全 checkpoint の重み平均を最終モデルとする。

**効果**: 単一チェックポイントへの依存を排除。seed789 型の「best 以外全て OOS 負」問題を緩和。
**実装**: `model.policy.state_dict()` の加重平均。SB3 API で実現可能。

### §5.3 対策の優先順位

| 優先度 | 対策 | 実装コスト | 期待効果 | 100K前に実施 |
|--------|------|----------|---------|-------------|
| **P0** | M1: net_arch [128,128] | 低 (YAML+コード1箇所) | ⭐⭐⭐ | ✅ |
| **P0** | M2: weight_decay 1e-4 | 低 (YAML追加) | ⭐⭐ | ✅ |
| P1 | M3: learning_starts 5000 | 低 (YAML変更) | ⭐⭐ | ✅ |
| P1 | M4: seeds 8個 | なし (YAML変更) | ⭐⭐ | ✅ |
| P2 | M5: Checkpoint Ensemble | 中 (コード追加) | ⭐ | △ |

---

## §6 推奨実験計画

### §6.1 Phase 1: 高速検証 (20K × 8 seeds)

100K 訓練の前に、対策の有効性を 20K で検証する。

```
実験A (baseline): 現行設定         [256,256], wd=0,    ls=1000, 4 seeds
実験B (M1 only):  net_arch 縮小    [128,128], wd=0,    ls=1000, 4 seeds
実験C (M1+M2):    net_arch + wd    [128,128], wd=1e-4, ls=1000, 4 seeds
実験D (M1+M2+M3): full mitigation  [128,128], wd=1e-4, ls=5000, 4 seeds
```

**判定基準**: corr の seed 間標準偏差が baseline より小さければ対策有効。

### §6.2 Phase 2: 本番 (100K × 8 seeds)

Phase 1 で最良の設定を採用し、100K × 8 seeds で G3 再評価。

---

## §7 code 変更計画

### 変更 1: `_create_sac_model()` に `policy_kwargs` 転送

```python
# scripts/v460/lib/tasks/sac_train.py
def _create_sac_model(env, sac_cfg, seed):
    from stable_baselines3 import SAC
    
    policy_kwargs = sac_cfg.get("policy_kwargs", {})  # NEW
    
    model = SAC(
        "MlpPolicy",
        env,
        ...,
        policy_kwargs=policy_kwargs if policy_kwargs else None,  # NEW
        seed=seed,
    )
    return model
```

### 変更 2: YAML 設定追加

```yaml
sac_hyperparameters:
  policy_kwargs:
    net_arch: [128, 128]
    optimizer_kwargs:
      weight_decay: 1.0e-4
```

### 変更 3: learning_starts の外部化

```yaml
sac_hyperparameters:
  learning_starts: 5000  # (既にYAMLで設定可能)
```

---

## Appendix A: SB3 SAC seed 伝播の完全フロー

```
SAC(seed=456)
  └─ OffPolicyAlgorithm.__init__(seed=456)
       └─ BaseAlgorithm.__init__(seed=456)
            └─ self.seed = 456

SAC._setup_model()
  └─ OffPolicyAlgorithm._setup_model()
       ├─ self._setup_lr_schedule()
       ├─ self.set_random_seed(456)     ← ★ここで全RNGを456に設定
       │    ├─ random.seed(456)
       │    ├─ np.random.seed(456)
       │    └─ torch.manual_seed(456)
       │
       ├─ self.replay_buffer = ReplayBuffer(...)   ← RNG消費なし
       │
       └─ self.policy = SACPolicy(...)              ← ★torch RNGを~215K回消費
            ├─ Actor(20→256→256→1×2)
            ├─ Critic(21→256→256→1 × 2)
            └─ CriticTarget = copy(Critic)

model.learn(20000)
  └─ collect_rollouts()
       ├─ steps 0-999: action_space.sample()        ← np.random 消費
       └─ steps 1000-19999: policy.predict(obs)      ← torch RNG (reparameterization)
  └─ train() × 38,000 回
       └─ replay_buffer.sample(128)                  ← np.random 消費
       └─ SGD update                                  ← 決定論的 (勾配計算)
```

## Appendix B: データソース

- 訓練コード: `scripts/v460/lib/tasks/sac_train.py`
- SAC モデル作成: `_create_sac_model()` (L395-413)
- SB3 seed 設定: `stable_baselines3.common.utils.set_random_seed()`
- ネットワーク構築: `SACPolicy._build()` → Actor + Critic + CriticTarget
- Replay Buffer: `stable_baselines3.common.buffers.ReplayBuffer`
- 環境: `ztb/trading/environment/heavy_env/core.py` (random_start=False)
- seed 管理: `ztb/utils/seed_manager.py` (訓練パイプラインでは未使用、SB3独自seed)
