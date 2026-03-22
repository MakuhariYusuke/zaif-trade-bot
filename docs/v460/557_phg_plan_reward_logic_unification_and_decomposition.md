# 557# [phg] [plan] 報酬計算ロジックの一元化と RewardCalculator の機能分解 (深掘り版)

> **ステータス**: 計画・設計 + 一部実装反映済み  
> **更新日**: 2026-03-23  
> **作成者**: Gemini CLI Agent

---

## 1. 核心的な目的：ロジックの「ztb 委譲」

`scripts/v460` 以下の実験用コード（特に `LiteTradingEnv`）における独自実装を廃止し、すべての報酬計算ロジックを `ztb` パッケージ側に委譲する。これにより、実験環境と本番環境の報酬定義が「100% 同一」であることを保証する。

### 1.1 独自実装の排除
- `LiteTradingEnv.step()` 内での PnL 計算やペナルティ計算を、すべて `RewardKernel` への呼び出しに置き換える（Phase 2 は実施済み）。
- `RewardCalculator.calculate_reward_simple()` も `RewardKernel` 委譲へ前進済みであり、Lite/Heavy の単純報酬系の基礎計算は共有化が始まっている。
- `RewardKernel` は `ztb/trading/environment/components/rewards/utils.py` (`RewardUtils`) などの軽量ヘルパーを積極的に活用する。

### 1.2 ヘルパー関数の再利用
`RewardUtils` に含まれる以下のロジックを `RewardKernel` および `scripts/v460` で共通利用する。
- `calculate_balance_penalty`: アクション比率の偏りに対するペナルティ。
- `calculate_activity_bonus`: 直近の取引頻度に対するボーナス。
- `calculate_position_size_bonus`: 適切なポジション量維持へのインセンティブ。

### 1.3 現在の前進状況

すでに前進済みのもの:
- `RewardKernel` の導入と Lite/Heavy 基礎報酬ロジックの共有
- `RewardCalculator.calculate_reward_simple()` の `RewardKernel` 委譲
- `reward_component_tracking.py` による
  - `build_reward_components(...)`
  - `extend_reward_components(...)`
  - `merge_reward_components(...)`
  - `set_reward_telemetry(...)`
  の shared 化
- `get_last_reward_components()` の snapshot 契約化
  - internal mutable payload の alias を避ける

まだ残るもの:
- `RewardCalculator` 本体の stateful orchestration
- stage ごとの repeated payload shaping
- telemetry と scalar payload の残差整理
- `RewardKernel` に寄せるべき stateless core と、`RewardCalculator` に残すべき stateful ownership の最終固定

直近の追加前進:
- `_last_reward_components` の更新点を `RewardCalculator` local helper に寄せ始めた
  - stage reset
  - scalar payload extend
  - detail merge
  - telemetry attach
- これにより `reward_component_tracking.py` は SSOT、`RewardCalculator` は local ownership の入口、という役割が明確になった
- `UnifiedTrainer` 側の optimization / advanced feature stats も helper 経由へ寄せる流れができ、報酬 payload の周辺契約も揃えやすくなった

---

## 2. アーキテクチャ設計

### 2.1 RewardKernel の拡張 (Phase 1 強化)
`RewardKernel` を単なる PnL 計算機から、**「環境に依存しない報酬エンジンの核」** に昇格させる。

- **Input**: `RewardContext` (PnL, Actions, Positions, etc.)
- **Params**: `RewardParams` (Scaling, Multipliers, etc.)
- **Logic**: `RewardUtils` を内部で呼び出し、ステートレスに報酬を算出。

現時点の線引き:
- `RewardKernel`
  - Lite/Heavy で共通化できる stateless core
  - simple reward / kernel-friendly な基礎 PnL ロジック
- `RewardCalculator`
  - component wiring
  - curriculum / trend / behavioral penalties
  - `_last_reward_components` の stage bookkeeping
  - external diagnostics payload shaping

### 2.2 設定同期の自動化
`LiteEnvConfig` が `ztb` 側の `RewardSettings` の一部を透過的に扱えるようにし、YAML 設定値がそのまま `RewardKernel` に流れる構造を構築する。

### 2.3 payload/telemetry 契約

報酬系は計算そのものだけでなく、外へ出す payload 契約も揃える。

- scalar payload
  - `reward_component_tracking.py` を SSOT にする
- non-scalar telemetry
  - `set_reward_telemetry(...)` 経由に寄せる
- outward snapshot
  - `get_last_reward_components()` は shallow snapshot を返し、internal dict を直接晒さない

この契約により、
- callback
- reporting
- heavy_env
- trainer/SAC
が見る `reward_components` の shape を段階的に揃える。

---

## 3. 実行ロードマップ (更新)

### Phase 1: RewardKernel の強化
- `RewardUtils` のロジックを `RewardKernel` に統合。
- `pytest` による `RewardCalculator.calculate_reward_simple` との整合性検証。

現状:
- 基礎部分は前進済み
- 今後は `RewardKernel` に無理に stateful logic を載せず、stateless core に限定する

### Phase 2: scripts/v460 への完全委譲
- `LiteTradingEnv` だけでなく、分析スクリプト等で報酬を再計算している箇所があれば、すべて `RewardKernel` に移行。

現状:
- `RewardKernel` を使う線はできている
- ただし報酬再計算箇所の横断棚卸しは継続対象

### Phase 3: payload / diagnostics 契約の収束
- `reward_component_tracking.py` を SSOT として stage payload をさらに揃える
- callback / reporting / heavy_env / trainer/SAC が見る `reward_components` の shape を整列
- snapshot 契約で mutable alias を防止する

### Phase 4: テスト駆動の品質保証
- `tests/unit/v460/test_558_reward_unification.py` の新設または等価 coverage の整理
- 数値的一致と payload 契約の両方を自動テストで監視

### Phase 5: 大分割の前提整理
- `RewardCalculator` を
  - stateless core
  - stage bookkeeping
  - telemetry shaping
  - component wiring
  に分ける split-first 設計を固める
- ここで初めて大きい分割に入る

## 3.1 直近の実行順

1. `RewardCalculator` の payload/telemetry 契約をさらに揃える
2. `RewardKernel` と `RewardCalculator` の境界を、stateful/stateless で固定する
3. callback / reporting / heavy_env / trainer 側の報酬 payload shape を揃える
4. その後に、必要なところだけ `RewardKernel` へ寄せる
5. `RewardCalculator` 本体の repeated stage logic を helper 経由で圧縮し、大分割前の破壊半径を下げる

---

## 4. 期待される効果

- **評価の信頼性向上**: `LiteTradingEnv` で学習したモデルを `HeavyTradingEnv` に持っていった際の「報酬の不連続性」を排除。
- **コード削減**: `RewardCalculator.py` の 2,200行という巨大さを、コンポーネントの切り出しによって劇的に削減する道筋を作る。
- **保守性**: 報酬ロジックの修正が必要な際、`RewardKernel` または `RewardUtils` を直すだけで全環境に反映される。

## 5. 現時点の判断

- `557#` は引き続き報酬系の詳細計画として使う
- 全体の実行順は `551#`、報酬系の深掘りは `557#`、全体母艦は `521#` とする
- 直近は「RewardKernel に何でも寄せる」のではなく、
  - stateless core は `RewardKernel`
  - stateful orchestration は `RewardCalculator`
  という線を崩さないことが重要

---
