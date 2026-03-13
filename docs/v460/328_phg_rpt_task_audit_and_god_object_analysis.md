# 328# rpt: タスク棚卸し + God Object 分割分析

**日付**: 2026-03-08
**種別**: rpt (調査・分析レポート)
**前提**: 327# (`4b1aa540e`), 326# (`401ac0116`)
**状態**: 分析完了 — コード変更なし (ドキュメントのみ)

---

## 1. 目的

326# (Mixin audit) → 327# (Proactive bug fix) を経て、プロジェクト全体のタスク棚卸しと
残存 God Object の分割戦略を策定する。

1. 全ドキュメント走査による **47 件の未完了タスク** の発見・分類
2. `fill_config.py` (1,963 行) と `fill_loop_orchestrator.py` (1,490 行) の **構造分析**
3. 分割アプローチの比較検討 — Mixin 以外の手法も含む

---

## 2. タスク棚卸し結果 (47 件)

### 2.1 P0: Gate 進捗ブロッカー (2 件)

| ID | 内容 | 出典 | 状態 | 依存 |
|---|---|---|---|---|
| G1.2 | 168h 連続クリーンデータ蓄積 | 000# §3.3 F7/F8 | ⏳ 計測中 (bot=`dcc3064a8`, 22日稼働) | — |
| G1.1 | Gate 正式判定 (F1–F8 全指標算出) | 280# R-2 | G1.2 待ち | G1.2 |

> **注**: Bot PID 58008 は `dcc3064a8` (2026-03-06 23:44 デプロイ) で稼働中。
> HEAD `4b1aa540e` とは **30 コミット差**。168h 計測は このコミットから。

### 2.2 P1: 収益性直結・高優先 (10 件)

| ID | 分類 | 内容 | 出典 | 状態 |
|---|---|---|---|---|
| P1-1 | 分析 | SkipGate 再訓練 (n≥500 preorder features) | 097# / 280# R-3 | データ蓄積待ち |
| P1-2 | AB test | spread_adaptive AB テスト (narrow_spread_bps 探索) | 093# / 280# R-4 | 未着手 |
| P1-3 | コード | Volatility Guard 動的ゲーティング | 107# / 280# R-5 | 設計済み・未実装 |
| P1-4 | コード | `fill_config.py` God Object 分割 (1,963→目標 <1,000) | 280# R-10 / 259# P1-1 | 本 doc で分析 |
| P1-5 | YAML | offset_ceiling_ratio サイド別 ceiling YAML 配線修正 | 321# | ✅ 321# で修正済み |
| P1-6 | 分析 | AS 確率の時間帯分布の深堀り | 306# H5 | 310# A で部分対応 |
| P1-7 | 分析 | regime 遷移間隔の統計分析 | 306# O2 | 未着手 |
| P1-8 | 分析 | Queue Position fill probability キャリブレーション | 306# O1 | 未着手 |
| P1-9 | 分析 | Offset Stage 寄与量の定量分析 | 306# S1 | 未着手 |
| P1-10 | 分析 | Microprice 有効性の回帰分析 | 310# C | 未着手 |

### 2.3 P2: 中優先 — ph5 前・品質改善 (21 件)

| ID | 分類 | 内容 | 出典 |
|---|---|---|---|
| P2-1 | コード | `OrderManager.execute_trade()` 実取引パス | 013# D-1 |
| P2-2 | コード | `post_only` maker 保証 | 013# D-3 |
| P2-3 | コード | `asyncio.to_thread` 残 5 メソッド | 013# C-4 |
| P2-4 | コード | adapter Protocol 化徹底 | 259# P1-2 / 280# R-11 |
| P2-5 | コード | `order_monitor` except narrow 化 (12 箇所) | 259# P1-3 / 280# R-12 |
| P2-6 | コード | OrderBook Protocol (OB) | 259# P2-1 / 280# R-13 |
| P2-7 | コード | SkipDecision 必須化 | 259# P2-2 / 280# R-14 |
| P2-8 | コード | `run_single_cycle` 分離 | 259# P2-4 / 280# R-15 |
| P2-9 | コード | evaluate 分離 | 259# P2-5 / 280# R-16 |
| P2-10 | コード | `run_continuous` 更なる分割 (~1,181行→目標 <500) | 280# R-10 |
| P2-11 | コード | Parkinson σ 推定器の VG/AS 統合 | 305# |
| P2-12 | コード | Kyle λ fill_rate_k 動的推定 | 266# |
| P2-13 | コード | Amihud ILLIQ baseline 動的算出 | 266# |
| P2-14 | コード | Dynamic Cycle Interval の σ_ref キャリブレーション | 306# L1 |
| P2-15 | テスト | phantom_guard 統合テスト拡充 | 237# |
| P2-16 | テスト | MCB/SAD 統合テスト拡充 | 211# |
| P2-17 | コード | Tier-2/3 統合 (PnL MC, RiskRuleEngine, Reconciliation) | 113# / 280# R-9 |
| P2-18 | 分析 | 日次 PnL レポート自動生成 | 168# |
| P2-19 | 分析 | AS 原因分類 (time-of-day vs regime vs spread) | 306# |
| P2-20 | 分析 | Balance-forced fill の品質分離分析 | 283# P1-5 |
| P2-21 | AB test | confidence_lot テスト | 151# P3-03 |

### 2.4 P3: 低優先 — v461+ (14 件)

| ID | 分類 | 内容 | 出典 |
|---|---|---|---|
| P3-1 | コード | SkipGate 単体テスト拡充 | 106# R3 |
| P3-2 | コード | lib → ztb 移動 (残 4 モジュール) | 106# R5 |
| P3-3 | コード | utils 70+ ファイル分割 | 106# R6 |
| P3-4 | リファクタ | `except Exception` 78 箇所 narrow 化 | 259# |
| P3-5 | リファクタ | `getattr` 29 箇所排除 | 259# |
| P3-6 | リファクタ | `hasattr` 6 箇所排除 | 228#/230# |
| P3-7 | リファクタ | `:object` 型 ~20 箇所 Protocol 化 | 277# C1/S3 |
| P3-8 | リファクタ | `type:ignore` 12 箇所排除 | 266# |
| P3-9 | テスト | FillCycleExecutorMixin 分離テスト | 323# |
| P3-10 | コード | UnifiedTrainer God Object (2,835 行) | 109# DUP3 |
| P3-11 | コード | alert_mode.json の JSON Schema 定義 | 215# |
| P3-12 | docs | YAML 設定リファレンス (全 170+ フィールド) | — |
| P3-13 | docs | アーキテクチャ図 (Mermaid MRO ダイアグラム) | 325# |
| P3-14 | 型安全 | config_access docstring 追加 | 259# P3-3 |

---

## 3. God Object 分析

### 3.1 現状ファイルサイズ

| ファイル | 行数 | 目標 | 超過 |
|---|---|---|---|
| `fill_config.py` | **1,963** | <1,000 (docstring 記載) | **+963** |
| `fill_loop_orchestrator.py` | **1,490** | <500 (280# R-10) | **+990** |
| `fill_cycle_executor.py` | 1,010 | — | 要注視 |
| `orchestrator_lifecycle.py` | 490 | — | ✅ OK |
| `orchestrator_post_cycle.py` | 413 | — | ✅ OK |
| `orchestrator_guards.py` | 217 | — | ✅ OK |
| `fill_record_helpers.py` | 268 | — | ✅ OK |

### 3.2 `fill_config.py` 構造分析

```
FillTestConfig (L22–L1942) — 単一 @dataclass
├── Fields (L22–L696)         ~675 行  ← 問題の中核
│   ├── Core/Timing           ~45 行  (symbol, cycle_interval, order_timeout...)
│   ├── Regime                ~80 行  (enable_regime, trending_offset_boost...)
│   ├── Drawdown Guard       ~60 行  (daily_drawdown_*, per_side_dd_*)
│   ├── SkipGate             ~95 行  (skip_gate_*, ev_weighted...)
│   ├── Risk/Safety          ~85 行  (loss_cap_*, toxic_fill_veto_*, one_sided_*)
│   ├── Stale/VG/市場理論     ~80 行  (stale_order_*, volatility_guard_*, AS_*)
│   ├── Dynamic Kill         ~55 行  (sell_dynamic_kill_*, buy_dynamic_kill_*)
│   ├── Balance/Side Control ~55 行  (balance_forced_*, forced_buy_delay_*)
│   ├── Infra/Tuning         ~50 行  (cb_*, hm_*, hot_reload_*, lock_*)
│   └── Misc Features        ~70 行  (imbalance_*, spread_adaptive_*, microprice_*)
├── __post_init__ (L697–L989)  ~293 行  バリデーション
├── YAML Parsers (L997–L1693)  ~697 行
│   ├── _parse_trading_features  (L997–L1088)    ~92 行
│   ├── _parse_skip_gate_section (L1090–L1188)   ~99 行
│   ├── _parse_stale_vg_section  (L1189–L1345)  ~157 行
│   ├── _parse_stopgap_section   (L1346–L1590)  ~245 行
│   └── _parse_infra_section     (L1591–L1693)  ~103 行
├── from_yaml (L1694–L1942)    ~249 行
└── Result Classes (L1948–L2017)
    ├── SkipGateResult         ~19 行
    ├── FillMonitorResult      ~22 行
    └── PnlMeasurement         ~28 行

compute_ev_offset_multiplier() — モジュールレベル関数 (L2018–L2046)
```

### 3.3 `fill_loop_orchestrator.py` 構造分析

```
RunSessionState (L69–L89)  — @dataclass (ループ状態)
FillLoopOrchestratorMixin (L91–L1595)
├── _effective_sleep        (L177–L203)   ~27 行
├── _make_loop_skip_record  (L204–L239)   ~36 行
├── _execute_skip           (L240–L298)   ~59 行
├── _acquire_lock           (L299–L303)    ~5 行
├── _release_lock           (L304–L308)    ~5 行
├── _update_lock_heartbeat  (L309–L313)    ~5 行
└── run_continuous          (L314–L1595) **~1,181 行** ← 巨大メソッド
    ├── Init/Heartbeat      (L314–L350)    ~37 行
    ├── DD halt 処理        (L352–L442)    ~91 行
    ├── Operator alert      (L444–L460)    ~17 行
    ├── MCB/SAD ガード      (L462–L535)    ~74 行
    ├── Hard skip UTC       (L537–L570)    ~34 行
    ├── Phantom guard       (L572–L610)    ~39 行
    ├── Per-side DD halt    (L612–L675)    ~64 行
    ├── Toxic veto          (L677–L715)    ~39 行
    ├── Phantom veto        (L717–L740)    ~24 行
    ├── Time filter         (L742–L880)   ~139 行
    ├── Regime update       (L882–L910)    ~29 行
    ├── Balance preflight   (L912–L990)    ~79 行
    ├── Bal forced + IE     (L990–L1000)   ~11 行
    ├── One-sided escalation(L1000–L1065)  ~66 行
    ├── Balance forced skip (L1067–L1140)  ~74 行
    ├── Forced buy delay    (L1142–L1210)  ~69 行
    ├── CycleGateAggregator (L1212–L1390) ~179 行
    ├── Toxicity budget     (L1392–L1420)  ~29 行
    ├── Degraded liquidation(L1422–L1490)  ~69 行
    └── run_single_cycle 呼出 + post_cycle (L1490–L1595) ~106 行
```

---

## 4. 分割アプローチ比較

### 4.1 `fill_config.py` — 3 つのアプローチ

#### A. サブ Dataclass 合成 (推奨)

```python
@dataclass
class RegimeConfig:
    enable_regime: bool = True
    regime_window: int = 20
    ...  # ~80 行

@dataclass
class DrawdownConfig:
    daily_drawdown_enabled: bool = False
    ...  # ~60 行

@dataclass
class SkipGateConfig:
    skip_gate_enabled: bool = False
    ...  # ~95 行

@dataclass
class FillTestConfig:
    """トップレベル設定 — サブ config を合成."""
    symbol: str = "btc_jpy"
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    drawdown: DrawdownConfig = field(default_factory=DrawdownConfig)
    skip_gate: SkipGateConfig = field(default_factory=SkipGateConfig)
    ...
```

**メリット**:
- 論理的グループ化が明確 — IDE での補完・ナビゲーションが改善
- 各サブ config を独立ファイルに配置可能 (e.g. `configs/regime_config.py`)
- サブ config 単位でのバリデーションが自然 (`RegimeConfig.__post_init__`)
- YAML 構造と 1:1 対応 (既に `regime:`, `skip_gate:` 等のネスト済み)

**デメリット**:
- **破壊的変更**: `self.config.enable_regime` → `self.config.regime.enable_regime` (全参照箇所修正)
- 既存 YAML のフラットキー (`cycle_interval_sec` 等) との混在
- `from_yaml()` の大幅改修

**参照箇所影響**: `config.` を `grep` すると数百箇所のアクセスパターン修正が必要。

#### B. YAML パーサー + バリデーター分離 (推奨・低リスク)

```python
# fill_config.py — フィールド定義のみ (~700 行 → 許容範囲)
@dataclass
class FillTestConfig:
    symbol: str = "btc_jpy"
    ...  # ~675 行のフィールド + __post_init__ 最小限

# fill_config_parser.py — YAML パース (~700 行)
class FillConfigParser:
    @staticmethod
    def _parse_trading_features(yaml_cfg: dict) -> dict: ...
    @classmethod
    def from_yaml(cls, yaml_cfg: dict) -> FillTestConfig: ...

# fill_config_validation.py — バリデーション (~300 行)
def validate_fill_config(config: FillTestConfig) -> None: ...

# fill_config_results.py — Result dataclasses (~70 行)
@dataclass
class SkipGateResult: ...
@dataclass
class FillMonitorResult: ...
@dataclass
class PnlMeasurement: ...
```

**メリット**:
- **非破壊的**: `config.field_name` のアクセスパターン変更なし
- `FillTestConfig` が純粋なデータコンテナに — SRP 達成
- YAML パーサーの独立テスト可能
- バリデーションの独立テスト・段階追加可能
- Result classes の循環参照リスク解消

**デメリット**:
- `from_yaml()` のエントリポイント移動 (`FillTestConfig.from_yaml()` → `FillConfigParser.from_yaml()`)
- 呼び出し元 (CLI, テスト) の import 変更が必要

#### C. Mixin パターン (非推奨)

`FillTestConfig` は dataclass であり、メソッドが `__post_init__` + `from_yaml` + staticmethod パーサーのみ。
Mixin は「振る舞いの分離」に適しており、「データの分離」には不向き。

**結論**: Config dataclass に Mixin は不適切。B を推奨。

### 4.2 `fill_loop_orchestrator.py` — 3 つのアプローチ

#### A. Guard フェーズ抽出 (推奨)

`run_continuous` の while ループ内は明確なフェーズ構造を持つ:

```python
# orchestrator_pre_cycle.py (新規)
class OrchestratorPreCycleMixin:
    """サイクル前ガード評価 — run_continuous の前半を分離."""

    async def _check_dd_halt(self, st: RunSessionState) -> bool: ...
    async def _check_operator_alert(self, st: RunSessionState) -> bool: ...
    async def _check_mcb_sad(self, st: RunSessionState) -> tuple[bool, bool, bool]: ...
    async def _check_hard_skip(self, st: RunSessionState) -> bool: ...
    async def _check_phantom(self) -> None: ...
    async def _resolve_side(
        self, st: RunSessionState, next_side: str,
    ) -> tuple[str, bool, bool, bool]: ...
    async def _check_time_filter(
        self, st: RunSessionState, next_side: str,
    ) -> tuple[str, bool]: ...
    async def _check_balance(
        self, st: RunSessionState, next_side: str, regime_mult: float,
    ) -> tuple[str, bool, bool, bool, bool]: ...
```

`run_continuous` は以下に縮小:

```python
async def run_continuous(self, hours: float) -> list[FillRecord]:
    st = await self._init_run_session()
    heartbeat_task = ...
    while time.time() < end_time and not self._kill_switch.is_killed():
        # Phase 1: Pre-cycle guards
        if await self._check_dd_halt(st): continue
        if await self._check_operator_alert(st): continue
        blocked, mcb_w, sad_w = await self._check_mcb_sad(st)
        if blocked: continue
        if await self._check_hard_skip(st): continue
        await self._check_phantom()
        next_side, balance_forced, is_rescue, one_sided, inv_escape = \
            await self._resolve_side_and_balance(st)
        if next_side is None: continue

        # Phase 2: Gate evaluation
        gate_result = await self._evaluate_gates(st, next_side, ...)
        if gate_result.blocked: continue

        # Phase 3: Execute cycle
        record = await self.run_single_cycle(...)
        await self._process_post_cycle(st, record, ...)
    return await self._finalize_run(st, ...)
```

**メリット**:
- `run_continuous` が ~100 行に縮小 (目標 <500 大幅達成)
- 各ガードが独立メソッド → テスト容易
- 既存 Mixin 構造 (guards/lifecycle/post_cycle) と整合

**デメリット**:
- `continue` 制御フローの分離が複雑 — 各メソッドが `bool` (should_continue) を返す設計
- ローカル変数の多数受け渡し (`_balance_forced`, `_is_rescue`, `_one_sided_balance` 等)

#### B. サイクル状態オブジェクト導入

```python
@dataclass
class CycleContext:
    """1 サイクルの実行コンテキスト."""
    next_side: str = ""
    balance_forced: bool = False
    is_rescue: bool = False
    one_sided_balance: bool = False
    inventory_escape: bool = False
    mcb_warning: bool = False
    sad_warning: bool = False
    halt_recovery_active: bool = False
    regime_mult: float = 1.0
```

ガードメソッドが `CycleContext` を変異させる:

```python
async def _resolve_pre_cycle(self, st, ctx: CycleContext) -> bool:
    """全ガードを順次評価し ctx を更新。True = skip (continue)."""
    ...
```

**メリット**:
- ローカル変数の爆発を解消 — 状態を構造体に集約
- フェーズ間のデータ受け渡しが明確
- テスト時に `CycleContext` をモック可能

**デメリット**:
- 新しい抽象層の導入 — 学習コスト
- `CycleContext` の肥大化リスク

#### C. Strategy パターン (非推奨)

各ガードを独立した Strategy オブジェクトにし、Chain of Responsibility で順次評価:

```python
class Guard(Protocol):
    async def evaluate(self, ctx: CycleContext) -> GuardResult: ...

guards = [DDHaltGuard(), MCBGuard(), TimeFilterGuard(), ...]
for guard in guards:
    result = await guard.evaluate(ctx)
    if result.blocked: break
```

**デメリット**: ガード間の依存関係 (e.g. MCB×SAD AND Escalation) が Chain パターンに不適合。
順序依存・状態共有が多すぎるため、過剰な抽象化。

### 4.3 推奨アプローチまとめ

| ファイル | 推奨 | リスク | 効果 |
|---|---|---|---|
| `fill_config.py` | **B. YAML パーサー分離** | 低 (非破壊) | 1,963 → ~700 + ~700 + ~300 + ~70 |
| `fill_loop_orchestrator.py` | **A+B. Guard 抽出 + CycleContext** | 中 | 1,490 → ~150 + ~600 (guards) |

---

## 5. fill_config.py 分割計画 (B: YAML パーサー分離)

### Step 1: Result dataclasses 分離

```
fill_config_results.py (新規)
├── SkipGateResult
├── FillMonitorResult
├── PnlMeasurement
└── compute_ev_offset_multiplier()
```

- **影響**: import パス変更のみ
- **リスク**: 最小 — 独立したクラスの物理的移動
- **行数削減**: ~100 行

### Step 2: バリデーション分離

```
fill_config_validation.py (新規)
└── validate_fill_config(config: FillTestConfig) -> None
```

- `__post_init__` → `validate_fill_config()` に委譲
- `__post_init__` は 1 行: `validate_fill_config(self)`
- **行数削減**: ~293 行

### Step 3: YAML パーサー分離

```
fill_config_parser.py (新規)
├── _parse_trading_features()
├── _parse_skip_gate_section()
├── _parse_stale_vg_section()
├── _parse_stopgap_section()
├── _parse_infra_section()
└── from_yaml() → parse_fill_config_yaml()
```

- **行数削減**: ~700 行
- `FillTestConfig.from_yaml()` は互換ラッパーとして残す (`return parse_fill_config_yaml(yaml_cfg)`)

### 結果

| ファイル | 行数 |
|---|---|
| `fill_config.py` | ~700 (フィールド定義 + 1 行 `__post_init__`) |
| `fill_config_parser.py` | ~700 (YAML パーサー) |
| `fill_config_validation.py` | ~300 (バリデーション) |
| `fill_config_results.py` | ~100 (Result classes + utility) |

→ **全体で 4 ファイル、各 <1,000 行、目標達成**

---

## 6. run_continuous 分割計画 (A+B: Guard 抽出 + CycleContext)

### Step 1: CycleContext 導入

`fill_config.py` or `fill_loop_orchestrator.py` に小さな dataclass を追加。

### Step 2: Guard メソッド抽出

`orchestrator_pre_cycle.py` (新規) に以下を抽出:

| メソッド | 元の行範囲 | 行数 |
|---|---|---|
| `_check_dd_halt` | L352–L442 | ~91 |
| `_check_operator_alert` | L444–L460 | ~17 |
| `_check_mcb_sad` | L462–L535 | ~74 |
| `_check_hard_skip` | L537–L570 | ~34 |
| `_check_phantom` | L572–L610 | ~39 |
| `_resolve_side` | L612–L740 | ~129 |
| `_check_time_filter` | L742–L880 | ~139 |
| `_check_balance_preflight` | L912–L1000 | ~89 |
| **合計** | | **~612** |

### Step 3: Gate 評価・実行部分

`run_continuous` に残す部分:

| 部分 | 行数 |
|---|---|
| Init + heartbeat | ~37 |
| while ループ (ガード呼び出し) | ~50 |
| Gate 評価 + 実行 | ~100 |
| one-sided / degraded | ~120 |
| **合計** | **~307** |

→ run_continuous ~307 行 + orchestrator_pre_cycle ~612 行 → **目標 <500 は未達だが半減**

### 補足: 目標 <500 への追加施策

One-sided escalation (~66 行) と forced_buy_delay (~69 行) を `orchestrator_guards.py` に移動、
balance_forced skip (~74 行) も含めると run_continuous は ~200 行に到達可能。

---

## 7. 既存メソッド再利用性調査

### 7.1 再利用可能な既存メソッド

| メソッド | 定義場所 | 再利用候補 |
|---|---|---|
| `_execute_skip()` | orchestrator (L240) | 新 guard メソッドから呼び出し可 |
| `_make_loop_skip_record()` | orchestrator (L204) | 同上 |
| `_effective_sleep()` | orchestrator (L177) | 同上 |
| `_opposite_side()` | guards (L167) | `_resolve_side` から使用 |
| `_is_time_filtered()` | guards (L200) | `_check_time_filter` から使用 |
| `_is_side_killed()` | guards (L33) | `_resolve_side` から使用 |
| `_check_balance_for_side()` | guards (L205) | `_check_balance_preflight` から使用 |
| `_inc_guard_fire()` | guards (L125) | 全 guard メソッドから使用 |
| `_feed_mcb_sad()` | guards (L154) | `_check_dd_halt` から使用 |
| `_assess_buy/sell_toxicity()` | guards (L114/118) | Gate 評価前に使用 |

→ 既存 Mixin の public メソッドを新 Mixin からそのまま呼び出せる。追加の共通化は不要。

### 7.2 `_execute_skip()` の汎用性

326# M-2 で `_opposite_side()` DRY 化を行った。
`_execute_skip()` (59 行) は以下を一括処理:
- skip record 生成 + batch append
- batch flush
- heartbeat 更新
- state save (オプション)
- last_side 更新 (オプション)
- sleep (オプション)

→ 新 guard メソッドからの呼び出しに十分な汎用性を持つ。再実装不要。

---

## 8. 実行優先度

```
Phase 1 (低リスク・即実行可能)
  ├── fill_config Step 1: Result classes 分離 (~100行)
  └── fill_config Step 2: Validation 分離 (~300行)

Phase 2 (中リスク・テスト重要)
  └── fill_config Step 3: YAML Parser 分離 (~700行)

Phase 3 (中リスク・設計検討)
  ├── CycleContext dataclass 導入
  └── orchestrator_pre_cycle.py 抽出

Phase 4 (収益性・データ依存)
  ├── G1.2 168h 計測完了
  ├── SkipGate 再訓練
  └── spread_adaptive ABテスト
```

---

## 9. 制約事項

- **Bot 稼働中**: PID 58008 (`dcc3064a8`) — コード変更は bot 再起動まで反映されない
- **Git stash**: 262# リバート混入 — **絶対に pop しない**
- **168h 計測**: コード変更は計測に影響しない (bot は旧 SHA で稼働中)
- **テスト**: 4,072 passed (327#) — リファクタ後も全パス必須

---

## 関連ドキュメント

- 280# [280_ph2_rpt_position_and_remaining_tasks.md](v460/280_ph2_rpt_position_and_remaining_tasks.md) — 前回の課題浚い上げ
- 259# [259_phg_rpt_codebase_sweep.md](v460/259_phg_rpt_codebase_sweep.md) — 型安全・品質 Sweep
- 325# [325_orchestrator_god_object_split.md](325_orchestrator_god_object_split.md) — Mixin 分割 (前回)
- 326# [326_mixin_audit_and_encapsulation_fix.md](326_mixin_audit_and_encapsulation_fix.md) — Mixin 監査
- 327# — Proactive bug fix (loss_cap_ratio, file handle leak)
