# Codex Task: 690# Bucket 別 Skip Budget (638# P1)

## 目的
SkipGate の連続 skip カウンタをレジーム × side でバケット化し、
特定レジーム/side の過剰 skip を防ぎつつ、他のレジーム/side では適切に skip を許容する。

現状: 全レジーム・全 side で単一の `_primary_consecutive_skip_count` しかなく、
ranging/sell の正当な連続 skip が safety valve に達すると、trending/buy まで強制 PASS される問題がある。

## 背景

### 638# で特定された課題
- 単一カウンタは全レジーム/side を混同 → 粗すぎる制御
- SG ML MI≈0.064 (672#) だが bypass_mode (686#) で対応済。SG の role は shift:
  「ML skip → 本番 block」から「ML skip → offset 調整の参考値 + budget 制御」へ

### 現状のアーキテクチャ
- `_primary_consecutive_skip_count`: evaluate() 内でインクリメント、PASS or safety valve でリセット
- `skip_gate_primary_max_consecutive_skip`: int (config, default=0=無効)
- `regime_value`: evaluate() のパラメータとして渡される (str | None)
- bypass_mode=true 時: skip 判定は計算されるが block しない (686#/306664e32)

## タスク

### Task 1: BucketedSkipBudget クラス

**新規作成**: `scripts/v460/lib/skip_gate_budget.py`

```python
@dataclass
class BucketKey:
    regime: str     # "trending_up", "ranging", "unknown" etc.
    side: str       # "buy", "sell"

@dataclass  
class BucketState:
    skip_count: int = 0
    pass_count: int = 0
    window_start_ts: float = 0.0

class BucketedSkipBudget:
    def __init__(self, config: FillTestConfig) -> None: ...
    
    def is_budget_exhausted(self, regime: str, side: str) -> bool:
        """現在バケットの skip count が budget に達しているか判定."""
        
    def record_decision(self, regime: str, side: str, *, skipped: bool) -> None:
        """skip/pass 判定を記録. window 超過時はバケットローテーション."""
        
    def get_state(self, regime: str, side: str) -> BucketState:
        """現在バケットの状態を返す (observability)."""
        
    def _rotate_if_expired(self, key: BucketKey) -> None:
        """window_min を超えたバケットをリセット."""
```

### Task 2: YAML 設定

**対象**: `configs/v460/fill_test.yaml`

```yaml
skip_gate:
  # 既存 (変更なし)
  skip_gate_primary_max_consecutive_skip: 0  # グローバル緊急ブレーキ (0=無効)
  
  # 新規: バケット制御
  budget_enabled: false          # true で有効化 (段階的ロールアウト)
  budget_window_min: 60          # バケットウィンドウ (分)
  budget_limits:                 # regime×side の最大 skip 数 / window
    default: 8                   # 未指定 regime×side のデフォルト
    trending_up:
      sell: 3                    # 急騰中の sell skip は厳格制限
      buy: 12                   # 急騰中の buy はより寛容
    trending_down:
      sell: 12                   # 暴落中の sell はより寛容
      buy: 3                    # 暴落中の buy skip は厳格制限
    ranging:
      sell: 6
      buy: 10
```

### Task 3: FillConfig 拡張

**対象**: `scripts/v460/lib/fill_config.py`, `fill_config_parser.py`, `fill_config_validation.py`

1. `budget_enabled: bool` (default=False)
2. `budget_window_min: int` (default=60)
3. `budget_limits: dict[str, dict[str, int] | int]` (default={"default": 8})
4. `get_budget_limit(regime: str, side: str) -> int` ヘルパー
5. validation: budget_limits の値は正整数、window_min > 0

### Task 4: SkipGateEvaluator 統合

**対象**: `scripts/v460/lib/skip_gate_evaluator.py`

1. `__init__` で `BucketedSkipBudget` インスタンスを生成 (budget_enabled=true 時のみ)
2. `evaluate()` 内の safety valve ロジック (L890-940) の **前** にバケットチェックを挿入:
   ```python
   # Bucket budget check (before primary safety valve)
   if self._budget is not None:
       if self._budget.is_budget_exhausted(sg_regime, side):
           # Budget exhausted → force PASS
           decision = SkipDecision(should_skip=False, ...)
           logger.info(f"[dt={trace_id}] skip_gate_budget_exhausted: regime={sg_regime}, side={side}")
   ```
3. 判定後に `self._budget.record_decision(sg_regime, side, skipped=decision.should_skip)` を呼ぶ
4. 従来の `_primary_consecutive_skip_count` はグローバル緊急ブレーキとして **残す** (budget と独立動作)

### Task 5: FillRecord 記録

**対象**: `scripts/v460/lib/fill_record_builder.py`, `ztb/metrics/fill_quality.py`

1. `skip_gate_budget_regime: str | None` — skip 時のレジーム
2. `skip_gate_budget_remaining: int | None` — skip 時の残り budget
3. `skip_gate_budget_exhausted: bool | None` — budget 枯渇で PASS 強制されたか

### Task 6: テスト

**対象**: `tests/unit/v460/test_690_skip_budget.py`

1. budget_enabled=false → 従来動作 (budget 影響なし)
2. budget_enabled=true → skip が budget limit に達したら PASS 強制
3. window ローテーション: window_min 超過でカウンタリセット
4. regime×side 独立: trending_up/sell が枯渇しても trending_up/buy は影響なし
5. unknown regime → default budget を使用
6. hot-reload: budget_limits 変更時にバケット ceiling のみ更新、カウンタ保持
7. primary safety valve との共存: 両方有効時に独立動作
8. FillRecord に budget_remaining が記録される
9. `python -m pytest tests/ -x --tb=short` で全テスト pass

## 受け入れ基準

- [ ] BucketedSkipBudget が regime×side×window でバケット管理する
- [ ] budget_enabled=false で完全に従来動作
- [ ] budget 枯渇時に PASS が強制され、cancel_reason に反映
- [ ] primary safety valve と独立して動作
- [ ] FillRecord に budget 関連フィールドが記録
- [ ] 新規テスト 8 件以上、全テスト pass
- [ ] YAML hot-reload で budget_limits が反映される

## リスク評価

- **低リスク**: budget_enabled=false がデフォルト。段階的ロールアウト可能
- **ロールバック**: budget_enabled: false で完全に旧動作復帰
- **SG MI≈0 問題**: budget は ML 品質に依存しない (skip 回数のメカニカルな制御)
- **依存**: bypass_mode (686#) と独立動作。bypass=true + budget=true は「skip スコアは計算するが block せず、ただし budget 統計は収集」
