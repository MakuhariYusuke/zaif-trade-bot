# 241# 240# Toxicity Budget セルフレビュー修正

**Phase**: 2 (バグ修正・品質向上)  
**前提**: 240# (`30ddc3009`)  
**テスト**: 3344 passed (+16)

---

## セルフレビューで発見した問題

### CRITICAL (3件)

#### C-1: graded response が到達不能コード
- **場所**: `cycle_gate_aggregator.py` Gate 4/5
- **問題**: `_apply_toxicity_graded()` は `g4.blocked=True` (= `is_buy_killed=True`)
  の中でのみ呼ばれていた。しかし `check_kill()=True` のとき
  `assess_toxicity()` は常に score≥1.0 (KILL) を返すため、
  `_apply_toxicity_graded()` は KILL → return False → 従来フロー。
  **YELLOW/ORANGE の段階的応答パスは完全な dead code だった。**
- **修正**: 段階的応答を gate 非 block 時 (elif branch) に移動。
  pre-kill ゾーン (check_kill=False だが score∈[0.3,1.0)) で
  offset 拡大と参加率制限を適用。

#### C-2: 評価順序による状態不整合
- **場所**: `fill_loop_orchestrator.py` L1590-1605
- **問題**: Python は keyword args を左→右で評価するため、
  `is_buy_killed=self._is_buy_killed()` (check_kill, 副作用あり) が
  `buy_toxicity=self._assess_buy_toxicity()` (assess_toxicity, 副作用なし)
  より先に実行される。`check_kill()` は `_cooldown` をデクリメントするため、
  最終 cooldown サイクルで `assess_toxicity()` が cooldown=0 を観測し
  YELLOW/ORANGE を返す可能性があった。
- **修正**: `_assess_buy/sell_toxicity()` を `evaluate()` 引数リストの外に出し、
  `check_kill()` の前に評価。

#### C-3: `object | None` 型注釈
- **場所**: `fill_loop_orchestrator.py` `_assess_buy/sell_toxicity()`
- **問題**: 戻り値型が `object | None` で mypy 型安全を完全破壊。
  プロジェクト方針「Any型使用の回避」違反。
- **修正**: `ToxicityAssessment | None` に修正。TYPE_CHECKING import 追加。

### SIGNIFICANT (4件)

#### S-1: `getattr()` 反パターン
- `getattr(config, "toxicity_budget_enabled", False)` は
  常に存在する dataclass フィールドへの冗長な防御。直接アクセスに修正。

#### S-2: DRY 違反
- `_assess_buy_toxicity()` と `_assess_sell_toxicity()` が 95% 同一コード。
  統一 `_assess_toxicity(mgr)` メソッドに集約。

#### S-3: ホットパス内 runtime import
- `_apply_toxicity_graded()` 内で毎回 `from ztb.risk... import ToxicityLevel`。
  モジュールレベルでインポートし、sentinel 定数にキャッシュ。

#### S-4: config バリデーション欠落
- toxicity 設定フィールドの制約チェックなし。
  `__post_init__` にバリデーション追加:
  - `0 <= warn_level < caution_level <= 1.0`
  - `warn_offset_mult >= 1.0`
  - `caution_offset_mult >= warn_offset_mult`
  - `kill_offset_mult >= caution_offset_mult`
  - `0 < min_participation <= 1.0`
  - `toxicity_budget_enabled=False` 時はスキップ

### LOW (1件)

#### L-1: ORANGE 参加率の連続性
- ORANGE ゾーン入口で `participation_rate=1.0` (skip なし) → docstring と微妙に矛盾。
  ただし線形補間による連続設計は Glosten-Milgrom 理論上は正しい(逆選択プレミアムは連続関数)。
  → 修正不要、docstring を維持。

---

## 修正ファイル一覧

| ファイル | 修正内容 |
|---------|---------|
| `scripts/v460/lib/cycle_gate_aggregator.py` | C-1: Gate 4/5 pre-kill graded, S-3: ToxicityLevel モジュール import |
| `scripts/v460/lib/fill_loop_orchestrator.py` | C-2: 評価順序修正, C-3: 型安全化, S-1: getattr除去, S-2: DRY統一 |
| `ztb/risk/sell_dynamic_kill.py` | S-4: __post_init__ バリデーション追加 |
| `tests/unit/v460/test_240_toxicity_budget.py` | 42→58 tests (+16), 新テストクラス3個追加 |

## 設計変更の理論的根拠 (C-1)

240# の元設計は「kill gate がブロックしたとき toxicity が KILL 未満なら段階的応答」
という "escape hatch" モデルだった。しかし:

1. `check_kill()=True` ⇔ `rolling_mean < threshold` ⇔ `score >= 1.0` ⇔ KILL
2. よって **gate block 時に非KILL toxicity は存在しない** (cooldown 中も KILL)

241# の修正後は "pre-kill zone modifier" モデル:   

1. `check_kill()=False` (まだ kill 未到達) だが `score ∈ [warn, 1.0)` → YELLOW/ORANGE
2. Gate は block しない (非 kill) が、toxicity offset/participation が CycleGateResult に設定
3. Orchestrator が participation_rate で確率的 skip、Executor が offset_mult でスプレッド拡大

これは Glosten-Milgrom の逆選択プレミアム理論に正確に対応:
- 情報非対称の度合いが高まるほどスプレッドを広げる (offset_mult)
- 逆選択コストが期待利益を侵食するほど参加頻度を下げる (participation_rate)
