# 382# Codex レビュー依頼: SB3スタブ致命的バグ修正 + SAC訓練パイプライン検証

## 0. 前提 (IDE クラッシュにより履歴消失・Context Reset)

本レビューは Session 379# での作業に対するもの。
Codex 側で IDE クラッシュが発生し会話履歴が消失しているため、
十分なコンテキストを含めて記載する。

---

## 1. プロジェクト概要

**zaif-trade-bot** は暗号通貨 (BTC/JPY) の自動マーケットメイキングボット。
v460 フェーズでは SAC (Soft Actor-Critic) による強化学習モデルを訓練し、
1分足の売買判断 (BUY/HOLD/SELL) を行うシステムを構築中。

### アーキテクチャ要点
- **環境**: `HeavyTradingEnv` — 17 OHLCV 特徴量 + 3 内部追跡特徴量 = obs_dim=20
- **行動空間**: 連続1D `[-1, 1]` → 閾値で離散化 (BUY/HOLD/SELL)
- **SB3**: stable-baselines3 2.7.0 (pip) の SAC / MlpPolicy
- **G2 Gate**: 4-seed × 50K steps 訓練 → 4条件 (E1〜E4) で合否判定

---

## 2. 発見された致命的バグ: SB3 スタブによるシャドウイング

### 2.1 症状
4 seeds × 全チェックポイント (5K〜50K) で **ROI = 0.0000** が出力。
モデルは保存されるが、一切のトレードが発生しない。

### 2.2 調査経緯

1. **閾値仮説** (不正確): `continuous_to_discrete_threshold = 0.3333` が HOLD ゾーン 66.7% を作り、SAC の tanh 初期出力が全て HOLD に分類 → 閾値を 0.10 に変更するも、依然 ROI=0.0000
2. **診断スクリプト作成**: `scripts/v460/diagnose_sac_actions.py` を作成・実行。以下が判明:
   - `model.policy` が `None`
   - `model.predict()` が `int(0)` を返す (numpy array でない)
   - `type(model)` は `<class 'stable_baselines3.SAC'>` だが `__module__` がローカルパス

### 2.3 根本原因

プロジェクトルートに **`stable_baselines3/` ディレクトリ** (ダミースタブ) が存在。
`sitecustomize.py` の `_prefer_local_package()` がこのスタブを site-packages 版より優先ロード。

**スタブの実装 (旧 `stable_baselines3/__init__.py`、現 `_sb3_test_stub/__init__.py`):**
```python
class _DummyAlgo(BaseAlgorithm):
    def learn(self, total_timesteps, **kwargs):
        return self  # ← 何もしない

    @classmethod
    def load(cls, path, env=None, **kwargs):
        return cls(env=env, **kwargs)  # ← 空インスタンスを返す

class SAC(_DummyAlgo):
    pass
```

`BaseAlgorithm.predict()`:
```python
def predict(self, observation, state=None, episode_start=None, deterministic=False):
    return 0, None  # ← 常に int(0) を返す
```

結果として:
- `SAC.learn()` → no-op (パラメータ更新なし)
- `SAC.predict()` → 常に `(0, None)` (int 型)
- `SAC.save()` → `return None` (no-op)
- 訓練ログには進捗が表示されるが、実質的に何も行われていなかった

### 2.4 `sitecustomize.py` の役割

Python インタプリタ起動時に自動実行されるフック。
`_prefer_local_package()` がプロジェクトルートの `stable_baselines3/` を
`sys.path` の先頭に挿入し、pip 版 SB3 よりローカルスタブを優先していた。

---

## 3. 修正内容 (Commit: `321c25de7`)

### 3.1 ディレクトリリネーム
```
stable_baselines3/ → _sb3_test_stub/
```
`_` prefix によりパッケージ名の衝突を回避。スタブ自体はテスト互換性のため保持。

### 3.2 `sitecustomize.py` の修正

```python
# Before:
def _prefer_local_package() -> bool:
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return True

# After:
def _prefer_local_package() -> bool:
    """Disabled: no longer prefers local stub over pip-installed SB3."""
    return False
```

`_replace_stub_with_filebacked()` の呼び出しも `pass` に置換。

### 3.3 `ztb/support/sb3_compat.py` の修正

```python
# Before: スタブを無条件に作成
sb3 = ModuleType("stable_baselines3")
sb3.SAC = type("SAC", (), {"learn": lambda self, *a, **k: self})

# After: 本物を優先 import、失敗時のみスタブ
try:
    import stable_baselines3 as sb3
except ImportError:
    sb3 = ModuleType("stable_baselines3")
    sb3.SAC = type("SAC", (), {"learn": lambda self, *a, **k: self})
```

### 3.4 YAML パラメータ変更 (`configs/v460/experiments/g2_sac_train.yaml`)

| パラメータ | Before | After | 理由 |
|---|---|---|---|
| `continuous_to_discrete_threshold` | 0.3333 | 0.10 | HOLD ゾーン 66.7% → 20% に縮小 |
| `learning_starts` | 100 | 1000 | バッファに多様なサンプルを蓄積してから学習開始 |

### 3.5 テスト更新

`tests/unit/v460/test_356_g2_sac_blockers.py`:
```python
# Before:
assert sac["learning_starts"] == 100
# After:
assert sac["learning_starts"] == 1000
```

---

## 4. 副次的修正 (Commits: `7d5fad87d`, `2a8269695`)

### 4.1 `inspect.signature()` キャッシュ化
**ファイル**: `ztb/trading/environment/components/calculators/reward_calculator.py`

```python
# 379# Perf: inspect.signature() is extremely expensive (~0.3ms/call).
# Cache the result per method object to avoid calling it every step.
if not hasattr(self, '_sig_cache'):
    self._sig_cache: dict[object, tuple[bool, frozenset[str]]] = {}
cache_key = reward_method
cached = self._sig_cache.get(cache_key)
if cached is None:
    sig = inspect.signature(reward_method)
    # ... cache the result ...
```

毎ステップの `inspect.signature()` 呼び出し (~0.3ms) を排除。

### 4.2 チェックポイント評価ステップ制限
**ファイル**: `scripts/v460/lib/tasks/sac_train.py`

```python
_CHECKPOINT_EVAL_MAX_STEPS = 5_000  # 973K全ステップ → 5Kに制限
```

チェックポイント評価が 973K ステップを全走査し ~30 分/回 かかっていたのを修正。

### 4.3 OOS 評価ステップ制限
**ファイル**: `scripts/v460/lib/sac_common.py`

```python
def evaluate_model_oos(model, env, n_episodes=1, max_steps_per_episode=10_000):
```

OOS 環境が 243K+ ステップを持つ場合、全走査は非現実的。10K ステップに制限。

---

## 5. 修正後の訓練結果 (本物の SB3 2.7.0)

| Seed | 最善 ROI (checkpoint) | 最終 ROI (50K) | OOS gross_roi | trade_count | 訓練時間 |
|---|---|---|---|---|---|
| 42 | **-0.0008** (20K) | -0.0019 | -0.00253 | 1001 | 27.0 min |
| 123 | **-0.0016** (30K) | -0.0024 | -0.00260 | 1401 | 32.6 min |
| 456 | **-0.0003** (35K) | -0.0007 | -0.00256 | 1109 | 31.6 min |
| 789 | **-0.0013** (50K) | -0.0013 | -0.00278 | 1455 | 34.0 min |

### G2 Gate 結果
```json
{
  "gate_result": "FAIL",
  "checks": {
    "positive_seed_ratio": { "value": 0.0, "threshold": 0.75, "pass": false },
    "roi_seed_std":        { "value": 0.000112, "threshold": 0.03, "pass": true },
    "convergence":         { "value": 0.2553, "threshold": 5.0, "pass": true },
    "worst_seed_roi":      { "value": -0.00278, "threshold": -0.02, "pass": true }
  }
}
```

- **E1 (positive_seed_ratio ≥ 0.75)**: FAIL — 全 seed 負の ROI
- **E2 (roi_seed_std ≤ 0.03)**: PASS — seed 間の分散は極めて小さい
- **E3 (convergence ≤ 5%)**: PASS — 30K 以降の ROI 変動は小さい
- **E4 (worst_seed_roi > -0.02)**: PASS — 最悪でも -0.0028

### ROI が負の分析
- トランザクションコスト 0.1% (Coincheck Maker fee) が 1 分足の価格変動幅を上回る
- モデルは実際にトレードしている (trade_count: 1001〜1455) — スタブ時代の ROI=0 とは本質的に異なる
- 50K steps は SAC としては短い (一般的な SAC 訓練は 500K〜1M)

---

## 6. レビュー依頼事項

### 6.1 【Critical】`sitecustomize.py` のデッドコード

`_prefer_local_package()` を無効化し `_replace_stub_with_filebacked()` の呼び出しを `pass` に置換したが、関数定義自体は残っている。

**質問**: 
- `_replace_stub_with_filebacked()` は今後使われる可能性があるか？ 完全削除すべきか？
- `sitecustomize.py` 自体を空にすべきか？他に担う役割があるか？

### 6.2 【Critical】`_sb3_test_stub/` の存続判断

テスト用にスタブを保持しているが、CI/CD 環境でも pip 版 SB3 がインストールされるなら不要。

**質問**:
- テストで SB3 の stub が必要なケースは残っているか？
- `sb3_compat.py` の `ensure_sb3_compat()` は import fallback としてのみ使われるべきか？
- `_sb3_test_stub/` を完全削除した場合のテスト影響は？

### 6.3 【High】`import_real_sb3()` の信頼性

`scripts/v460/lib/sac_common.py` の `import_real_sb3()` は `sys.path` 操作と `sys.modules` パージで強制的に pip 版を読み込む。

```python
def import_real_sb3() -> object:
    _sb3_keys = [k for k in sys.modules if k == "stable_baselines3" or k.startswith("stable_baselines3.")]
    for k in _sb3_keys:
        sys.modules.pop(k, None)
    # ... sys.path からプロジェクトルートを除去 ...
    sb3 = importlib.import_module("stable_baselines3")
    if not hasattr(sb3, "__version__"):
        raise ImportError("Loaded stub instead of real SB3")
```

**質問**:
- `sys.modules` パージ + `sys.path` 操作のアプローチは堅牢か？
- `__version__` 属性チェックはスタブ検出として十分か？
- より堅牢な検証方法 (e.g., `hasattr(sb3.SAC, 'action_space')`) は必要か？

### 6.4 【High】パフォーマンス修正の妥当性

- `inspect.signature()` キャッシュ: `_sig_cache` を `dict[object, ...]` で `self` に動的追加。これは `hasattr` チェックによる遅延初期化だが、`__init__` で初期化すべきか？
- `max_steps_per_episode=10_000` / `_CHECKPOINT_EVAL_MAX_STEPS=5_000`: これらの制限値は妥当か？短すぎて評価精度に影響しないか？

### 6.5 【Medium】閾値 0.10 の妥当性

HOLD ゾーンを 66.7% → 20% に縮小した。

**質問**:
- 0.10 は SAC の tanh 出力分布に対して適切か？
- `adaptive_threshold_mode` や `z_score_threshold` を導入すべきか？
- 閾値を学習パラメータ化する (`ThresholdManager` の拡張) アプローチの是非?

### 6.6 【Medium】訓練パラメータの改善方向

現在: `gamma=0.80, lr=3e-4, batch_size=256, total_timesteps=50K`

**質問**:
- 50K steps で SAC が十分学習できるか？推奨ステップ数は？
- 訓練時のトランザクションコストを 0% にして reward signal を強化すべきか？
- `curriculum_learning` の段階的導入は有効か？

---

## 7. レビュー対象ファイル一覧

| ファイル | 変更概要 | 行数 |
|---|---|---|
| `sitecustomize.py` | `_prefer_local_package()` 無効化 | 125行 (大幅削減) |
| `ztb/support/sb3_compat.py` | pip版優先 import に変更 | 50行 |
| `_sb3_test_stub/__init__.py` | 旧 `stable_baselines3/` からリネーム | 49行 |
| `configs/v460/experiments/g2_sac_train.yaml` | 閾値・learning_starts 変更 | 113行 |
| `scripts/v460/diagnose_sac_actions.py` | 新規: 行動分布診断ツール | 183行 |
| `scripts/v460/lib/sac_common.py` | OOS max_steps + `import_real_sb3()` | 247行 |
| `scripts/v460/lib/tasks/sac_train.py` | checkpoint eval 制限 | 427行 |
| `ztb/trading/environment/components/calculators/reward_calculator.py` | signature キャッシュ | 2214行 (変更箇所: L1080-1095) |
| `tests/unit/v460/test_356_g2_sac_blockers.py` | assertion 更新 | 微修正 |

---

## 8. Git Commit Chain (379# Session)

```
59b9301b0 379# docs: SB3スタブ修正セッションログ追記
321c25de7 379# Critical Fix: ローカルSB3スタブが本物をシャドウ → SAC訓練が無操作だった
c340e379d 379# Fix: 閾値 0.3333→0.10 + learning_starts 100→1000
2a8269695 379# Perf: evaluate_model_oos に max_steps_per_episode=10K 導入
7d5fad87d 379# Perf: inspect.signature キャッシュ化 + checkpoint eval 5Kステップ制限
```

`git diff --stat 7d5fad87d..59b9301b0`: 20 files changed, 473 insertions, 84 deletions

---

## 9. 特に注視すべき観点

1. **セキュリティ**: `sitecustomize.py` の `sys.path` 操作は安全か？ 他の依存パッケージに影響しないか？
2. **テスト分離**: テスト時に誤ってスタブが使われる再発リスクはないか？
3. **再現性**: `import_real_sb3()` は CI/仮想環境/Docker でも確実に動作するか？
4. **根本設計**: なぜスタブが最初に必要だったのか？テスト高速化のためなら、`conftest.py` での mock が適切では？
