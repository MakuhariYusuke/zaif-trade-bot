# 382# Gemini レビュー依頼: SB3 スタブ修正・SAC 訓練パイプライン・Next Steps

## 0. コンテキスト

Gemini は 381# で fill_test 分析の Ghost Metric Paradox を指摘し、
380# を Rev.2 に軌道修正した実績がある。今回は SAC 訓練パイプライン側の
致命的バグ修正に対するレビューを依頼する。

**前回 (381#) の Gemini の仕事**:
- `forced_buy_delay` が 3/9 に撤廃済みなのに 7 日間データで集計していた問題を指摘
- YAML 設定 (`sell_hour_offset_boost`, `offset_ceiling_ratio_buy`) の適用を承認・即時実行

---

## 1. 今回のバグ概要 (379# Session)

### 1.1 致命的問題

プロジェクトルートの `stable_baselines3/` ディレクトリがダミースタブで、
pip 版 SB3 2.7.0 をシャドウ。`sitecustomize.py` がスタブを強制ロードした結果、
**全ての SAC 訓練が no-op** (learn=何もしない, predict=常に0, save=何もしない) だった。

### 1.2 修正

| 修正箇所 | 変更内容 |
|---|---|
| `stable_baselines3/` → `_sb3_test_stub/` | リネームでシャドウ回避 |
| `sitecustomize.py` | `_prefer_local_package()` を `return False` に |
| `ztb/support/sb3_compat.py` | pip 版優先 import、失敗時のみ stub fallback |
| `g2_sac_train.yaml` | `threshold: 0.3333→0.10`, `learning_starts: 100→1000` |
| `reward_calculator.py` | `inspect.signature()` キャッシュ化 |
| `sac_train.py` | checkpoint eval 5K ステップ制限 |
| `sac_common.py` | OOS eval 10K ステップ制限 |

---

## 2. 修正後の訓練結果

```
┌──────┬─────────────┬──────────┬──────────────┬─────────────┬──────────┐
│ Seed │ Best ROI    │ Final    │ OOS ROI      │ trade_count │ Time     │
│      │ (checkpoint)│ ROI(50K) │              │  (3 ep)     │          │
├──────┼─────────────┼──────────┼──────────────┼─────────────┼──────────┤
│ 42   │ -0.0008(20K)│ -0.0019  │ -0.00253     │ 1001        │ 27.0 min │
│ 123  │ -0.0016(30K)│ -0.0024  │ -0.00260     │ 1401        │ 32.6 min │
│ 456  │ -0.0003(35K)│ -0.0007  │ -0.00256     │ 1109        │ 31.6 min │
│ 789  │ -0.0013(50K)│ -0.0013  │ -0.00278     │ 1455        │ 34.0 min │
└──────┴─────────────┴──────────┴──────────────┴─────────────┴──────────┘
```

**G2 Gate**: FAIL (E1: positive_seed_ratio = 0.0 < 0.75)
- E2 (seed std): PASS (0.000112 < 0.03)
- E3 (convergence): PASS (0.2553 < 5.0)
- E4 (worst ROI): PASS (-0.0028 > -0.02)

→ モデルは確かにトレードしているが、**トランザクションコスト 0.1% が 1 分足の微小な価格変動を上回り、全 seed で ROI が負**。

---

## 3. Gemini への質問・レビュー項目

### 3.1 【Critical】SB3 スタブの完全削除判断

`_sb3_test_stub/` を保持する理由は「テスト互換性」だが:
- 現在の pip 環境には SB3 2.7.0 がインストール済み
- テストで SB3 の import が必要な場合は本物が使える
- `sb3_compat.py` の `ensure_sb3_compat()` はテスト時の import フォールバック

**Gemini に判断を求める**:
1. `_sb3_test_stub/` を完全削除して問題ないか？
2. 削除する場合、`sb3_compat.py` の fallback コードも不要か？
3. `sitecustomize.py` を空ファイルにすべきか、完全削除すべきか？

### 3.2 【Critical】ROI 改善の戦略

全 seed で ROI が負 (-0.0003 〜 -0.0028) である。原因仮説:

**仮説 A**: トランザクションコスト 0.1% が reward signal を圧殺
- 1分足の典型的な価格変動: ~0.01%〜0.05%
- コスト 0.1% を超えるには ~10bps 以上の値動きが必要
- 提案: 訓練時 cost=0% で学習し、評価時に cost=0.1% を適用？

**仮説 B**: 50K ステップでは学習不足
- SAC は通常 500K〜1M ステップ推奨
- しかし 1M ステップ × 4 seeds は訓練時間 ~12 時間
- 提案: 100K → 200K → 500K の段階的スケール？

**仮説 C**: 特徴量が弱い
- 17 OHLCV 特徴量は基本的な市場統計量
- v460 microstructure features (板情報等) は ph4 予定
- 提案: ph4 まで待つか、feature engineering を先行すべきか？

**仮説 D**: gamma=0.80 が短期すぎる
- γ=0.80 は ~5 ステップ先の割引
- 1分足で 5 分先のみを考慮 → 短期ノイズに振り回される？
- 提案: γ=0.95 や γ=0.99 との比較実験？

**Gemini の見解を求める**: 上記 A〜D のどれが最も効果的か、または他のアプローチがあるか？

### 3.3 【High】`import_real_sb3()` のアーキテクチャ

`sac_common.py` に `import_real_sb3()` がある。`sys.modules` パージと `sys.path` 操作で
pip 版 SB3 を強制ロードする防御的コード。

```python
def import_real_sb3() -> object:
    # sys.modules から sb3 関連を全て除去
    _sb3_keys = [k for k in sys.modules if k.startswith("stable_baselines3")]
    for k in _sb3_keys:
        sys.modules.pop(k, None)
    # sys.path からプロジェクトルートを一時除去
    for p in list(sys.path):
        if "site-packages" not in p and (p == "." or p == _project_root):
            sys.path.remove(p)
    try:
        sb3 = importlib.import_module("stable_baselines3")
        if not hasattr(sb3, "__version__"):
            raise ImportError("Loaded stub")
    finally:
        # sys.path を復元
```

**質問**:
- スタブを完全削除すれば、この関数自体が不要になるのでは？
- `sys.modules` パージは副作用が大きい。もっとシンプルな方法はないか？

### 3.4 【High】パフォーマンス修正の設計品質

1. **`_sig_cache` の遅延初期化**:
```python
if not hasattr(self, '_sig_cache'):
    self._sig_cache: dict[object, tuple[bool, frozenset[str]]] = {}
```
`__init__` で初期化すべきか？ `hasattr` チェックの毎ステップコストは？

2. **`max_steps` ハードコーディング**:
- checkpoint eval: 5,000 ステップ
- OOS eval: 10,000 ステップ

データセット全体は 1.2M 行 (train: 973K, val: 243K)。
5K/10K は全体の 0.5%/1% に過ぎない。これで十分な評価精度が得られるか？

### 3.5 【Medium】G2 Gate 基準の妥当性

現在の G2 Gate:
- **E1**: positive_seed_ratio ≥ 0.75 (= 3/4 seeds が正の ROI)
- **E2**: roi_seed_std ≤ 0.03
- **E3**: convergence (30K 以降 ROI 変動 ≤ 5%)
- **E4**: worst_seed_roi > -0.02

E2〜E4 は PASS しているが E1 が FAIL。

**質問**:
- E1 の基準 (0.75) は厳しすぎるか？ 0.50 (= 2/4 seeds) に緩和すべきか？
- それとも E1 を達成できないモデルは本当に使い物にならないのか？
- ROI ではなく Sharpe ratio や win rate で gate を設計すべきか？

### 3.6 【Medium】訓練アーキテクチャの Next Steps

現在無効化されている機能:
- `curriculum_learning` / `curriculum_stage`
- `hybrid_config` (entry/exit override)
- `domain_randomization`
- `advanced_market_regime`
- `signal_guidance_enabled`
- `adaptive_threshold_mode`

これらは「365# 報酬非定常化回避」で意図的に無効化されている。

**質問**: ROI 改善のために、どの機能を優先的に有効化すべきか？
特に:
1. `curriculum_learning`: 段階的に取引コストを導入する？
2. `adaptive_threshold_mode`: 閾値を行動分布に適応させる？
3. `domain_randomization`: 汎化性能向上のためのノイズ注入？

---

## 4. 381# からの連続性

381# で Gemini が指摘した key principles:
- **「環境の破壊的変更を跨いだデータ集計は分析を汚染する」**
- **Ghost Metric Paradox の回避**

今回の SB3 スタブ問題も同根:
- **「テスト用のモック／スタブが本番パスに混入すると、結果全体が汚染される」**
- スタブの存在自体が「正常な訓練が行われている」という誤認を生んでいた

381# の教訓を SAC 訓練パイプラインに拡張適用できているか、確認を求める。

---

## 5. レビュー対象ファイル

優先度順:
1. `sitecustomize.py` (125行) — 修正の核心
2. `ztb/support/sb3_compat.py` (50行) — import ロジック
3. `_sb3_test_stub/__init__.py` (49行) — 削除判断
4. `configs/v460/experiments/g2_sac_train.yaml` (113行) — パラメータ妥当性
5. `scripts/v460/lib/sac_common.py` (247行) — `import_real_sb3()` + OOS eval
6. `scripts/v460/lib/tasks/sac_train.py` (427行) — checkpoint eval
7. `scripts/v460/diagnose_sac_actions.py` (183行) — 診断ツール品質
8. `ztb/trading/environment/components/calculators/reward_calculator.py` (L1080-1095) — sig cache

---

## 6. Git 情報

**Commit Range**: `7d5fad87d..59b9301b0` (5 commits)
**Diff**: 20 files changed, 473 insertions, 84 deletions (スタブリネーム分含む)
**ブランチ**: `main` (HEAD)
