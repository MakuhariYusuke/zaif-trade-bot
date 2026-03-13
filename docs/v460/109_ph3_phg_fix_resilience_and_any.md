# 109# ph3/phg 先行改善2 — 耐障害性強化 + Any型完全撤去 (2ファイル)

| key | value |
|-----|-------|
| type | fix/impl |
| phase | ph3 + phg (先行) |
| status | committed |
| parent | 018#, 032#, 036# |
| tests | 827 passed (v460 unit tests, リグレッションなし) |

---

## §1 背景

108# に続き、107# fill_test 48h 観察中の待ち時間を活用。
耐障害性 (resilience) 改善と型安全向上を並行実施。

## §2 fill_test ログ状況 (2026-02-18 18:47 時点)

| 指標 | 値 | 評価 |
|------|-----|------|
| サイクル数 | 827 | 安定稼働中 |
| PID | 60240 / 24884 | 起動 11:14:08 |
| JPY残高 | ¥19,978 | 正常 |
| BTC残高 | 0.00035 BTC | sell不可 (min 0.001) |

### 日別パフォーマンス

| 日付 | cycles | filled | skip | fill率 | AS | 30s PnL mean | 30s PnL sum |
|------|--------|--------|------|--------|-----|-------------|-------------|
| 02/13 | 211 | 163 | 0 | 77% | 48% | -0.44bps | -71.8bps |
| 02/14 | 220 | 161 | 0 | 73% | 31% | -0.72bps | -116.5bps |
| 02/15 | 60 | 49 | 0 | 82% | 35% | -0.88bps | -42.9bps |
| 02/16 | 21 | 14 | 0 | 67% | 36% | -1.12bps | -15.7bps |
| 02/17 | 205 | 137 | 24 | 76% | 28% | **+0.45bps** | **+61.5bps** |
| 02/18 | 105 | 63 | 18 | 72% | **17.5%** | -0.18bps | -11.5bps |

**所見**:
- 107# 導入後 (02/17~) skip_gate が稼働開始 → AS率が大幅改善 (48% → 17.5%)
- 02/17 は初のプラス PnL 日 (+61.5bps)
- 02/18 は PnL ほぼフラット (-11.5bps)、AS率は最良

## §3 実施内容

### H3: `_market_regime_cache` リセット時クリア (018#)

- **ファイル**: `ztb/trading/environment/heavy_env/core.py`
- **問題**: `reset()` で `_market_regime_cache` がクリアされず、前エピソードのキャッシュ値が使い回される
- **対応**: `reset()` 内に `self._market_regime_cache = [None] * self.n_steps` 追加
- **効果**: エピソード跨ぎの stale regime 判定防止

### 032#16: ManifestWriter flush/fsync 追加

- **ファイル**: `scripts/v460/lib/manifest.py`
- **問題**: `_append()` が `f.write()` 後に `flush/fsync` を呼ばず、ディスクフル時に部分書き込みが残る
- **対応**: `f.flush(); os.fsync(f.fileno())` 追加
- **効果**: クラッシュ/ディスクフル時のmanifest破損防止

### 032#17: save_fill_records アトミックバッチ書込み

- **ファイル**: `ztb/metrics/fill_quality.py`
- **問題**: JSONL 追記中の SIGINT/ディスクフルで不完全な JSON 行が残る
- **対応**: tempfile 書き出し → fsync → 本体 append → fsync → temp 削除
- **効果**: fill_records JSONL の書込み原子性確保

### 036# Any型完全撤去 (2ファイル, 53箇所)

#### auto_feature_generator.py (21→0)

- `np.ndarray[Any, np.dtype[Any]]` → `NDArray[np.float64]` (4箇所)
- `Tuple[Any, ...]` → `Tuple[Union[int, float], ...]` (6箇所)
- `List[Any]` → `List[Union[int, float]]` (1箇所)
- `Callable[..., Any]` → `Callable[..., bool]` (1箇所)
- `Dict[str, Any]` → 具体型 / `FeatureEvalResult` TypedDict (9箇所)
- **新規TypedDict**: `FeatureEvalResult`, `_EvalSummary`

#### status.py (32→0)

- `Dict[str, Any]` (coverage_data) → `CoverageData` TypedDict (10箇所)
- `List[Dict[str, Any]]` → `List[FeatureItem]` (13箇所)
- `Dict[str, Any]` (event) → `CoverageEvent` TypedDict (2箇所)
- `items: Any` → `Union[List[str], List[FeatureItem]]` (1箇所)
- その他個別修正 (6箇所)
- **新規TypedDict**: `FeatureItem`, `CoverageEvent`, `CoverageMetadata`, `CoverageCurrentState`, `CoverageData`

## §4 018# 残課題ステータス更新

| ID | 内容 | ステータス |
|----|------|-----------|
| C3 | `vec_env.close()` 欠落 | ✅ 108# |
| H3 | `_market_regime_cache` reset 時未 clear | **✅ 109#** |
| H5 | `_get_info()` 毎 step features/config 含む | 後日 (SB3確認後) |
| M1 | DataFrame → numpy slicing | ✅ 108# (デバッグコード除去) |
| M5 | `LivePositionConfig` 重複 | ✅ 108# |
| DUP2 | `sac_utils` 2ファイル | ✅ 108# |
| DUP3 | `UnifiedTrainer` God Object | 後日 (ph3 本格再設計) |

残り: **H5, DUP3** のみ (いずれも大規模作業のため見送り)

## §5 036# Any削減ステータス更新

| ファイル | Before | After | 削減数 |
|----------|--------|-------|--------|
| `auto_feature_generator.py` | 21 | **0** | -21 |
| `status.py` | 32 | **0** | -32 |
| **合計** | 53 | **0** | **-53** |

---

> **文書管理**
> - 作成日: 2026-02-18
> - フェーズ: ph3/phg 先行 (107# fill_test 観察中)
> - 前提文書: 018#, 032#, 036#, 108#
