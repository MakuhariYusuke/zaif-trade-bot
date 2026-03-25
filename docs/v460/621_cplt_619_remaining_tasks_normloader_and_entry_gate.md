# 621# 619# 残課題解消: NormLoader 推論統合 + entry_gate observe 接続

- **日付**: 2026-03-25
- **著者**: Copilot
- **種別**: impl / config
- **目的**: 619# §4 の残課題 T1〜T4 を解消し、Sidecar 推論の正規化精度と entry_gate observe を改善

---

## §1 概要

619# §4 で「次のステップ (620#以降)」として列挙された 4 課題:

| # | 課題 | 本 621# での対応 |
|---|------|-----------------|
| T1 | NormLoader → Sidecar 推論接続 | ✅ 実装完了 |
| T2 | Feature Parity 実証 | ✅ 診断ログ統合 (cos_sim + max_diff) |
| T3 | eDRC 有効化判断 | ⏸️ 時期尚早（前提条件未達） |
| T4 | entry_gate 本稼働 | ✅ observe 接続完了 |

---

## §2 T1: NormLoader → Sidecar 推論統合

### 変更前

`_update_sidecar_signal()` は `_get_latest_obs(env)` で訓練環境の OnlineScaler (Welford running stats) 経由の正規化済み obs を取得していた。env オブジェクト依存であり、将来の env 非依存推論に対応不可。

### 変更後

`_update_sidecar_signal()` に `train_df` パラメータを追加。以下の優先順位で obs を取得:

1. **NormLoader パス (優先)**: `train_df` から最終行の生特徴量を抽出 → `NormLoader(norm_path).normalize()` で Z-score + ±5.0 clip
2. **env パス (フォールバック)**: NormLoader 非該当時は従来の `_get_latest_obs(env)`

NormLoader は `_export_feature_norms()` が retrain 時に出力する `norm.json` (バッチ統計: mean/std) を使用。OnlineScaler の Welford running stats よりも安定した正規化を提供。

### 正規化パラメータの整合性

| パラメータ | OnlineScaler (HeavyEnv) | NormLoader | 一致 |
|-----------|------------------------|------------|------|
| clip | 5.0 | 5.0 | ✅ |
| epsilon | 1e-5 | 1e-10 | ≈ (影響なし) |
| 統計量 | Welford running | バッチ固定 | ≈ (訓練終了時は収束) |

---

## §3 T2: Feature Parity 診断

NormLoader パス使用時、env パスの obs も同時に取得し、以下を比較ログ出力:

```
[621#] NormLoader parity: cos_sim=0.999987 max_diff=0.0142 dim=17
```

- **cos_sim**: 1.0 に近ければ方向が一致（0.999+ を期待）
- **max_diff**: 最大要素差（0.1 以下が正常）
- **dim mismatch 時**: env obs にトラッカー追加特徴量がある場合は警告を出し env fallback

この診断が retrain ごとに出力されるため、Train-Serve Skew の継続的モニタリングが自動化される。

---

## §4 T3: eDRC 有効化判断 — 時期尚早

以下の前提条件が未達のため、有効化を見送り:

1. **Side 別 `edrc_c_base` 未実装**: 現在は buy/sell 共通 `c_base=0.40` だが、sell の逆選択率 (36%) が高く buy (20%) と同一 ceiling は不適切
2. **入力ウィンザライゼーション未実装**: σ_bps / OFI の extreme values による ceiling 暴発リスク (586#)。589# の指数クリップは応急措置
3. **検証データ不足**: 575# 有効化時のテレメトリバグにより有効 fill わずか 2-3 件。統計的評価が一切できていない

### 有効化の前提条件

- [ ] `edrc_c_base_buy` / `edrc_c_base_sell` のサイド別パラメータ導入
- [ ] σ_bps, OFI の percentile ウィンザライゼーション (input-stage defense)
- [ ] テレメトリ正常環境での A/B 検証 (48h × 50+ fills)

---

## §5 T4: entry_gate observe 接続

### 変更内容

`configs/v460/fill_test.yaml` に CalibrationMap パスを再設定:

```yaml
entry_gate_enabled: false                    # observe モード維持
entry_gate_calibration_map_path: "models/v460/entry_gate_calibration.json"
```

606# で接続 → revert で切断されていた CalibrationMap を再接続。`enabled: false` なので EV≤0 でもブロックせず、ログ出力のみ。

### CalibrationMap 現状 (2026-03-25)

| セグメント | n_eff | p_win (推定) | 評価 |
|-----------|-------|-------------|------|
| global | ~200 | 0.516 | 十分 |
| ranging | ~168 | 0.479 | 十分 |
| ranging_Buy | ~96 | 0.430 | ⚠️ 低い |
| ranging_Sell | ~73 | 0.555 | 十分 |
| trending_down | ~17 | 0.659 | n_min 未達 |
| trending_up | ~19 | 0.660 | n_min 未達 |

### 本稼働 (enabled: true) の判断基準

observe ログ蓄積後、以下を評価して判断:

1. BLOCK 率 10-30% が適正範囲
2. BLOCK サイクルの仮想 PnL が負（正しいブロック判断）
3. PASS サイクルの PnL が現行以上

---

## §6 テスト結果

| テスト | 結果 |
|--------|------|
| NormLoader ユニットテスト (12件、新規) | ✅ 全パス |
| SAC retrain scheduler (45件) | ✅ 全パス |
| 全テストスイート (2237件) | ✅ 全パス (127 skipped) |

---

## §7 変更ファイル一覧

| ファイル | 内容 |
|---------|------|
| `scripts/v460/ml/sac_retrain_scheduler.py` | T1: NormLoader 推論パス + T2: Feature Parity 診断 |
| `configs/v460/fill_test.yaml` | T4: `entry_gate_calibration_map_path` 再設定 |
| `tests/unit/features/test_norm_loader.py` | NormLoader ユニットテスト (新規) |
