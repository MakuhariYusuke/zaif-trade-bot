# v456 Revised Action Plan: 外部レビュー反映版実行計画

> **Version**: v456.1  
> **Date**: 2026-01-13  
> **Status**: Final Plan

---

## 1. エグゼクティブサマリー

外部AIレビュー（06_review_response.md）で指摘された**Critical Issues**を最優先で対処し、
段階的にリスクを管理しながらv456を実装する。

### 主要変更点
1. **優先順位の逆転**: GRU導入を最後に（Phase 1のMLP成功が前提条件）
2. **データ整合性の最優先化**: MTFリーク防止、正規化分離を Week 1 で完了
3. **Train-Live Parityの徹底**: フィルタリングは環境内部に統合

---

## 2. Critical Issues 対応表

| Issue ID | 内容 | 対応策 | 完了基準 | 担当Week |
|----------|------|--------|----------|----------|
| **C-1** | MTFリサンプリングの未来リーク | クローズドバーのみ使用 | テスト100%パス | Week 1 |
| **C-2** | 正規化パイプラインの混在 | グループ分離実装 | Sin/Cos歪みなし確認 | Week 1 |
| **C-3** | GRU + Off-Policy設計不足 | MLPベースライン優先 | GRUは Phase 2 以降 | Week 6+ |
| **M-1** | 報酬シェーピング支配 | キャリブレーション実装 | Shaping/PnL < 0.5 | Week 2 |
| **M-2** | FX/ベーシス無視 | USDJPY, USDTプレミアム追加 | 特徴量追加完了 | Week 2 |
| **M-3** | Train-Liveミスマッチ | 環境step()内統合 | 同一ロジック確認 | Week 3 |

---

## 3. 改訂版フェーズ計画

### Phase 0: データ整合性確保 (Week 1) ← **最優先**

#### 0-1. MTF リーク防止 (Day 1-2)
```python
# 実装タスク
tasks = [
    "get_mtf_closed_bar() 実装",
    "test_mtf_no_future_leak.py 作成",
    "バックテストデータでの検証",
]

# 完了基準
success_criteria = {
    "leak_test_pass": True,
    "edge_case_coverage": ["10:00境界", "00:00境界", "週末"],
}
```

#### 0-2. 正規化分離 (Day 3-4)
```python
# 実装タスク
NORMALIZATION_GROUPS = {
    "online_zscore": ["base_features", "global_continuous"],
    "no_scaling": ["cyclical_time", "regime_onehot", "mtf_categorical"],
}

# 検証コード
def test_normalization_separation():
    """Sin/Cos特徴量がスケーリングされていないことを確認"""
    time_sin = obs["time_hour_sin"]
    assert -1.0 <= time_sin <= 1.0, "Not normalized!"
    # mean/stdが変更されていないことを確認
```

#### 0-3. タイムゾーン統一 (Day 5)
```python
# 実装: validate_and_convert_timestamp()
# テスト: Naive timestamp拒否、UTC/JST変換一貫性
```

### Phase 1: 特徴量追加 + 非GRUベースライン (Week 2-3)

> **⚠️ 第2次レビュー対応**: 特徴量数を88に統一（02_feature_engineering_spec.md v456.2 参照）

#### 1-1. MTF + Time + Global 特徴量追加
| 特徴量グループ | 数 | 正規化 | 備考 |
|---------------|---|--------|------|
| Base (1m) | 30 | Online Z-Score | 既存維持 |
| MTF (5min/15min/1h) | 27 | No Scaling | カテゴリカル多め |
| Cyclical Time | 6 | No Scaling | Sin/Cos |
| Global Market | 9 | 6連続+3フラグ | USDT premium含む |
| Regime | 13 | No Scaling | One-Hot |
| Account | 3 | Pre-Norm | 既存維持 |
| **Total** | **88** | - | 統一済み |

#### 1-2. 報酬シェーピング キャリブレーション
```python
# 実装
calibrated_coeffs = auto_calibrate_shaping_coefficients(
    base_coefficients=REWARD_PARAMS_V456,
    sample_episodes=collect_sample_episodes(n=100),
    target_ratio=0.3,  # シェーピングはPnLの30%以下
)

# 検証
validation = validate_reward_scale_balance(sample_episodes, max_shaping_ratio=0.5)
assert validation["is_valid"], validation["recommendation"]
```

#### 1-3. MLP SAC 学習 + ベースライン確立
```python
# 学習設定
MLP_BASELINE_CONFIG = {
    "policy": "MlpPolicy",        # GRU不使用
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "total_timesteps": 500_000,
}

# 成功基準
BASELINE_SUCCESS_CRITERIA = {
    "sharpe_ratio": 0.3,           # 最低限
    "return_vs_v455": "better",    # -9.3%より改善
    "trade_count": ">= 100",       # 十分なサンプル
}
```

### Phase 2: フィルタリング統合 (Week 4-5)

#### 2-1. 環境内Soft Filter統合
```python
class FastIntradayEnvV456(FastIntradayEnv):
    def step(self, action):
        # 1. Soft Filter (ポジション調整)
        filtered_action, mult = self._apply_soft_filter(action)
        
        # 2. Calibration Gate (EV判定)
        gated_action, gate_info = self._apply_calibration_gate(filtered_action)
        
        # 3. 実行
        return super().step(gated_action)
```

#### 2-2. Train-Live Parityテスト
```python
def test_train_live_parity():
    """学習時と推論時で同一フィルタリングが適用されることを検証"""
    
    # 学習モード
    env_train = FastIntradayEnvV456(mode="train")
    _, _, _, _, info_train = env_train.step(test_action)
    
    # 推論モード（ライブ想定）
    env_live = FastIntradayEnvV456(mode="live")
    _, _, _, _, info_live = env_live.step(test_action)
    
    # フィルタリング結果が同一であること
    assert np.allclose(info_train["gated_action"], info_live["gated_action"])
```

### Phase 3: GRU導入（条件付き） (Week 6+)

#### 3-0. 導入条件チェック
```python
GRU_PREREQUISITE = {
    "mlp_baseline_sharpe": 0.3,        # Phase 1で達成必須
    "mlp_baseline_return": -0.05,      # v455より改善
    "sequence_replay_design": "done",  # 設計完了
    "burn_in_design": "done",          # burn-in設計完了
}

def check_gru_prerequisites():
    """GRU導入の前提条件を確認"""
    results = load_mlp_baseline_results()
    
    if results["sharpe"] < GRU_PREREQUISITE["mlp_baseline_sharpe"]:
        raise ValueError(
            f"GRU導入の前提条件未達成: Sharpe={results['sharpe']:.2f} < 0.3. "
            "Phase 1のMLP最適化を継続してください。"
        )
    
    return True
```

#### 3-1. シーケンスリプレイ実装
```python
# Burn-in方式
SEQUENCE_REPLAY_CONFIG = {
    "sequence_length": 60,
    "burn_in_length": 20,
    "overlap": 10,
}
```

---

## 4. 週次マイルストーン

### Week 1: データ整合性 ✅
- [ ] MTFリーク検出テスト作成・パス
- [ ] 正規化グループ分離実装
- [ ] タイムゾーン検証実装
- **Gate**: 全Critical Issueテストパス

### Week 2: 特徴量追加
- [ ] MTF 27特徴量追加（5min/15min/1h）
- [ ] Cyclical Time 6特徴量追加
- [ ] Global Market 9特徴量追加（6連続+3フラグ）
- **Gate**: 観測空間88次元、NaN/Inf検出なし

### Week 3: ベースライン学習
- [ ] 報酬シェーピングキャリブレーション
- [ ] MLP SAC学習開始
- [ ] バックテスト評価
- **Gate**: Sharpe > 0.3, Return > -5%

### Week 4: フィルタリング統合
- [ ] Soft Filter環境内統合
- [ ] Calibration Gate環境内統合
- [ ] Train-Live Parityテスト
- **Gate**: 同一ロジック確認

### Week 5: 検証・最適化
- [ ] フルバックテスト（60日以上）
- [ ] 統計的有意性検証
- [ ] ハイパラ微調整
- **Gate**: Sharpe > 0.5, Return > 0%（95%CI）

### Week 6+: GRU導入（条件付き）
- [ ] 前提条件チェック（Sharpe > 0.3）
- [ ] シーケンスリプレイ実装
- [ ] GRU SAC学習
- **Gate**: Sharpe > 1.0, Return > +5%（挑戦目標）

---

## 5. リスク管理

### 5.1. 早期撤退条件
```python
ABORT_CRITERIA = {
    "week_3_sharpe_below_0": "特徴量設計見直し",
    "week_4_train_live_mismatch": "環境実装見直し",
    "week_5_return_below_v455": "全体設計見直し",
}
```

### 5.2. フォールバック計画
| 状況 | フォールバック |
|------|--------------|
| MTFリークテスト失敗 | MTF特徴量を一時除外、1m特徴量のみで継続 |
| 報酬ハッキング発生 | シェーピング係数を全て0.1xに削減 |
| GRU収束せず | MLPベースラインで運用、GRUは次バージョン |

---

## 6. 成果物一覧

### コード
- [ ] `ztb/features/time/cyclical.py`
- [ ] `ztb/features/generators/multi_timeframe/v456_engine.py`
- [ ] `ztb/features/global_market.py` (拡張)
- [ ] `ztb/trading/environment/fast_intraday_env_v456.py`
- [ ] `ztb/trading/filters/soft_filter.py`

### テスト
- [ ] `tests/unit/features/test_mtf_no_future_leak.py`
- [ ] `tests/unit/features/test_normalization_separation.py`
- [ ] `tests/unit/features/test_timezone_handling.py`
- [ ] `tests/integration/test_train_live_parity.py`
- [ ] `tests/integration/test_reward_calibration.py`

### ドキュメント
- [x] `docs/v456/00_improvement_proposal.md` (改訂済)
- [x] `docs/v456/01_technical_specification.md` (改訂済)
- [x] `docs/v456/02_feature_engineering_spec.md` (改訂済)
- [x] `docs/v456/03_implementation_checklist.md` (改訂済)
- [x] `docs/v456/07_revised_action_plan.md` (本文書)

---

## 7. 承認と次のステップ

### 7.1. 承認依頼事項
1. Phase 0（データ整合性）の最優先化
2. GRU導入のPhase 3への延期
3. 統計的KPI基準の採用

### 7.2. 即時アクション
1. Week 1 Day 1: MTFリーク検出テスト作成開始
2. Week 1 Day 3: 正規化グループ分離PR作成
3. Week 1 Day 5: タイムゾーン検証PR作成

---

*本計画は外部レビュー（06_review_response.md）の指摘を全面的に反映したものです。*
