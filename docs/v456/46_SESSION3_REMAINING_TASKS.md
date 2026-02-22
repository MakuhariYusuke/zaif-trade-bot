# 残タスク分析 - Phase 4 & Session 3 計画

## 📋 残タスク一覧

### Phase 4 内での残タスク

**高優先度（既実装）✅**
- [x] Metrics Calculation Unification (Session 1)
- [x] Over-fitting Indicator Standardization (Session 1)
- [x] Window Splitting Validation Enhancement (Session 1)
- [x] TimeSeriesWindow Validation Strengthening (Session 1)
- [x] Checkpoint/Resume Implementation (Session 2)
- [x] Dependency Injection Framework (Session 2)
- [x] Exception Handling System (Session 2)
- [x] Test Directory Restructuring (Session 2)
- [x] E2E Integration Tests (Session 2)
- [x] Checkpoint ztb.utils 統合 (Session 2 - 本日)

**中優先度（保留中）⏳**
- [ ] **パフォーマンス最適化** (Session 3)
  - 50+ ウィンドウ評価での速度改善
  - マルチプロセッシング vs asyncio 検討
  - メモリプロファイリング

- [ ] **E2E テストデータ拡張** (Session 3)
  - 1000/2000/1500 bars での大規模テスト
  - 複数市場環境での検証

---

## 🎯 Session 3 計画（新規タスク）

### タスク1: パフォーマンス最適化
**優先度**: 中
**難易度**: 高
**推定工数**: 4-6時間

#### 背景
- 現在: 各ウィンドウ逐次評価（シングルスレッド）
- 問題: 50+ ウィンドウで 1ウィンドウ=30分 × 50 = 25時間+ の評価時間
- 目標: 並列化により 2-4時間へ短縮

#### 検討方式

| 方式 | 利点 | 欠点 | 対応性 |
|-----|------|------|--------|
| **Multiprocessing** | GIL回避可 | Checkpoint共有複雑 | ⭐⭐⭐ |
| **Asyncio** | I/O効率的 | CPU処理に弱い | ⭐⭐ |
| **Ray** | 分散対応 | 依存増加 | ⭐⭐⭐⭐ |
| **Threading** | シンプル | GILで効果薄 | ⭐ |

#### 実装フロー

```python
# 1. 現状測定（cProfile）
python -m cProfile -s cumtime evaluate_walk_forward.py > profile.txt

# 2. ボトルネック特定
# - 訓練時間: ~90% (SAC学習)
# - 評価時間: ~5%
# - チェックポイント: ~5%

# 3. 並列化戦略
# a) Window-level parallelization
#    各ウィンドウを独立プロセスで処理
#    
# b) Gradient accumulation + Mini-batch
#    複数ウィンドウの勾配を累積

# 4. Checkpoint同期
# - Manager による共有メモリ
# - ファイルベースの同期 (現在)
```

#### 実装チェックリスト
- [ ] cProfile による詳細測定
- [ ] Multiprocessing Pool 実装
- [ ] Checkpoint Manager の並列安全化
- [ ] エラーハンドリング (子プロセスクラッシュ時)
- [ ] Progress tracking for parallel evaluation
- [ ] テスト: 複数ウィンドウでの正確性検証

---

### タスク2: E2E テストデータ拡張
**優先度**: 中
**難易度**: 低
**推定工数**: 2-3時間

#### 背景
- 現在: 300/500/400 bars (小規模)
- 問題: WalkForwardSplitter の最小要件未達成
- 目標: 1000/2000/1500 bars での堅牢テスト

#### 実装フロー

```python
# 1. テストデータセット拡張
#    train: 1000 bars (約4日分 at 1H)
#    val:   2000 bars (約8日分)
#    test:  1500 bars (約6日分)

# 2. 追加テストケース
test_walk_forward_integration_large_dataset()
test_checkpoint_resume_large_scale()
test_parallel_evaluation_50_windows()

# 3. パフォーマンス測定
- 評価時間: expected ~30 seconds/test
- メモリ使用量: peak ~1.5 GB
- モデル精度: validation vs test の比較
```

#### 実装チェックリスト
- [ ] テストデータセット生成 (1000/2000/1500 bars)
- [ ] WalkForwardSplitter の 50 ウィンドウテスト
- [ ] メモリプロファイリング
- [ ] 評価時間測定
- [ ] モデル精度検証
- [ ] CI/CD統合テスト

---

### タスク3: 高度な再開戦略
**優先度**: 低
**難易度**: 中
**推定工数**: 3-4時間

#### 機能要件
1. **部分復元**: 指定ウィンドウ以降のみ再開
2. **複数CheckpointMerge**: 複数実行結果の統合
3. **差分チェックポイント**: Diff形式による容量削減
4. **バージョン管理**: Checkpoint互換性追跡

#### 実装サンプル

```python
class AdvancedResumeManager:
    """Advanced checkpoint management"""
    
    def resume_from_window(self, run_id, start_window_id):
        """指定ウィンドウから再開"""
        pass
    
    def merge_checkpoints(self, run_ids):
        """複数実行結果をマージ"""
        pass
    
    def apply_diff_checkpoint(self, base, diff):
        """差分チェックポイント適用"""
        pass
```

---

## 🔍 残タスク優先順位マトリックス

| タスク | 優先度 | 難易度 | インパクト | 推奨度 |
|--------|--------|--------|-----------|--------|
| **パフォーマンス最適化** | 🔴 高 | 🔴 高 | 🟢 中 | ⭐⭐⭐⭐ |
| **E2E テスト拡張** | 🟡 中 | 🟢 低 | 🟢 中 | ⭐⭐⭐ |
| **高度な再開戦略** | 🟡 中 | 🟡 中 | 🔵 低 | ⭐⭐ |
| **メモリ最適化** | 🟡 中 | 🔴 高 | 🟢 中 | ⭐⭐ |

---

## 🚀 推奨される実行順序

### Phase 3A: 安定性強化（1-2日）
```
1. E2E テストデータ拡張
   └─ 小スコープ、高確実性
   
2. 大規模データセットでの検証
   └─ 4-8 ウィンドウでの評価
```

### Phase 3B: パフォーマンス最適化（2-3日）
```
1. Profiling & Bottleneck Analysis
   └─ cProfile, memory_profiler
   
2. Multiprocessing 実装
   └─ Window-level parallelization
   
3. 負荷テスト（50+ windows）
   └─ 実測パフォーマンス評価
```

### Phase 3C: 高度機能（オプション）
```
1. 差分チェックポイント
2. 複数実行マージング
3. Checkpoint互換性管理
```

---

## 📊 成功基準

### パフォーマンス最適化
```
成功基準:
✓ 50ウィンドウ評価: <4時間以内
✓ メモリ: peak <2GB
✓ 正確性: 並列化前後で結果一致
✓ テスト: 単一プロセス版との比較で誤差 <0.01%
```

### テスト拡張
```
成功基準:
✓ 1000/2000/1500 bars テスト通過
✓ 複数ウィンドウ (10+) での検証通過
✓ Checkpoint/Resume の堅牢性確認
✓ CI: 60秒以内で完了
```

---

## 🔧 技術デット

### 既知の問題（低優先度）
- [ ] Unused imports in evaluation modules
- [ ] Deprecated API calls in older trainers
- [ ] Hardcoded paths in some test fixtures

### 技術負債（中期対応）
- [ ] Type hints coverage: 현재 85%, 목표 95%
- [ ] Docstring coverage: 現在 80%, 目標 95%
- [ ] Test coverage: 現在 75%, 目標 85%

---

## 📅 実装ロードマップ

```
Week of 2026-01-15
├─ Mon: E2E テストデータ拡張開始
├─ Tue-Wed: Profiling & Optimization 設計
├─ Thu-Fri: Multiprocessing 実装
└─ Weekend: 統合テスト & 検証

Week of 2026-01-22
├─ Mon-Tue: 50+ ウィンドウ実測
├─ Wed-Thu: ボトルネック最適化
├─ Fri: ドキュメント更新 & リリース準備
└─ Weekend: ソーク テスト (24時間運用)
```

---

## 🎓 Learning Outcomes (Session 3 実施時)

```
技術スキル向上:
✓ 並列処理パターン（Multiprocessing/asyncio）
✓ プロファイリング手法（cProfile/memory_profiler）
✓ 分散チェックポイント戦略
✓ 大規模モデル評価の最適化

プロジェクト成果:
✓ Walk-Forward 分析フレームワーク完成
✓ 本番対応 Checkpoint/Resume システム
✓ 高速・安定した評価パイプライン
✓ 拡張可能な訓練基盤
```

---

**策定日**: 2026-01-15
**状態**: 🟢 Ready for Session 3
**推定開始日**: 2026-01-20 (予定)

