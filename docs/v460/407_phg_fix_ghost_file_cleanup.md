# 407# Ghost File Cleanup + Performance + Stability

**Date**: 2026-03-13  
**Commit**: `7dee401d0`  
**Phase**: phg (フェーズ横断)  
**Type**: fix / perf  
**前提**: 406# セルフレビューで発見されたゴーストファイル問題 + 性能・安定性改善

---

## 0. 背景

406# のセルフレビューで、session037（Codex）が `ztb/trading/environment/` 配下の71ファイルを削除したまま git に再追跡していない「ゴーストファイル問題」が発覚。  
`git clone` して環境を再構築すると、これらのファイルが欠落し import エラーが頻発する致命的状態であった。  
本チケットではこの問題の完全解決に加え、コードレビュー中に発見された S4 バグと P1/P3/P5 パフォーマンス改善を同時実施した。

---

## 1. ゴーストファイル整理（71ファイル再追跡）

### 1.1 問題の構造
- session037 が `ztb/trading/environment/` 配下のファイルを物理削除したが、`git add` で追跡し直さなかった
- ローカルでは `.pyc` キャッシュ等で動作するが、`git clone` した新環境では全て欠落
- 影響範囲: `components/`, `heavy_env/`, `utils/`, トップレベル env ファイル群

### 1.2 対処
- 66ファイルを `git add` で再追跡（新規追加 64 + 修正 2）
- 再追跡対象の主要ディレクトリ:
  - `ztb/trading/environment/__init__.py`
  - `ztb/trading/environment/bridge.py`
  - `ztb/trading/environment/constants.py`
  - `ztb/trading/environment/environment.py`
  - `ztb/trading/environment/factory_v456.py`
  - `ztb/trading/environment/fast_intraday_env.py`
  - `ztb/trading/environment/fast_intraday_env_v456.py`
  - `ztb/trading/environment/schema_env_factory.py`
  - `ztb/trading/environment/types.py`
  - `ztb/trading/environment/components/` (20+ ファイル)
  - `ztb/trading/environment/heavy_env/` (mixins 含む)
  - `ztb/trading/environment/utils/`

---

## 2. デッドコード11ファイル → アーカイブ

orphaned な `reward/` コンポーネント 10ファイル + `fixed_ttl_wrapper.py` を `archived/dead_reward_components/` に移動。

| # | ファイル | 理由 |
|---|---------|------|
| 1 | `action_penalty.py` | RewardCalculator 未使用 |
| 2 | `base_reward_calculator.py` | 旧基底クラス, 継承なし |
| 3 | `diversity_bonus.py` | 未参照 |
| 4 | `drawdown_penalty.py` | 未参照 |
| 5 | `fixed_ttl_wrapper.py` | TTL ラッパー廃止済 |
| 6 | `growth_bonus.py` | 未参照 |
| 7 | `pnl_focused_reward.py` | 未参照 |
| 8 | `position_penalty.py` | 未参照 |
| 9 | `stagnation_penalty.py` | 未参照 |
| 10 | `win_rate_bonus.py` | 未参照 |
| 11 | `win_streak_bonus.py` | 未参照 |

`reward/__init__.py` をアクティブコンポーネントのみにクリーンアップ:
- `BalanceCurriculumManager`, `LongTermMetrics`, `MTFWeightManager`
- `OpportunityCostPenaltyCalculator`, `TrendDetector`, `UnrealizedLossPenaltyCalculator`

---

## 3. バグ修正

### S4 CRITICAL: タプル代入バグ
- **場所**: `RewardCalculator.calculate_reward_simple()`
- **問題**: `continuous_action_value = (None,)` — タプルとして代入されていた
- **修正**: `continuous_action_value = None` に変更
- **影響**: 後続の比較演算 (`is None`) が False を返し、不正な報酬計算パスに入る可能性

---

## 4. パフォーマンス改善

### P1: `_get_nested_setting()` キャッシュ導入
- **問題**: 毎ステップ約30回呼ばれる設定取得で毎回文字列パース
- **対処**: `_settings_cache: dict` を `RewardCalculator` に追加、初回読み取り時にキャッシュ
- **効果**: 文字列解析コストを根絶

### P3: 二重GC統合
- **問題**: `core.py` に `DEFAULT_GC_STEP_INTERVAL` によるGCと `MemoryManager` のGCが二重実行
- **対処**:
  - `DEFAULT_GC_STEP_INTERVAL` 定数を削除（コメント化）
  - GCスケジューリングを `MemoryManager.should_collect_garbage_at_step()` に一元化
  - `streaming.py`: 直接 `gc.collect()` → `memory_manager.collect_garbage()` 委譲
- **効果**: GC呼び出し回数の削減と管理の一元化

### P5: `collect_garbage()` 戻り値型修正
- **問題**: `collect_garbage()` / `collect_garbage_aggressive()` が `None` 返却
- **対処**: `int` (回収オブジェクト数) を返すように修正
- **効果**: GCの効果を測定可能に

---

## 5. 安定化

| 項目 | 内容 |
|------|------|
| `should_collect_garbage` 廃止 | プロパティを `is_gc_enabled` に統一 |
| `streaming.py` GC委譲 | `gc.collect()` の直接呼出を `memory_manager` 経由に変更 |
| `test_bankruptcy_drawdown` 対応 | `is_gc_enabled` API に合わせてテスト更新 |

---

## 6. テスト

- **新規テスト**: 11件 (`tests/unit/v460/test_407_ghost_cleanup.py`)
  - `TestS4TupleBugFix` (1): ソースコード検査でタプル代入バグの再発防止
  - `TestP1SettingsCache` (3): キャッシュ初期化・蓄積・一貫性検証
  - `TestP3UnifiedGC` (3): デフォルト間隔・無効化・二重GC排除
  - `TestP5CollectGarbageReturn` (2): 戻り値型チェック
  - `TestDeadCodeRemoval` (2): orphan import 消去・アクティブ import 確認
- **追加テスト**: `test_bankruptcy_drawdown.py` — `is_gc_enabled` 対応
- **結果**: 4768 passed, 33 skipped

---

## 7. 影響範囲まとめ

| カテゴリ | 変更数 | リスク |
|----------|--------|--------|
| ゴーストファイル再追跡 | 64 A + 2 M | LOW (既存ローカルファイルの git 追跡のみ) |
| デッドコード アーカイブ | 11 files → `archived/` | LOW (未参照コード) |
| S4 バグ修正 | 1 行 | MEDIUM (報酬計算パスに影響) |
| P1 キャッシュ | `_settings_cache` 追加 | LOW |
| P3 GC統合 | 3 箇所 | LOW |
| P5 戻り値修正 | 2 メソッド | LOW |
| テスト | 11+1 | — |

---

## 8. 残課題 → 408# へ

- RewardCalculator God Object 問題 (2252行/50メソッド) → 408# Codex 調査で深堀り
- `simplified_reward_calculator.py`, `reward/metrics.py`, `bridge.py` の追加アーカイブ候補 → 408# §9.1
- `reward/` と `rewards/` ディレクトリ統合 → 408# §12 P2
