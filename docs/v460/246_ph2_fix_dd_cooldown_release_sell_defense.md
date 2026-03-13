# 246# DD Halt Cooldown Release + Sell Defence Hardening

## 概要

245# 本番ログ分析で特定した 2 つの重大問題を対処する:

1. **DD Halt の過大な機会損失**: 4/6日で halt 発動 → 15h+ idle → 回復機会喪失
2. **Sell 側慢性的損失**: pass_pnl=-1.316bps, win_rate=34.4% (DEGRADED)

## 変更一覧

### A. DD Halt Cooldown Release (新機能)

**理論的根拠**: Optimal stopping theory — 無期限 halt の機会損失コストが、
lot 縮小での再開リスクを上回る時間閾値が存在する。

| パラメータ | デフォルト | 本番値 | 説明 |
|-----------|-----------|--------|------|
| `cooldown_release_sec` | 0 (無効) | 7200 (2h) | halt 後この秒数で部分解除 |
| `cooldown_release_lot_scale` | 0.3 | 0.3 | 解除時の lot 倍率 (30%) |

**動作フロー**:
1. DD hard halt 発動 (`daily_pnl_bps <= -50bps`)
2. `is_halted()` が True を返し、サイクルスキップ
3. `cooldown_release_sec` (7200s=2h) 経過
4. `is_halted()` が False を返す + `cooldown_released=True`
5. `get_cooldown_lot_scale()` が 0.3 を返す
6. `fill_cycle_executor` が lot を 30% に縮小して取引再開
7. 日替わりで全状態リセット

**変更ファイル**:
- `scripts/v460/lib/daily_drawdown_guard.py`: DailyDrawdownState に `cooldown_released` 追加、`is_halted()` にクールダウン遷移ロジック、`get_cooldown_lot_scale()` メソッド追加、export/import/metrics 対応
- `scripts/v460/lib/fill_config.py`: `dd_cooldown_release_sec`, `dd_cooldown_release_lot_scale` フィールド + YAML パース追加
- `scripts/v460/run_fill_test.py`: 2箇所の DailyDrawdownGuard コンストラクタにパラメータ配線
- `scripts/v460/lib/fill_cycle_executor.py`: `get_cooldown_lot_scale()` による lot 縮小適用

### B. Sell Defence Hardening (YAML 設定変更)

245# 分析に基づく sell 側防御パラメータの強化:

| パラメータ | 旧値 | 新値 | 理論根拠 |
|-----------|------|------|----------|
| `sell_guard.offset_floor` | 0.20 | 0.30 | Glosten-Milgrom: AS premium 増額 |
| `sell_dynamic_kill.threshold_bps` | -0.5 | -0.3 | 早期 kill で損失蓄積防止 |
| `trending_sell_offset_boost_factor` | 2.0 | 3.0 | Kyle 1985: trending_up は情報コスト大 |
| `toxic_fill_veto_threshold_bps` | -5.0 | -3.0 | tail risk 連鎖遮断の閾値引き下げ |

### C. テスト

- 11 件の新規テスト追加 (`TestCooldownRelease246`, `TestCooldownReleaseConfig246`)
- 既存テスト 3 件を YAML 変更に合わせて更新
- **3420 passed** (3409 → 3420, +11)

## 期待効果

- DD halt による日次 15h+ の idle 時間を 2h に短縮
- 残り時間を 30% lot で稼働 → 機会損失 ~85% 削減
- sell 側損失率の改善 (offset_floor 0.30 で -1.316bps → 推定 -0.8bps 以下)
- 連鎖損失の早期遮断 (toxic_veto -3.0bps)

---

*Implementation: 246#*
*Tests: 3420 passed*
