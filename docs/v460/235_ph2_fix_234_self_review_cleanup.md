# 235# 234# セルフレビュー: dead param cleanup・duty_cycle guard・escalation fixes

## 概要

234# (Gate bypass 廃止・縮退清算) のセルフレビューで発見した 7 件のコード品質問題を修正。

## 修正内容

### C-6 [HIGH]: duty_cycle=1 で永久 skip バグ

**問題**: `(counter % 1) != 1` は常に True → duty skip が永久発動し1回も実行されない。
**修正**: `max(config, 1)` ガード追加。`duty_cycle=1` は「毎回実行」として正しく動作。

### B-5 [HIGH]: freeze/cooldown 消化後の即再発動ループ

**問題**: freeze 消化後に `_one_sided_consecutive_count` が高いまま → 即座に再 freeze。
**修正**: freeze/cooldown 発動時にカウンタを `_os_limit` まで巻き戻し。
消化後は limit 地点から再カウントし、offset 分の猶予を確保。

### B-1 [MEDIUM]: dead parameter cleanup

**問題**: 234# で `not balance_forced` を全削除した結果、6 メソッドで `balance_forced` 引数が未使用。
**修正**: シグネチャから削除し、`evaluate()` 側の呼び出しも整理。
- `_check_unknown_regime_buy`, `_check_unknown_regime_sell`
- `_check_ranging_buy_low_vol`, `_check_trending_sell`
- `_check_buy_dynamic_kill`, `_check_sell_dynamic_kill`

### C-3 [MEDIUM]: spread 不明時の degraded offset ガード

**問題**: `spread_at_order` が None/0 の場合、offset は 3 倍に拡大されるが価格は再計算されない。
**修正**: `else` ブランチに警告ログを追加し、不整合状態を明示化。

### C-5 [LOW]: YAML パース型安全

**問題**: `degraded_liquidation_enabled` に `bool()` キャストなし。
**修正**: `bool(止血["degraded_liquidation_enabled"])` に変更。

### B-4 [LOW]: duty counter トグルリセット改善

**問題**: degraded → normal → degraded の遷移でカウンタが毎回リセット → duty cycle 制限が実質無効化の可能性。
**修正**: リセット時にログを出力し、運用時の監視を可能に。

### B-2 [LOW]: dead config DEPRECATED 注釈

**問題**: `balance_forced_apply_trending_offset` が実コードで参照ゼロだが注釈なし。
**修正**: `# DEPRECATED: 234# dead config` + `# TODO(235#)` 注釈追加。

## 変更ファイル

| ファイル | 変更 |
|---------|------|
| `cycle_gate_aggregator.py` | 6 メソッドから `balance_forced` 引数削除、呼び出し側整理 |
| `fill_loop_orchestrator.py` | duty_cycle max ガード、freeze/cooldown カウンタ巻き戻し、リセットログ |
| `fill_cycle_executor.py` | spread 不明時の degraded offset 警告ログ |
| `fill_config.py` | bool キャスト、DEPRECATED 注釈 |
| `test_234_gate_bypass_removal.py` | +10 テスト (dead param, duty_cycle, deprecated) |

## テスト結果

```
3195 passed, 0 failed
```
