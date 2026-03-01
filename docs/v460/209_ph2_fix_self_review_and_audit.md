# 209# セルフレビュー + コード監査 — 8件修正

> **日付**: 2026-03-02  
> **前提**: 207#/208# 実装直後のセルフレビュー + コードベース全体の監査  
> **コミット**: `da7698e98`

---

## 1. 背景

207# (堅牢性修正 5件 + one-sided 制限) と 208# (Velocity SSOT) の実装完了後、
以下の 2 観点からコードレビューを実施した:

1. **セルフレビュー**: 207#/208# で変更した箇所の論理エラー・エッジケース・型安全性
2. **コード監査**: 既存コードベース全体のリスク点・欠陥の発見

---

## 2. セルフレビュー発見事項 (5件)

### H-1: Toxic Veto 両サイド封鎖時のデッドロック (HIGH)

| 項目 | 内容 |
|---|---|
| 問題 | 両サイドが同時に veto 封鎖されると `continue` で skip し、サイクル末尾の veto デクリメント処理に到達しない。UTC 日替わりまで最大24時間のデッドロック |
| 発生条件 | buy veto 残2 + sell veto 残3 のような同時封鎖状態 |
| 修正 | both-blocked の `continue` パスにもデクリメント処理を追加 |

### M-2: Config バリデーション欠落 (MEDIUM)

`one_sided_consecutive_limit < 0` や `interval_mult <= 0` が未検証。
`__post_init__` にバリデーション追加。

### M-3: `_one_sided_consecutive_count` 日替わり未リセット (MEDIUM)

`maybe_reset_day()` ブロックで toxic_veto はリセットしていたが、
one-sided カウンタは忘れていた。一貫性のためリセット追加。

### M-4: Docstring 境界条件不一致 (MEDIUM)

`compute_instant_velocity_bps` の docstring が `dt > max_dt` (超えた) と記述していたが
実装は `dt >= max_dt` (以上)。docstring を「以上」に修正。

### L-1: ホットパス内 import (LOW)

`maker_price.py` の `compute()` 内で `from velocity_math import ...` していたものを
モジュールトップレベルに移動。循環 import の心配なし (leaf module)。

---

## 3. コード監査発見事項 (3件修正 + 残課題記録)

### H1: HealthMonitor severity overwrite (HIGH)

| 項目 | 内容 |
|---|---|
| 問題 | RSS が critical レベルでも disk_free_warn に該当すると `status["level"]` が `"warning"` に上書きされ、OOM 検知が失敗する |
| 修正 | severity escalation: `if status.get("level") != "critical"` ガード追加 |

### H5: コアタイミングパラメータ未検証 (HIGH)

`cycle_interval_sec`, `poll_interval_sec`, `order_timeout_sec` が 0 以下でもクラッシュせず
タイトループや `asyncio.sleep(negative)` を引き起こす。
`__post_init__` に `> 0` バリデーション追加。

### M4: Sleep 乗数上限なし (MEDIUM)

`interval × soft_dd_mult × loss_cooldown × one_sided_mult` の積で最大 2160秒 (36分) に達し得る。
`max_cycle_sleep_sec: float = 600.0` フィールドを追加し、10分でキャップ。

### M6: 動的属性のクラスレベル宣言欠落 (MEDIUM)

`_in_hard_skip_hour`, `_halt_iter_count` が `getattr(self, ..., default)` で動的生成。
クラスレベル宣言に追加し、mypy 検出・IDE 補完を有効化。

---

## 4. 監査で検出した未修正残課題

| ID | 重要度 | 内容 | 理由 |
|---|---|---|---|
| H2 | HIGH | Hot-reload 後 MakerPrice が旧 FFD 参照を保持 | hot-reload は稀、影響限定的 |
| H3 | HIGH | CycleGateAggregator の velocity hard-skip がデッドコード | velocity_skip_as_offset_enabled=True (soft mode) がデフォルトのため実害なし |
| H4 | HIGH | SellDynamicKillManager rolling window が非永続化 | fill_records warmup と同様の仕組みが必要 (要設計) |
| M1 | MEDIUM | warmup のレコード 2 回走査 | パフォーマンス (数千件で顕在化、実運用では数十件/日) |
| M5 | MEDIUM | Gate pre-check のキャッシュスプレッドが stale になりうる | advisory-only で実害軽微 |
| M7 | MEDIUM | Partial fill のハンドリング不在 | Coincheck 0.001 BTC では稀 |

---

## 5. テスト

新規 12 テスト:
- `TestConfigValidation209` (6件): limit負値、mult零値、timing零値、max_sleep デフォルト/負値
- `TestSleepClampLogic209` (3件): under/capped/disabled
- `TestVetoDeadlockFix209` (2件): 片方削除・両方クリア
- `TestInstantVelocityBoundary209` (1件): dt==max_dt 境界

v460 全体: 584 passed (5 件は既存の無関係な失敗)
