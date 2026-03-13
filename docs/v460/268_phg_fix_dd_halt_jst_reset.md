# 268# DD halt JST 日付リセット + 本番インシデント対応

| 項目 | 値 |
|---|---|
| Issue | 268# |
| 種別 | bugfix / incident |
| フェーズ | phg (横断品質改善) |
| Commit | `4b863d211` |
| テスト | 3688 passed, 32 skipped (+8 from 267#) |
| 発生日時 | 2026-03-03 10:51 JST ～ 2026-03-04 06:04 JST (約19h停止) |

---

## 背景 — 本番インシデント

2026-03-03 10:51 JST に DD hard halt がトリガーされ、bot が約 19 時間停止。
3 つの原因が収束して最大影響を発生させた。

## 根本原因分析

### 原因 1: UTC 日境界問題 (主因)

`DailyDrawdownGuard._utc_today()` が UTC 00:00 = **JST 09:00** で日付リセット。
JST 10:51 に halt → 次のリセットは翌 JST 09:00 → **最大約 22 時間** の halt。

```
JST 10:51  DD hard halt トリガー
JST 09:00  ← 次の日付リセット (UTC 00:00)
= 約 22h の halt ウィンドウ
```

### 原因 2: cooldown_release 未搭載版が稼働

本番は 246# の `cooldown_release` (2h 後の部分解除) 導入前のコードで稼働中。
cooldown_release があれば 2h 後に lot 30% で再開できたが、旧コードでは日替わりまで完全停止。

### 原因 3: 267# デプロイが UTC 21h に重複

267# commit 後のデプロイ作業が hard halt 継続中の時間帯に重なり、
実質的に halt 解除を待つだけの状態が続いた。

## 修正内容

### `_utc_today()` → `_today()` (設定可能 TZ)

```python
# Before (266#以前)
@staticmethod
def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")

# After (268#)
def _today(self) -> str:
    return datetime.now(self._day_reset_tz).strftime("%Y%m%d")
```

### `day_reset_utc_offset_hours` パラメータ追加

| 層 | 変更 |
|---|---|
| `DailyDrawdownGuard.__init__` | `day_reset_utc_offset_hours: float = 0.0` 引数追加 |
| `DailyDrawdownGuard.__init__` | `self._day_reset_tz = timezone(timedelta(hours=...))` |
| `FillTestConfig` | `dd_day_reset_utc_offset_hours: float = 9.0` (デフォルト JST) |
| `run_fill_test.py` | config → `DailyDrawdownGuard` コンストラクタに渡す |
| YAML パーサー | `loss_control.daily_drawdown.day_reset_utc_offset_hours` サポート |

### 影響範囲

| 箇所 | 影響 |
|---|---|
| `maybe_reset_day()` | `_utc_today()` → `_today()` |
| `import_state()` | `_utc_today()` → `_today()` |
| docstring | UTC固定の記述を「設定可能なタイムゾーン」に更新 |

### 最大 halt 時間の短縮

| 条件 | Before (UTC) | After (JST) |
|---|---|---|
| JST 10:51 に halt | 翌 JST 09:00 = **22h** | 翌 JST 00:00 = **13h** |
| JST 23:59 に halt | 翌 JST 09:00 = **9h** | JST 00:00 = **1min** |
| 平均的な halt | ~14h | ~8h |

## 変更ファイル一覧

| ファイル | 変更内容 |
|---|---|
| `scripts/v460/lib/daily_drawdown_guard.py` | `_utc_today`→`_today`, `_day_reset_tz` 追加 |
| `scripts/v460/lib/fill_config.py` | `dd_day_reset_utc_offset_hours` 追加 |
| `scripts/v460/run_fill_test.py` | config パラメータ渡し |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +8 テスト (TZ リセット検証) |
| `tests/unit/v460/test_215_dd_fix_alert_mode.py` | TZ パラメータ対応 |
| `CHANGELOG.md` | 268# セクション追加 |

## テスト追加 (+8)

TZ 日付リセットの境界条件テスト:
- JST 23:59 → 00:00 リセット検証
- UTC offset 0 / 9 での日付判定
- `import_state` の TZ 対応
- 既存テストの互換性維持

## 3/4 デプロイ後タイムライン

268# は 12:20 JST にデプロイ。以下のパターンが繰り返し発生:

| 時刻 (JST) | イベント |
|---|---|
| 06:04 | 267# で再起動 (sha=`227024d549aa`), state restored: pnl=-50bps, halted |
| 06:04 | cooldown_release: 69172s >= 7200s → 部分解除 |
| 06:10 | 2 回目の再起動 (同 sha) |
| 07:00 | sell per-side halt 継続、buy に切替 |
| 07:04 | **deadlock**: buy=JPY不足, sell=per-side halt → 223# 拒否 |
| 07:14 | Per-side halt released (15 cycles exhausted) |
| 07:16 | balance_forced → degraded liquidation sell 開始 |
| 07:36 | **PER-SIDE HALT**: sell -38.21bps → 再 halt |
| 08:00 | Per-side halt released (15 cycles) |
| **09:00** | **Day reset**: 20260303→20260304 (UTC=00:00, **旧コード最後**) |
| 09:49 | **PER-SIDE HALT**: sell -34.12bps |
| 10:09–11:15 | deadlock × 12 cycles (6min間隔) |
| 11:21 | Per-side halt released |
| 11:29 | **PER-SIDE HALT**: sell -32.15bps → 即再 halt |
| 11:56–12:14 | deadlock × 4 cycles |
| **12:20** | **268# デプロイ** (sha=`4b863d211de0`), state: pnl=-24.38bps |
| 12:21 | 2 回目起動 (同 sha), state restored |
| 12:20–13:21 | deadlock 継続 (sell per-side halt 未解除) |
| 13:27 | Per-side halt released |
| 13:45 | sell filled @ ¥10,666,367 (pnl=-5.46bps) |
| 13:47 | **PER-SIDE HALT**: sell -37.61bps → **4 回目の halt** |
| 14:00 | buy filled @ ¥10,690,939 (pnl=-1.62bps) ← 最後の fill |
| 14:05– | **deadlock 継続中**: buy=JPY¥1,028不足, sell=per-side halt |

## 発見された改善課題

### トレード視点の分析 — 「ガチホは意図的か？」

#### 3/3–3/4 の BTC/JPY 価格推移とポジション

```
¥10,886,374  ← 3/3 09:38 sell (この付近が3/3の高値)
   ↓  -1.04%
¥10,773,418  ← 3/4 07:34 sell (DD halt復帰後の最初の約定)
   ↑  +0.39%
¥10,815,118  ← 3/4 08:17 sell ★ 本日の高値 sell
   ↓  -0.67%
¥10,742,784  ← 3/4 10:01 buy  (最後の buy → 以降 sell halt)
   ■■■■■■■ 10:09–11:21 deadlock (72min) ■■■■■■■
¥10,745,311  ← 3/4 11:27 sell (halt解除直後)
   ↑  +0.17%
¥10,764,071  ← 3/4 11:48 buy  (per-side halt 即再突入)
   ■■■■■■■ 11:56–13:21 deadlock (85min) ■■■■■■■
   ↓  -0.91%  ★ この下落中にガチホを強制された
¥10,666,745  ← 3/4 13:45 sell (halt解除直後、下落途中で売り)
   ↑  +0.23%
¥10,690,906  ← 3/4 14:00 buy  (直後に再 halt)
   ■■■■■■■ 14:05–  deadlock 継続中 ■■■■■■■
¥10,707,427  ← 3/4 14:40 現在価格 (API取得)
```

**保有**: BTC 0.002 (@平均 ≈ ¥10,690,939) + JPY 1,028

#### 価格変動 vs bot の判断

| 区間 | 価格変動 | bot の行動 | 評価 |
|---|---|---|---|
| 3/3 09:38 → 3/3 10:51 | -¥21,500 (-0.20%) | 10:50 buy → **DD halt** | 🟡 halt 自体は正当だが、halt 前の sell PnL が悪い |
| 3/3 10:51 → 3/4 06:04 | — (19h 停止) | **完全停止** | 🔴 268# で修正済 |
| 3/4 07:22 → 07:46 | ¥10,770k trending_down | sell_dynamic_kill 発動 | ⚠️ 下降トレンド検出 → sell 禁止は防御的に正しいが… |
| 3/4 08:00 → 09:00 | ¥10,815k に回復 | sell_dynamic_kill **継続** | 🔴 **反転上昇したのに kill が解除されず sell 機会を逃した** |
| 3/4 09:00 → 09:39 | ¥10,783k → ¥10,714k (-0.6%) | ranging → trending_down | ✅ 09:39 に sell 約定 — 正しい判断 |
| 3/4 09:49 → 11:21 | sell halt (推定 ¥10,740k–¥10,770k) | **72min deadlock** | 🟡 レンジ内で方向感なし、実害は軽微 |
| 3/4 11:48 → 13:27 | **¥10,764k → ¥10,655k (-1.0%)** | **85min deadlock** | 🔴 **本日最大の下落を完全に見逃し** |
| 3/4 13:27 → 13:45 | ¥10,655k → ¥10,666k | halt 解除 → sell | ✅ 解除後すぐ売れたが、**底値付近** (既に ¥100k 下落済) |
| 3/4 14:00 → 14:40 | ¥10,691k → ¥10,707k (+0.15%) | **deadlock 継続** | 🟡 ガチホ結果的に微プラスだが偶然 |

#### 「下がる兆候を掴んだのに売ってない」問題の検証

**① 07:22 trending_down 検出 → sell_dynamic_kill が上書き**

```
07:22  Regime transition: ranging → trending_down
07:22  sell kill: regime=trending_down
  ...sell_dynamic_kill が 08:54 まで継続 (92min!)
08:17  ← この間にレンジ回復 (¥10,815k) → sell 機会を完全に逃す
```

regime_detector は trending_down を検出したが、`sell_dynamic_kill` が「売ると損する」と判断して売りを禁止した。
**問題**: sell_dynamic_kill は「過去の sell PnL の移動平均」ベースで判断するため、
過去の悪い sell 結果が残っている限り、新しいトレンドが来ても解除されない。
→ **過去の失敗に引きずられて、新しい機会を逃す** (recency bias の逆)

**② 11:48 → 13:27 の ¥10,764k → ¥10,655k (-1.0%) 下落を完全に見逃し**

```
11:29  PER-SIDE HALT: sell -32.15bps (1 fill で即再 halt)
11:56  deadlock 開始 (buy=JPY不足, sell=halt)
12:20  268# デプロイ (halt 状態は引き継ぎ)
13:21  balance_forced 最後の拒否
13:27  halt 解除 → sell_guard_reject (offset 調整中)
13:33  spread_too_narrow
13:39  skip_gate
13:45  SELL @ ¥10,666,367 ← ★ 解除から 18min も掛かった
```

**¥10,764k → ¥10,655k の -¥109k (-1.0%)** の下落中、bot は 85 分間完全に沈黙。
0.002 BTC 保有で **約 ¥218 の評価損** (全資金の ≈ 1% に相当)。

この間に売れていれば（0.001 BTC でも）: ¥10,760 × 0.001 = ¥10.76 の上乗せ分を確保できた。

**③ halt 解除後もすぐに売れない問題**

13:27 に halt が解除されたが、実際の sell 約定は 13:45 (18min 後)。
- 13:27: `sell_guard_reject` — sell guard がまだ慎重
- 13:33: `spread_too_narrow` — spread ¥0 が min(¥1,000) 未満
- 13:39: `skip_gate` — ML gate が拒否
- 13:45: ようやく sell 約定

halt 解除直後に「即座に市場に戻る」ための fast-path がない。

#### Regime Detector の判断精度

| 時刻 | Regime 判定 | 実際の価格動向 | 精度 |
|---|---|---|---|
| 07:22 | trending_down | ¥10,770k → ¥10,815k (反転上昇) | ❌ false positive |
| 07:46 | ranging | ¥10,815k → ¥10,780k | ✅ |
| 09:39 | trending_down | ¥10,717k (下落中) | ✅ 正確 |
| 09:52 | ranging | ¥10,746k (底から反転) | ✅ |
| 13:45 | trending_down | ¥10,666k (下落後) | ✅ だが検出が遅い (11:48 から ¥100k 下落後) |
| 14:05 | ranging | ¥10,690k | ✅ |

**結論**: trending_down の検出自体は概ね正確だが：
1. **11:48–13:45 の ¥100k 下落時に trending_down を検出できなかった** (per-side halt でサイクルが回らず、regime 更新すらされなかった)
2. 検出しても sell_dynamic_kill が上書きするケースがある

#### 総合評価

| 観点 | 結論 |
|---|---|
| 「ガチホ」は意図的か？ | **No** — 3 層の防御 (per-side halt, sell_dynamic_kill, balance_forced deadlock) が重なり、**売りたくても売れない**状態 |
| 上昇局面で保有は正解か？ | 結果的に 14:00 → 14:40 で +¥17k (+0.15%) だが、**11:48–13:45 の -¥109k を受けた後の話**。実質的には -¥92k のダメージ後の微回復 |
| 下落兆候を掴めていたか？ | 部分的に Yes (09:39, 13:45 の trending_down) だが、**最大の下落 (11:48–13:45) は halt 中でそもそも検出不能** |
| maker bot として適切か？ | 🔴 **No** — maker bot は「方向に賭けない」が原則。強制ガチホは方向リスクの直接被曝 |

### I1 🔴 CRITICAL — Balance-forced デッドロック

**症状**: buy=JPY 不足 + sell=per-side halt → 完全停止 (6min 間隔で空サイクル)

**コード**: `fill_loop_orchestrator.py` L1699:
```python
# 223# P0: balance_forced 後に per-side halt を再チェック
if self._daily_drawdown_guard.is_side_halted(next_side):
    logger.warning(
        "[223#] balance_forced → sell is per-side halted — "
        "refusing to bypass halt (safety > liveness)"
    )
```

**原因**: 223# は「safety > liveness」原則で per-side halt を尊重するが、
buy が JPY 不足で不可能な状態では sell が唯一の回復手段。
sell を拒否すると JPY を得る方法がなくなり **完全デッドロック**。

**提案 (269# 候補)**:
- balance_forced + per-side halt + 反対側不可 の 3 条件が揃った場合のみ
  「liquidation sell」として per-side halt をオーバーライド
- lot は recovery scale (0.35×) または min lot で制限
- 専用ログ/メトリクスで追跡

### I2 🟡 — Per-side halt → release → 即再 halt パターン

**症状**: halt 解除 → 1-2 fills → 累計 PnL が閾値以下 → 即再 halt (90min 無駄)

**タイムライン例**:
- 07:14 released → 07:36 re-halt (22min, 数 fill)
- 08:00 released → 09:49 re-halt (1h49min, day reset 込み)
- 11:21 released → 11:29 re-halt (**8min, 1 fill**) ← 最悪
- 13:27 released → 13:47 re-halt (20min, 2 fills)

**原因**: per-side halt は累計 PnL を追跡するが、release 時にリセットしない。
recovery 期間 (lot scale 0.35×) でも 1 回の負 PnL で -30bps 閾値を再突破。

**提案**: release 時に per-side PnL アキュムレータを部分リセット (例: 50%) するか、
release 後 N cycles は再 halt 猶予期間を設ける。

### I3 🟡 — デッドロック中のサイクルカウント

**症状**: balance_forced_halt_block の空サイクルが per-side halt の 15-cycle カウントに含まれる。
結果的に ~90min のタイマーとして機能するが、「市場状況を見て回復判断」ではなく
「固定時間待ち」になっている。

**提案**: 空サイクル (skip) を halt cycle カウントから除外するか、
デッドロック検出時は即座に limited release するか検討。

### I4 🟢 — JST リセット未検証

268# のデフォルト `day_reset_utc_offset_hours=9.0` は config に明示設定されておらず、
`FillTestConfig` のデフォルト値に依存。次回の JST 00:00 (= UTC 15:00) リセットで
正常動作を確認する必要がある。

### I5 🟡 — sell_dynamic_kill の過剰持続

**症状**: 07:22 に trending_down 検出 → sell kill 発動 → 08:54 まで 92 分間持続。
しかし 08:17 には ¥10,815k まで反転上昇しており、**sell 機会を完全に逃した**。

**原因**: sell_dynamic_kill は「過去 50 件の sell PnL 移動平均 < -1.0bps」で発動し、
新しい良好な sell が入らない限り解除されない。**halt 中は sell できないので PnL が更新されず、
kill が自然解除される条件が満たされない** (kill ←→ halt の相互ロック)。

**提案**: sell_dynamic_kill に時間ベースの上限 (例: 30min) を設けるか、
regime が ranging に戻ったら kill を解除する条件を追加。

### I6 🟡 — halt 解除後の再参入遅延

**症状**: 13:27 halt 解除 → 13:45 sell 約定まで **18 分**。
その間 sell_guard_reject → spread_too_narrow → skip_gate と 3 連鎖ブロック。

**原因**: halt 解除後も通常のゲート判定が全て適用される。
halt 中は市場を「見ていなかった」にも関わらず、通常時と同じ慎重さで復帰する。

**提案**: halt 解除直後の N cycles は skip_gate を緩和するか、
balance_forced_rescue モード同様の fast-path を用意。

---

## 横展開 — ブロッキング機構の相互作用分析

### 全ブロッキング層のインベントリ

本 bot は 6 層の独立したブロッキング機構を持つ。各層は **局所的には正しい安全判断** を下すが、
**複数層が同時に作用する場合の相互作用は設計時に考慮されていない**。

| 層 | 機構 | 判断基準 | 解除条件 | 場所 |
|---|---|---|---|---|
| L1 | **DailyDrawdownGuard** (集約 halt) | daily PnL ≤ hard_limit_bps | 日替わりリセット or cooldown_release (2h) | `daily_drawdown_guard.py` |
| L1' | **Per-side halt** | side PnL ≤ per_side_hard_limit_bps | 15 cycles (~90min) or 日替わり | 同上 |
| L2 | **sell_dynamic_kill** | rolling sell PnL mean < threshold | rolling mean ≥ threshold (新 fill 必要) | `sell_dynamic_kill.py` |
| L2' | **buy_dynamic_kill** | rolling buy PnL mean < threshold | 同上 (buy 側) | 同上 |
| L3 | **CycleGateAggregator** (9 gates) | regime / vol / velocity / spread | 各 gate 固有 | `cycle_gate_aggregator.py` |
| L4 | **SkipGateEvaluator** (ML gate) | ML model probability | 市場状況変化 | `skip_gate_evaluator.py` |
| L5 | **Balance preflight** | JPY / BTC 残高 ≥ min lot | 反対 side の約定で残高回復 | `fill_loop_orchestrator.py` |
| L6 | **Veto 系** (toxic / phantom) | 直前 fill の異常判定 | N cycles 自然減衰 | 同上 |

### 相互作用マトリクス

以下のマトリクスは、2 つの層が **同時に作用した場合** の結果を示す。

```
            L1   L1'  L2   L2'  L3   L4   L5   L6
L1  (halt)   -    -    -    -    -    -   [C]   -
L1' (p-halt) -    ■   [B]   -    -   [F]  [A]  [E]
L2  (s-kill) -   [B]   -   [C]   -    -    -    -
L2' (b-kill) -    -   [C]   -    -    -    -    -
L3  (gate)   -    -    -    -    -    -    -    -
L4  (ML)     -   [F]   -    -    -    -    -    -
L5  (bal.)  [C]  [A]   -    -    -    -    -    -
L6  (veto)   -   [E]   -    -    -    -    -    -

■ = 同一層の自己ループ (I2: halt→release→即再halt)
[A]–[F] = 後述のパターン ID
```

### 識別されたデッドロック/相互ロックパターン

#### Pattern A 🔴 OBSERVED — Balance-forced + Per-side halt デッドロック

```
L5 (buy=JPY不足) + L1' (sell=per-side halt) → 完全停止
```

- **発生**: 3/4 に 3 回以上発生、各 72–90min
- **メカニズム**: buy が JPY 不足 → `balance_forced_switch` で sell に切替 → 223# が per-side halt を尊重して拒否 → `continue` → 次サイクルも同一パス
- **コード**: `fill_loop_orchestrator.py` L1699–1732
- **根本原因**: 223# は「safety > liveness」原則で halt 貫通を禁止するが、**唯一の回復経路 (sell → JPY 獲得) を塞ぐため、liveness が永久に失われる**
- **本日の実害**: 3h+ の完全停止、¥10,764k→¥10,655k (-1.0%) の下落を完全に見逃し

#### Pattern B 🔴 OBSERVED — sell_dynamic_kill ←→ Per-side halt 相互ロック

```
L1' (sell halt) → sell 不可 → L2 に新 PnL 入力なし → kill 解除不能
L2 (sell kill) ← kill 持続 → sell halt 解除後も kill が即座に再ブロック
```

- **発生**: 3/4 07:22–08:54 (92min)
- **メカニズム**: sell_dynamic_kill は rolling sell PnL mean で判定するが、per-side halt 中は sell ができないため PnL が更新されず、kill の自然解除条件が満たされない。halt 解除後も kill が残存し、**kill → halt 解除 → kill ブロック → 新しい fill なし → kill 持続** のループに陥る
- **コード**: `sell_dynamic_kill.py` (rolling mean 判定) + `daily_drawdown_guard.py` (per-side halt)
- **根本原因**: L2 がフィードバック (新しい fill PnL) を必要とする解除条件だが、L1' が その フィードバックの供給を遮断する。**情報遮断による解除不能**

#### Pattern C 🟡 CODE — Dual-kill + Quiescence + Balance-forced 3 層相互作用

```
L2 (sell kill) + L2' (buy kill) + 249# quiescence → 完全静観
  + L5 (balance_forced) → 250# degraded liquidation で部分救済
  + L1 (集約 halt) → ??? (未テスト)
```

- **発生**: 未観測だが、長期連敗シナリオで発生しうる
- **メカニズム**: buy/sell 両方が killed → 249# quiescence が「静観」を選択。ここに balance_forced が重なると 250# が degraded liquidation を許可するが、**さらに集約 halt が重なった場合の挙動は未定義**
- **コード**: `cycle_gate_aggregator.py` L237–256
- **リスク**: 250# は集約 halt を考慮していない。集約 halt + dual kill + balance_forced の 3 条件が揃うと、250# が degraded liquidation を許可するが、orchestrator 側で集約 halt がブロックする可能性 → **検証が必要**

#### Pattern D 🟡 POTENTIAL — Mirror of A (buy halt + sell BTC 不足)

```
L5 (sell=BTC不足) + L1' (buy=per-side halt) → 完全停止
```

- **発生**: **未観測** (buy per-side halt はログ全期間で 1 度も発生していない)
- **メカニズム**: Pattern A の完全なミラー。buy per-side halt が発動し、かつ BTC 残高が sell 最小 lot 未満なら同一デッドロックに陥る
- **発生確率**: 低い。buy PnL は市場参入直後にのみ大きく負になる傾向があり、per-side halt 閾値を超えにくい。ただし **アーキテクチャ上は同一の脆弱性**

#### Pattern E 🟡 CODE — Toxic veto + Per-side halt カウンタ停滞

```
L6 (toxic veto on sell) + L1' (sell halt)
→ veto カウンタが通常の cycle 消費で減衰しない
→ 226# S2 修正で halt_block path でも減算するが、
   halt 中の「実効的な veto 期間」が拡張される
```

- **発生**: 226# で部分修正済みだが、完全な解消ではない
- **メカニズム**: toxic veto は N cycles で自然減衰する設計だが、per-side halt 中は orchestrator が `continue` する。226# S2 で halt_block パスでもカウンタ減算を追加したが、**halt 中のサイクル間隔は通常の 5 倍 (sleep ×5)** のため、実時間での veto 持続が 5 倍に拡張される
- **コード**: `fill_loop_orchestrator.py` L1710–1725

#### Pattern F 🟡 OBSERVED — Halt 解除後の多段ゲート再参入遅延

```
L1' 解除 → L3 (sell_guard_reject) → L3 (spread_too_narrow) → L4 (skip_gate)
→ 実約定まで 18min
```

- **発生**: 3/4 13:27–13:45 (18min)
- **メカニズム**: halt 解除後、orchestrator は通常フローに復帰する。しかし halt 中は市場を「見ていなかった」にも関わらず、通常時と同じ慎重さで全ゲートを適用する。sell_guard、spread、skip_gate が順次ブロックし、**halt 解除から実質的な市場参入まで 18 分の空白** が生じる
- **影響**: halt 中に大きな価格変動があった場合、復帰遅延の間にさらなる損失が発生。**ただし安全性との兼ね合いがあるため、盲目的な fast-path は逆効果** になりうる

#### 自己ループ (I2): Per-side halt → release → 即再 halt

```
L1' release → 1-2 fills → side PnL 累計がまだ閾値以下 → 即再 halt
```

- **発生**: 3/4 に 4 回発生。最短 8min (11:21→11:29, 1 fill のみ)
- **メカニズム**: per-side halt は固定 cycle (15) で解除されるが、解除時に side PnL アキュムレータをリセットしない。累計 PnL が依然 -30bps 以下の状態で release されるため、1 回の負 fill で即座に再 halt
- **コード**: `daily_drawdown_guard.py` L320–336 (`tick_side_halt`)。release 時は `side_halted_×=False` にするだけで PnL リセットなし

### 構造的問題の診断

#### 1. 各層の独立設計

6 層のブロッキング機構は **順次追加** され、各 Issue/PR で個別に設計・テストされた:

| 層 | 初出 Issue | 設計時に考慮した相互作用 |
|---|---|---|
| L1 (集約 halt) | 168# | ─ (最初の層) |
| L1' (per-side halt) | 205# §9.5 | L1 との整合性のみ |
| L2/L2' (dynamic kill) | 157#/171# | L5 (inv_bypass) |
| L3 (cycle gate) | 193# | L2+L2' の dual-kill (219#) |
| L4 (skip gate) | 124# | ─ (ML 独立判定) |
| L5 (balance preflight) | 091# | L1' (223# で追加) |
| L6 (toxic/phantom veto) | 205#/238# | L1' (226# S2 で追加) |

**相互作用を横断的に検討した設計レビューは一度も行われていない。**
各 Issue で発見された問題 (例: 223# のhalt貫通バグ → 223# で safety 優先に修正 → Pattern A のデッドロック発生) は **点の修正** であり、修正が新たな相互作用問題を生んでいる。

#### 2. Safety と Liveness のトレードオフの非対称性

全層が **"safety > liveness"** 原則で設計されている:

- 223#: 「halt 中の side を balance_forced で貫通するバグ → **安全のため拒否**」
- 249#: 「dual-kill bypass は危険 → **quiescence で静観**」
- sell_dynamic_kill: 「PnL が悪い → **新しいエビデンスが入るまで永久 kill**」

各層は独立に正しい判断だが、**複数層が同時に safety を選択すると、liveness がゼロ** になる。
結果として bot は「何もしない」ことが最も安全な行動となるが、**maker bot にとって「何もしない」は方向リスクへの直接被曝** であり、実際には安全ではない。

> **maker bot のリスク方程式**: `リスク = max(取引リスク, ポジション保有リスク)`
>
> 全層が取引リスクを最小化しても、ポジション保有リスクが無制限に増大すれば全体リスクは増加する。

#### 3. フィードバック遮断

Pattern B で顕著だが、**ブロッキングがブロッキングの解除条件を遮断する** 構造が存在する:

```
        ┌──── halt blocks sell ────┐
        ↓                          │
  sell_dynamic_kill           per-side halt
  (needs new sell PnL)        (needs 15 cycles)
        │                          ↑
        └── kill blocks sell ──────┘
```

L1' と L2 は独立に判定するが、**L1' が L2 の解除に必要なデータ供給を遮断** し、
L2 が L1' 解除後の唯一の回復経路をブロックする。

### アーキテクチャ改善の方向性

| レベル | 提案 | 難度 | 効果 |
|---|---|---|---|
| **点修正** | Pattern A: balance_forced + per-side halt + 反対側不可 → liquidation sell | 低 | 🔴 deadlock 即時解消 |
| **点修正** | Pattern B: sell_dynamic_kill に時間上限 (30min) | 低 | 🔴 kill↔halt ループ解消 |
| **点修正** | I2: per-side halt release 時に PnL 部分リセット | 低 | 🟡 即再halt 防止 |
| **点修正** | Pattern F: halt 後 fast re-entry mode | 中 | 🟡 復帰遅延短縮 |
| **構造改善** | **Cross-layer Liveness Monitor**: 全層を統合監視し、N 分以上の完全停止を検出したら最も弱い層を緩和する "circuit breaker for circuit breakers" | 中 | 🔴 全パターン包括的解消 |
| **構造改善** | **Holding Risk Evaluator**: ポジション保有リスクを定量化し、保有リスク > 取引リスクの場合は safety 原則を段階的に緩和 | 高 | 🔴 根本原因 (safety/liveness 非対称) の解消 |

---

## 269# 実装候補の優先順位

横展開分析を反映した更新版:

| 優先度 | Issue | タイトル | パターン | 期待効果 |
|---|---|---|---|---|
| **P0** | I1 | Balance-forced deadlock Liquidation sell | A | deadlock 完全排除。3/4 だけで 3h+ の機会損失 |
| **P1** | I5 | sell_dynamic_kill 時間上限 (30min) | B | kill↔halt 相互ロック解消。反転上昇時の sell 機会確保 |
| **P1** | I2 | Per-side halt release 時 PnL 部分リセット | 自己ループ | 90min halt→1fill→90min halt のループ排除 |
| **P2** | I6 | halt 解除後 fast re-entry mode | F | 18min の再参入遅延を数分に短縮 |
| **P2** | I7 | Cross-layer Liveness Monitor | 全体 | N分完全停止→最弱層を緩和する包括安全弁 |
| **P3** | I8 | Holding Risk Evaluator | 全体 | ポジション保有リスクの定量化。safety/liveness 非対称の根本解消 |
| **P2** | I3 | 空サイクル halt カウント除外 | deadlock 中の意味のない時間経過排除 |
| **P3** | I4 | JST リセット動作確認 | 今夜 00:00 JST で自動確認可能 |
