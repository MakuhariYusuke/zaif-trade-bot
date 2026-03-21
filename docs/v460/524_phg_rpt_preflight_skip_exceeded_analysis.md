# 524# preflight_skip_exceeded 停止分析 + ログタイムライン

> 作成日: 2026-03-21  
> 対象: 2026-03-21 19:14 停止イベント (run_id=1774083309_6ada1e89)  
> SHA: fbc097c36  
> n_records: 14615 (quarantined: 1145, clean: 13470)

---

## §1 事象概要

fill_test が `preflight_skip_exceeded` を理由に 2026-03-21 19:14:37 JST で停止。
- 14587 cycles 完了, 4535 filled
- 最終 run (hot-swap後): n=37, filled=4, FR=0.108, PnL=+0.818bps

停止時の残高状態:
```
jpy: 255.28270903 (free)
btc: 0.000000 (free)  /  btc_reserved: 0.00224928 (locked in open order)
```

---

## §2 ログタイムライン

### Phase 1: 正常動作 → 最終 fill (17:00–17:15)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 17:00:16 | Cycle 14519 fill | buy filled, pnl=+1.32bps, dust_sweep lot restored |
| 17:01:17 | バランス確認 | `jpy: 25649` → lot 0.002251 BTC |
| 17:07:18 | Cycle 14523 fill | buy filled, pnl=-0.83bps |
| 17:11:21 | Cycle 14524 fill | sell filled (micro_timeout attempt 2/4), pnl=+3.53bps |
| 17:14:57 | Cycle 14525 fill | buy filled (micro_timeout attempt 3/4), pnl=+1.22bps |

### Phase 2: 残高枯渇 (17:16–17:19)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 17:16:29 | **残高激減** | `jpy: 252.34, btc: 0.00225092` ← 直前 25,649 JPY → 252 JPY |
| 17:18:30 | buy 不足検知 | `252 < min 11389` |
| 17:18:30 | sell dust_sweep | lot = 0.00225092 BTC, lot_scale保護発動 |
| 17:18:30 | **sell 開始** | 496# Recovery Skew で sell kill-gate をバイパス → sell 試行 |
| 17:19:05 | Cycle 14526 sell | **unfilled** (wait=10.9s) → lot restored |

**※ 17:16 の JPY 25,649→252 の急落**: Cycle 14525 buy fill 後、buy 注文約定で JPY を消費 → BTC を取得。正常な流れ。問題は「取得した BTC が sell 注文に locked された後、その sell が unfilled でキャンセルされずに残存した」こと。

### Phase 3: 第1次 preflight 膠着 (17:20–17:59)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 17:20:06 | 両側不足開始 | `jpy: 252, btc_reserved: 0.00225092` → BTC全額locked |
| 17:22–17:28 | 連続 preflight fail | 2分間隔で both-side insufficient |
| 17:28:10 | **preflight_pause #1/3** | 300s 待機 |
| 17:41:16 | **preflight_pause #2/3** | 300s 待機 |
| 17:54:22 | **preflight_pause #3/3** | 300s 待機 → pause quota 枯渇 |

### Phase 4: 40分間の sell 実行成功 (18:03–18:15)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 17:59:23 | Cycle 14580 (sell) | 何らかの理由で sell が可能に（btc 解放?） |
| 18:03:33 | Cycle 14581 fill | sell filled, pnl=-3.99bps, sidecar=error |
| 18:07:46 | Cycle 14582 fill | **buy** filled, pnl=+0.52bps |
| 18:09:23 | Cycle 14583 fill | sell filled, pnl=+6.32bps |
| 18:12:41 | Cycle 14584 fill | **buy** filled, pnl=+0.43bps |
| 18:14:52 | Cycle 14585 (sell) | sell 注文発行 → **最終サイクル** |

### Phase 5: 注文スタック → 最終停止 (18:15–19:14)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 18:14:53 | sell order 発行 | id=8770081973, qty=0.00224928, price=11,280,197 |
| 18:15:04 | cancel order 1 | id=8770081973 → 成功 |
| 18:15:10 | re-quote order 2 | id=8770082358, price=11,280,380 |
| 18:15:21 | cancel order 2 | id=8770082358 → 成功 |
| 18:15:26 | **re-quote order 3** | id=**8770082779**, price=11,279,544 |
| 18:15:26 | Cycle 14585 完了 | **filled=False**, wait=10.9s |
| — | **order 8770082779 未キャンセル** | cycle 完了判定されたが注文がCoincheck上に残存 |
| 18:15:26 | FFD reset | fast_fill_defense multiplier 2.14→1.00 |
| 18:20~ | 両側不足開始 | `jpy: 255, btc: 0, btc_reserved: 0.00224928` |
| 18:25:13 | preflight_pause #1/3 | 300s 待機 (pause_count リセット?) |
| 18:38:20 | preflight_pause #2/3 | 300s 待機 |
| 18:51:26 | preflight_pause #3/3 | 300s 待機 → quota 再枯渇 |
| 19:14:37 | **SAFE_STOP** | 連続 preflight スキップ 10 回 → kill switch |

### Phase 6: 再起動後 (19:15–)

| 時刻 | イベント | 詳細 |
|------|----------|------|
| 19:15:13 | **stale order cancel** | id=8770082779 (sell), price=11,279,544, qty=0.00224928 |
| 19:15:17 | state restore | n=14615, guard fire counts 復元 |
| 19:17:19 | buy 不足 | `jpy: 255` → freezing buy (522# 正常動作) |
| 19:19:33 | Cycle 14617 | sell unfilled → 正常サイクル再開 |
| 19:26:15 | Cycle 14619 fill | **sell filled**, pnl=+1.45bps ← BTC解放後の売り |

---

## §3 根本原因分析

### 3.1 直接原因: sell 注文残存 (id=8770082779)

Cycle 14585 の micro_timeout re-quote で発行された sell order が **キャンセルされずに Coincheck 上に残存**した。

推定メカニズム:
1. micro_timeout attempt 3/4 で sell order 8770082779 を発行 (18:15:26)
2. 同秒で cycle 完了判定 (filled=False, wait=10.9s)
3. cycle 完了処理が order cancel を行わず cycle を抜けた可能性
4. BTC 全量 (0.00224928) が reserved にロック → sell 不可
5. JPY (255 JPY) では buy 不可 (必要: ~11,380 JPY)
6. 両サイド膠着 → preflight cascade → 停止

### 3.2 構造的問題: preflight ループ中の open order 確認欠如

preflight 両側不足時、`_handle_preflight_failure()` は以下のカスケードのみ実行:
1. balance_shrink (3回連続で lot 縮小)
2. preflight_pause (5回連続で 300s 待機, 最大3回)
3. SAFE_STOP (10回連続)

**欠落している手順**: 
- `btc_reserved > 0` の検出 → open order のキャンセル → BTC 解放
- これにより sell が可能になり、膠着を自力で回復できた可能性

### 3.3 補足: pause_count のリセット問題

Phase 3 で pause 3/3 まで消費された後、Phase 4 で一時的に正常動作に復帰。
Phase 5 で再度膠着した際に pause が #1/3 から再カウントされている。
→ `_preflight_pause_count` が正常動作復帰時にリセットされた可能性。
これ自体は安全側（retry の機会が増える）だが、意図的な設計かは要確認。

---

## §4 提案: preflight 両側不足時の open order キャンセル

### 4.1 提案仕様

`_handle_preflight_failure()` の先頭に open order 確認・キャンセルを追加:

```
if btc_reserved > 0 (or any locked balance):
    1. get_open_orders() で未約定注文を取得
    2. 全注文をキャンセル
    3. 1サイクル待機して残高再取得
    4. 改善した場合 → 通常サイクルに復帰 (preflight_skip_count=0)
    5. 改善しない場合 → 既存カスケード (shrink → pause → stop)
```

### 4.2 リスク考慮

- **API レート制限**: get_open_orders + cancel_all で 2 API コール追加 (max_preflight_skip=10 なので最大20コール)
- **レースコンディション**: キャンセル中に約定する可能性 → cancel 失敗は無視して進行
- **既存 stale order cleanup との重複**: startup 時の stale cleanup はあるが、runtime 中の cleanup は未実装

### 4.3 実装方針案

- `orchestrator_balance.py` の `_handle_preflight_failure()` 冒頭に挿入
- 既存の `orchestrator_guards.py` の stale order cancel ロジックを参考に再利用
- config: `preflight_cancel_stale_enabled: bool = True` (default on)
- ログ: `[preflight_stale_cancel] found X open orders, cancelled Y`

---

## §5 ログ集計 (2026-03-21 全体)

### 5.1 all_run 集計

```
all_run: n=13470 filled=4446 FR=0.330 PnL=-0.256bps AS=0.279
```

### 5.2 レジーム別

| レジーム | n | filled | FR | PnL(bps) | AS |
|----------|---|--------|----|----------|-----|
| ranging | 10379 | 3359 | 0.324 | -0.285 | 0.270 |
| trending | 521 | 236 | 0.453 | -0.043 | 0.280 |
| trending_down | 757 | 283 | 0.374 | +0.329 | 0.297 |
| trending_up | 1058 | 297 | 0.281 | -0.531 | 0.303 |
| unknown | 755 | 271 | 0.359 | -0.390 | 0.351 |

### 5.3 停止前直近セッション (hot-swap 後)

```
current_run (1774083309_6ada1e89): n=37 filled=4 FR=0.108 PnL=+0.818bps AS=0.250
trailing_200: n=200 filled=41 FR=0.205 PnL=-0.360bps AS=0.195
```

### 5.4 直近 fill 品質 (Phase 4: 18:03–18:15)

| 時刻 | サイクル | side | pnl(bps) | sidecar | 備考 |
|------|----------|------|----------|---------|------|
| 18:03 | 14581 | sell | -3.99 | error | sidecar 異常 |
| 18:08 | 14582 | buy | +0.52 | error | |
| 18:11 | 14583 | sell | +6.32 | error | 良好 fill |
| 18:13 | 14584 | buy | +0.43 | error | |
| 18:15 | 14585 | sell | unfilled | error | → 注文スタック |

sidecar=error が全セッションで継続 → sidecar signal もしくは retrain_scheduler 障害の可能性。

---

## §6 JPY 精度 (繊の単位) に関する補足観察

ログで確認された JPY 残高: `255.28270903` (小数点以下8桁 = 繊の単位)

現在のコードの JPY 精度処理:

| 箇所 | 処理 | 精度 |
|------|------|------|
| Coincheck API 応答 | 文字列で返却 (`'255.28270903'`) | 8桁 |
| adapter.py 変換 | `float()` 変換 | 15-16桁 (float64) |
| balance_checker 比較 | `jpy_free < jpy_needed` (float) | 精度保持 |
| ログ表示 | `:.0f` フォーマット | 表示のみ整数化 |
| 注文 rate | `str(round(price))` → 整数 | **0桁** |
| coincheck.yaml | `price_precision: 0` | 整数 |

**結論**: JPY 残高の精度は float64 として保持されており計算に問題なし。注文レートの整数丸めは Coincheck BTC/JPY の注文板が1円刻みである前提に基づく。繊単位の精度が注文レートに適用可能かは取引所 API 仕様の確認が必要。

---

## §7 AI レビュー依頼事項

1. **§4 の open order キャンセル提案**: 安全性・実装方針の妥当性
2. **pause_count リセット挙動**: 意図的設計 vs バグか
3. **micro_timeout 完了時の order leak**: cycle 完了時の未キャンセル注文検出メカニズムの要否
4. **sidecar=error 継続**: retrain_scheduler の正常性確認要否
5. **JPY 繊単位**: 注文レートへの適用可否 (Coincheck API が小数レートを受け付けるか)
