# 206# 205レビュー応答 P0 3施策実装 — Hard Skip / Toxic Veto / 片側DD

> **日付**: 2026-03-01  
> **前提**: 205# レビュー (§1-§8 Codex分析 / §9 Gemini セカンドオピニオン) への P0 応答  
> **コミット**: `3ee34cb0d`

---

## 1. 背景

205# は 200#〜204# の進捗を横断的にレビューした文書。

- **§1-§8 (Codex)**: 根本未達点・盲点・次の優先順位を分析
- **§9 (Gemini)**: 市場微細構造理論に基づく P0 アクションプランを要求

Gemini §9 の要求のうち、即座に実装可能な 3 施策を P0 として実装した。

| §9 要求 | 対応 | 優先度 |
|---|---|---|
| §9.4 時間帯 Hard Skip (Kyle proxy) | **本 PR で実装** | P0 |
| §9.2 Toxic Fill 同一サイド拒否 | **本 PR で実装** | P0 |
| §9.5 片側完全封鎖 (One-sided Hard Halt) | **本 PR で実装** | P0 |
| §9.1 Velocity 完全 SSOT 化 | 未着手 (P1) | P1 |
| §9.3 OFI/PIN Toxic Flow 検知 | 未着手 (P2・新データソース必要) | P2 |

### Gemini 主張の検証結果

206# 実装前に §9 の主張を既存コードと照合し、妥当性を確認した。

| 主張 | 検証結果 |
|---|---|
| §9.4 時間帯制限をプロキシとして即実装すべき | **妥当** — 204# 分析で UTC16 (AS64%) / UTC21 (PnL-125.8bps/日) が突出。既存 soft offset (+0.3〜0.5) では不十分 |
| §9.2 loss cooldown (202# A) では不十分 | **妥当** — interval 2x 延長は同サイクルのみ、同一サイドの連鎖損失は防げない |
| §9.5 offset rescue だけでは片側 DD に無力 | **妥当** — 202# B は offset 保護のみ。片側で -30bps 超の出血を止める仕組みがない |
| §9.1 mid_trend_bps を破棄して一本化 | **部分的に妥当** — 計算式は共通化済 (velocity_math.py) だが信号源は別系統のまま |
| §9.3 OFI/PIN 導入が急務 | **方向性は正しいが P2** — coincheck REST API では板の volume 変化を高頻度取得できないため、まず proxy 指標から |

---

## 2. 実装

### 2.1 時間帯 Hard Skip (§9.4 — Kyle proxy)

**根拠**: Kyle (1985) の流動性モデルに基づき、ノイズトレーダー層が薄い深夜帯は情報トレーダーの Price Impact が激増する。soft offset では抑制不十分な最悪時間帯はサイクル全停止が合理的。

**設計**:
- `hard_skip_utc_hours: list[int]` — YAML で指定した UTC 時間帯で取引完全停止
- 既存の `hour_offsets` (soft offset) との共存: hard skip 対象時間は soft 以前にサイクル中断
- `cancel_reason = "hard_skip_utc_hour"` で fill_records に記録

**設定値**:
```yaml
hard_skip_utc_hours: [16, 21]
# UTC 16 = JST 01:00 (AS 64%, -79.8bps/日)
# UTC 21 = JST 06:00 (PnL -125.8bps/日)
```

**コード変更** ([fill_loop_orchestrator.py](../../scripts/v460/lib/fill_loop_orchestrator.py)):
- `_step_single()` 冒頭で `datetime.now(timezone.utc).hour` を判定
- 対象時間帯なら即座に `cancel_reason = HARD_SKIP_UTC_HOUR` でサイクル中断
- `cycle_interval_sec` を待って次サイクルへ (interval 内に時間帯が変わればスキップ解除)

### 2.2 Toxic Fill 同一サイド拒否 (§9.2)

**根拠**: Easley-O'Hara (1992) PIN モデル — toxic flow は方向性を持つ。大損直後に同一サイドを再試行すると連鎖損失に陥る。202# A の interval 延長 (1 サイクル限定) では根本遮断できない。

**設計**:
- `toxic_fill_veto_threshold_bps: float` — この PnL 以下の約定で発動
- `toxic_fill_veto_cycles: int` — 同一サイドを封鎖するサイクル数 (0=無効)
- 約定後の PnL 判定で `_toxic_veto_side` / `_toxic_veto_remaining` を設定
- 次サイクル以降、該当サイドが選択されたら `cancel_reason = TOXIC_FILL_SIDE_VETO` でスキップ
- 反対サイドはそのまま実行可能 (片側のみの一時封鎖)

**設定値**:
```yaml
toxic_fill_veto_threshold_bps: -5.0  # PnL ≤ -5bps で同一サイド拒否
toxic_fill_veto_cycles: 3            # 3 サイクル封鎖
```

**コード変更** ([fill_loop_orchestrator.py](../../scripts/v460/lib/fill_loop_orchestrator.py)):
- `_step_single()` のサイド決定後、`_toxic_veto_side == side` かつ `_toxic_veto_remaining > 0` で skip
- `_on_fill_completed()` で PnL ≤ threshold 判定 → veto 発動 + ログ出力
- veto 消化ごとに `_toxic_veto_remaining -= 1`、0 到達で解除

### 2.3 片側 DD Halt (§9.5 — Ho-Stoll / Avellaneda-Stoikov)

**根拠**: Ho-Stoll (1981) の最適スプレッドモデル — 一方向で構造的に逆選択を食らい続ける状況では、offset 調整ではなく該当サイドの流動性提供を「完全停止」するのが最適解。

**設計**:
- `DailyDrawdownGuard` にサイド別 PnL 追跡を追加
  - `daily_pnl_bps_buy` / `daily_pnl_bps_sell`: サイド別累積 PnL
  - `side_halted_buy` / `side_halted_sell`: サイド別封鎖フラグ
  - `side_halt_remaining_buy` / `side_halt_remaining_sell`: サイクルベース自動解除カウンタ
- `update_pnl(bps, side=)` で累積し、閾値超過で `tick_side_halt()` → 片側封鎖
- `is_side_halted(side)` で `fill_loop_orchestrator` が照会
- 解除条件:
  - `per_side_halt_cycles > 0`: 指定サイクル数経過で自動解除
  - `per_side_halt_cycles == 0`: UTC 日替わり (`_check_day_rollover()`) まで永続封鎖

**設定値**:
```yaml
daily_drawdown:
  per_side_enabled: true
  per_side_hard_limit_bps: -30.0  # 片側累積 PnL ≤ -30bps で封鎖
  per_side_halt_cycles: 0         # 0 = UTC 日替わりまで永続封鎖
```

**コード変更**:
- [daily_drawdown_guard.py](../../scripts/v460/lib/daily_drawdown_guard.py): +112 行
  - `DailyDrawdownState` に 6 フィールド追加
  - `tick_side_halt()` / `is_side_halted()` / サイド別リセット
  - `export_state()` / `import_state()` でサイド別状態の永続化
- [fill_loop_orchestrator.py](../../scripts/v460/lib/fill_loop_orchestrator.py): への統合
  - DD halt 判定ブロックでサイド別封鎖チェック → `PER_SIDE_DD_HALT`
  - 封鎖サイドの場合、反対サイドへ自動切り替え → 反対も封鎖なら全停止

---

## 3. 関連変更

### 3.1 cancel_reasons.py
- 3 定数追加: `HARD_SKIP_UTC_HOUR`, `TOXIC_FILL_SIDE_VETO`, `PER_SIDE_DD_HALT`
- `AUDIT_CANCEL_REASONS` frozenset に追加
- `CancelReason` Literal 型に追加

### 3.2 fill_config.py
- 6 フィールド追加:
  - `hard_skip_utc_hours: list[int]`
  - `toxic_fill_veto_threshold_bps: float`
  - `toxic_fill_veto_cycles: int`
  - `per_side_dd_enabled: bool`
  - `per_side_dd_hard_limit_bps: float`
  - `per_side_dd_halt_cycles: int`
- YAML パーサ対応

### 3.3 hindsight_filter.py
- `H10` / `H11` / `H12` カテゴリ追加 (hindsight 分析での 206# 施策分類)

### 3.4 run_fill_test.py
- `fill_loop_orchestrator` への新パラメータ伝搬
- hard_skip / toxic_veto / per_side_dd の設定注入

---

## 4. テスト

### 新規テスト (14 tests)

| テストクラス | テスト数 | 内容 |
|---|---:|---|
| `TestPerSideDDHalt` | 8 | 片側 PnL 追跡、閾値封鎖、日替わりリセット、tick 解除、both-side halt、export/import |
| `TestCancelReasons205` | 3 | 3 新定数の存在・AUDIT 所属・CancelReason 型包含 |
| `TestFillTestConfig205` | 3 | YAML→config マッピング (hard_skip, toxic_veto, per_side_dd) |

### 既存テスト修正
- `test_total_halt_days_increments`: 日付一致テスト脆弱性修正 (2050年固定)

### テスト結果
```
181 passed, 0 failed
```

---

## 5. 205# 残課題ステータス

### Codex 分析 (§1-§8) のステータス

| § | 課題 | ステータス | 備考 |
|---|---|---|---|
| §3.1 | reprice 停止後もtoxic fill残存 | **206# で部分対応** | toxic_veto + per_side_dd で連鎖防止。pre-trade 選別は今後 |
| §3.2 | velocity SSOT 未達 | P1 | 計算式共通化は完了、信号源統一は未着手 |
| §3.3 | 202# 施策の runtime 未検証 | 要観測 | loss cooldown / rescue / VG sell supplement の発火ログ確認待ち |
| §3.4 | soft_triggered_today 不整合 | **203# で修正済** | `414b28568` |
| §3.5 | HALT startup の fill_records 欠落 | 軽微 | 分析影響は限定的 |
| §4.1 | toxic fast fill の pre-trade 化 | **206# で対応** | toxic_veto + per_side_dd |
| §4.2 | one-sided 縮退運転の拡張 | **206# で部分対応** | per_side_dd halt で片側封鎖。inventory unwind は今後 |
| §4.3 | 204# What-If の過学習リスク | 注意事項 | 直接ルール化せず shadow metric で検証すべき |
| §4.4 | postonly_crossing_skip 未発火 | 要観測 | 効果・副作用は発火後に評価 |
| §4.5 | index broken link | **修正済** | — |

### Gemini 要求 (§9) のステータス

| § | 要求 | ステータス |
|---|---|---|
| §9.1 | Velocity 完全 SSOT (mid_trend_bps 破棄) | P1 |
| §9.2 | Toxic Fill 同一サイド拒否 | **206# 実装完了** |
| §9.3 | OFI/PIN Toxic Flow 検知 | P2 (新データソース必要) |
| §9.4 | 時間帯 Hard Skip (Kyle proxy) | **206# 実装完了** |
| §9.5 | 片側 DD 完全封鎖 | **206# 実装完了** |

---

## 6. 変更ファイル一覧

```
10 files changed, 453 insertions(+), 48 deletions(-)
```

| ファイル | 変更 |
|---|---|
| `configs/v460/fill_test.yaml` | +11 (3 施策の YAML 設定追加) |
| `docs/v460/index.md` | +7/-1 (206# エントリ追加) |
| `scripts/v460/analysis/hindsight_filter.py` | +3 (H10/H11/H12 追加) |
| `scripts/v460/lib/cancel_reasons.py` | +12 (3 定数 + AUDIT + Literal) |
| `scripts/v460/lib/daily_drawdown_guard.py` | +112 (per-side DD 追跡・封鎖) |
| `scripts/v460/lib/fill_config.py` | +28 (6 フィールド + YAML パーサ) |
| `scripts/v460/lib/fill_loop_orchestrator.py` | +103 (3 施策の orchestrator 統合) |
| `scripts/v460/run_fill_test.py` | +64/-48 (パラメータ伝搬リファクタ) |
| `tests/unit/v460/test_145_structural_fixes.py` | +3 (cancel_reason テスト追加) |
| `tests/unit/v460/test_168_daily_drawdown_guard.py` | +158 (14 新規テスト) |
