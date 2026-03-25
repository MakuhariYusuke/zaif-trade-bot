# 629# 626#/627#/628# レビュー評価・盲点深掘り・Sidecar キャッシュバグ修正

## 概要

626# (sell 構造損失分析), 627# (多角的検証レビュー), 628# (セカンドオピニオン) の 3 文書を横断評価し、盲点を補完。
627#/628# が指摘した **sidecar stale→error キャッシュバグ** を P0 として即時修正。

---

## 1. レビュー妥当性評価

### 1.1 627# の判定

| 指摘 | 妥当性 | 根拠 |
|------|:------:|------|
| sell 単独犯論は強すぎる | **正当** | broad window pnl30: buy=-0.19bps, sell=-0.06bps → buy も負けている |
| velocity 主犯論は強すぎる | **正当** | same-SHA worst sell に vel=-0.57/-1.12 あり → 上昇中だけではない |
| 時間軸混在 (pnl30 vs pnl120) | **正当** | SG は pnl30 予測。120s で「誤判定」と断ずるのは時間軸の取り違え |
| sidecar stale→error バグ | **正当** | コード確認で再現確定 (本文書で修正) |
| 残損本丸は sell×ranging×ev_offset | **正当** | same-SHA 24 sell 全件 decision_path=ev_offset |
| clamp 飽和 (99-100%) | **正当** | ceiling buy=0.35, sell=0.40 でほぼ全件飽和 → risk discrimination 喪失 |
| buy×primary_only も要注意 | **妥当だが優先度低** | same-SHA buy +0.40bps で改善中。broad では未安定 |

**627# 総評**: 626# の方向性を支持しつつ、過剰主張を的確に補正。特に「sidecar stale→error」バグの発見は626#では到達できなかった実装レベルの問題であり、高い価値がある。

### 1.2 628# の判定

| 指摘 | 妥当性 | 根拠 |
|------|:------:|------|
| Velocity threshold の Zスコア動的化 | **方向性は正当** | 固定 bps は σ 変動に追従しない。ただし実装コスト要考慮 |
| Regime threshold 引き下げ時の下流暴走リスク | **正当** | trending 発火増 → offset boost 1.8× の乱発リスク |
| 626# の「閾値スケール不適合」は数学的に正しい | **正当** | 120s σ≈5.6bps に対し 50bps 閾値は確かに異次元 |
| sidecar 修正のコード提案 | **方向正当、不完全** | キャッシュヒット側のロジックは既存コードで対応済み (注 ※1) |

※1: 628# の修正提案は `(mtime, signal)` をキャッシュ保存する方向で正しい。キャッシュヒット側は既に `if cached_signal is not None and ttl_sec > 0: if _is_stale(...)` のパスがあるため、signal 実体を保存するだけで2回目以降も正しく `stale` を返す。追加の改修は不要。

### 1.3 626# (自己レビュー) の補正

| 626# の主張 | 補正後の評価 |
|-------------|------------|
| 「Sell が損失の 95%」 | pnl120 では正しい。pnl30 では buy も負けている。**集計軸を明示すべき** |
| 「Velocity 閾値 2.5bps に」 | P0 ではなく P1 が適切。ladder で 6.0→4.0→3.0 が安全 |
| 「Skip Gate 偽陰性」 | SG は pnl30 予測。pnl30 では score +1.96 が適切な可能性 → 「判定軸」が異なる |
| 「Toxicity 段が None」 | toxicity_budget_enabled=true なのに None → 遅行指標問題 (後述 §3) |

---

## 2. 盲点の深掘り

### 2.1 Toxicity が None な真の理由 — 遅行指標の構造限界

626# は「VPIN 0.53–0.80 なのに toxicity が None」を疑問視した。調査結果:

- **toxicity_budget_enabled = true** (sell/buy 両方) → 機能は有効
- **toxicity は rolling PnL ベース** (`rolling_mean / threshold_bps`)、VPIN とは独立
- **VPIN は volatility_guard 段** で独立処理 (`velocity_threshold_bps: 12.0`)

すなわち toxicity は「過去の PnL が悪化した後」に発火する **遅行指標**。AS 損失が出る **前** の fill ではまだ rolling_mean が閾値に到達しておらず、Green (offset_mult=1.0, 記録上は None) のまま通過する。

**構造的問題**: toxicity は「これから AS が来る」ことを予見できない。VPIN は先行指標だが、volatility_guard の閾値 12.0bps が高すぎて発火しない (626# §3 と同根)。

### 2.2 Regime 窓の実時間 — 626# の暗黙の誤認

626# は「120 秒で 20–26bps 移動を ranging と判定」と述べたが、正確には:

- regime detector の窓 = **20 observations × 120s cycle = 2400s = 40 分**
- `trend_threshold_pct: 0.5%` = **40 分間**で 0.5% 変動が trending の条件

40 分で 0.5% は「日次スケールの流用」ではなく、**40 分スケールとして設定された閾値**。ただし:

- 40 分で 0.5% 変動 = 5000bps × 0.005 = 25bps → 実測の worst fill mid 移動 (20–26bps/120s) と同程度
- 120 秒内の 20bps マイクロトレンドが 40 分の回帰窓では吸収されてしまう
- **問題の本質は「窓が長すぎる」ことであり「閾値が高すぎる」こととは区別すべき**

| アプローチ | 操作対象 | 効果 | リスク |
|-----------|---------|------|-------|
| 閾値引き下げ (0.5%→0.15%) | `trend_threshold_pct` | 短い trending も拾える | false positive 増加 |
| 窓短縮 (20→5 obs) | `regime_window` | 120s×5=10分窓でマイクロトレンド検出 | レジーム頻繁切替 |
| 両方 | 両方 | 最大感度 | 過剰反応リスク |

**推奨**: まず閾値のみ引き下げ (0.5%→0.20%)。窓短縮は P2 として評価。

### 2.3 Clamp 飽和の意味 — 628# が指摘し切れなかった問題

627# は buy 99%, sell 100% が clamped と報告。これは:

- ceiling: buy=0.35, sell=0.40
- hard_skip: ceiling × 2.5 (buy=0.875, sell=1.0) 超で cycle 棄却
- 上流 9 段 pipeline の出力がほぼ全件で ceiling を超過 → **ceiling が低すぎるか、基本 offset が高すぎる**

構造的帰結:
1. velocity 段が 1.5× boost を出しても ceiling で切り捨て → **velocity boost が無意味化**
2. trending 段が 1.8× boost を出しても同様 → 防御 boost が価格に反映されない
3. **「段が発火しない」問題と「段が発火しても ceiling で潰される」問題は同時に存在する可能性**

626# は前者のみ分析していたが、627#/628# を踏まえると後者も疑うべき。ただし 3/25 データでは trending/velocity が全件 None なので、後者は現時点では理論的指摘に留まる。

### 2.4 Skip Gate 30s vs 120s — 628# が正しく切り分けた

626# の SG 偽陰性指摘:
- SG score +1.96 で pnl120=-25.3 → 「誤判定」と結論

しかし SG は pnl30 を予測するモデル:
- pnl30 では +3.3 → SG score +1.96 は**妥当な判定**
- 120s で悪化したのは SG の予測範囲外

| 時刻 | pnl30 | pnl120 | SG score | SG 評価 |
|------|:-----:|:------:|:--------:|:-------:|
| 13:54 | -0.4 | -25.3 | +1.96 | 30s では neutral |
| 14:03 | -0.3 | -21.2 | +2.96 | 30s では neutral |

**根本問題**: SG が 30s で判定しても、maker のリスクは 120s で顕在化する。これは SG の予測ホライズン (30s) が不十分である可能性を示す。

---

## 3. Sidecar stale→error キャッシュバグ修正 (P0)

### バグのメカニズム (627# §6.1 の確認と追認)

```
[1回目] ファイル読込 → signal パース成功
  → _is_stale(signal.timestamp, ttl) = True
  → _store_sidecar_cache(path, (mtime, None))  ← BUG: signal を捨てる
  → return (None, "stale")

[2回目] mtime 同一でキャッシュヒット
  → cached_signal is None → True
  → return (None, "error")  ← stale が error に化ける
```

### 修正内容

`scripts/v460/lib/sidecar_signal_io.py`:

```python
# 修正前 (stale 時に signal を捨てていた)
_store_sidecar_cache(abs_path, (mtime, None))

# 修正後 (signal 実体を保持 → 次回 _is_stale() で都度判定)
_store_sidecar_cache(abs_path, (mtime, signal))
```

キャッシュヒット時の既存ロジック:
```python
if cached_signal is not None and ttl_sec > 0:
    if _is_stale(cached_signal.timestamp, ttl_sec):
        return None, "stale"  # ← 2回目以降もここに来る
```

signal 実体が保持されるため、2 回目以降も `_is_stale()` パスを通り正しく `"stale"` を返す。

### 回帰テスト追加

`tests/unit/v460/test_sidecar_sac_integration.py::TestSignalStaleness::test_stale_signal_twice_stays_stale`:
- stale signal を 2 回読み出し、2 回目も `"stale"` であることを確認
- 78 passed (77 + 1)

### 影響範囲

| 影響 | 修正前 | 修正後 |
|------|--------|--------|
| fill_record の sidecar_signal_status | 初回 stale → 以後 error | 全件 stale |
| テレメトリ集計 | error 件数が過大計上 | stale/error が正しく分離 |
| sidecar revive 判定 | error 多発で「壊れている」と誤診 | 「古いが読める」と正しく認識 |
| attribution 分析 | error 汚染 | stale と error の原因切り分け可能 |

---

## 4. 3 文書の合意点と未合意点

### 合意点 (626#/627#/628# 三者一致)

1. **大損バグの止血は前進** (620#–625#)
2. **sell × ranging × ev_offset の AS テール** が current SHA の主残損
3. **velocity 閾値 6.0bps は実データ域に対して高すぎる**
4. **regime trending 閾値 0.5% は到達困難** — trending 防御が死んでいる
5. **sidecar は機能していない** — stale→error バグ + signal 未更新

### 未合意点 (要実装・検証)

| 論点 | 626# | 627# | 628# | 本文書の判定 |
|------|------|------|------|------------|
| velocity 閾値目標 | 2.5–3.0bps | 段階的 (6→4→3) | Zスコア動的化 | **段階的が安全、Zスコアは P2** |
| sell 単独犯か | 損失の 95% | buy も負けている | 売り寄りだが両面 | **same-SHA は sell 主犯、broad は両面** |
| SG 偽陰性か | 偽陰性 | 時間軸混在 | 同上 | **30s では正当判定**。120s リスクは SG の守備範囲外 |
| regime 閾値 | 0.15–0.20% | 明言なし | 0.15% 妥当 | **0.20% から開始** |

---

## 5. 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `scripts/v460/lib/sidecar_signal_io.py` | stale 時のキャッシュ保存: `(mtime, None)` → `(mtime, signal)` |
| `tests/unit/v460/test_sidecar_sac_integration.py` | `test_stale_signal_twice_stays_stale` 追加, import 追加 |

## 6. 残アクション (優先順)

| 優先度 | アクション | 関連文書 |
|:------:|-----------|---------|
| **P0** | ~~sidecar stale→error 修正~~ | 627#/628# → 本文書で完了 |
| **P1a** | velocity 閾値 6.0→4.0 (段階的引き下げ第一弾) | 626#/627# |
| **P1b** | regime trend_threshold_pct 0.5%→0.20% | 626#/628# |
| **P1c** | VG velocity_threshold_bps 12.0→6.0 | 626# |
| **P2** | clamp 飽和の ceiling 引き上げ検討 | 627# §5.3 |
| **P2** | Velocity Zスコア動的化 R&D | 628# §2.1 |
| **P3** | SG 予測ホライズン 30s→60s 検討 | 626#/本文書 §2.4 |
