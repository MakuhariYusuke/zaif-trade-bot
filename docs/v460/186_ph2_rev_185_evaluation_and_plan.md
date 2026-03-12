# 186# 185レビュー評価: 逆選択防御 + Trend 追随 — 妥当性検証と実装計画

> **種別**: rev  
> **フェーズ**: ph2  
> **日付**: 2026-02-28  
> **レビュー対象**: 185# (Codex レビュー + Gemini 3.1 Pro セカンドオピニオン)  
> **交差検証**: 178# 未達事項  
> **レビュアー**: Copilot (Claude Opus 4.6)

---

## 0. 結論

185# の両レビュアーの問題認識は**概ね正しい**。  
ただし各提案の実現性と優先順位については再整理が必要。

| 判断 | 内容 |
|------|------|
| ✅ 合意 | 「守りすぎて薄利・低参加」が現在の主ボトルネック |
| ✅ 合意 | Trend Mode ヒステリシス化は最優先 |
| ✅ 合意 | strictness clamp 未実装は設計欠陥 |
| ⚠️ 部分合意 | Macro Regime は方向として正しいが、179# の既存基盤を先に活用すべき |
| ⚠️ 部分合意 | Chase 順方向限定は正しいが、実装範囲が曖昧 |
| ❌ 不同意 | Gemini の「1 run=1 変更を即刻捨てよ」は品質劣化を招く |

---

## 1. 185# 所見の妥当性評価

### 所見 1 (CRITICAL): デプロイ前提のズレ — ✅ 事実確認済み

184# は「稼働コード=180#」と記載したが、実際は 182# (`3a1f9e380`) で稼働中。  
さらに 183# の YAML 変更は hot-reload で部分適用済み（velocity skip ±6bps が発動確認）、  
しかし 183# のコード差分（narrow_spread_guard）は未適用。

**→ 混在状態の把握は正しい。今後レビュー文書では `run_id + git_sha` を必ず明記する。**

### 所見 2 (CRITICAL): Trend Mode がほぼ発火しない — ✅ コード確認済み

**コード実態**:
- `gated_regime()` は **ハードゲート**: `confidence < 0.55` で即 `ranging` 降格
- ヒステリシス (enter/exit 分離) は**未実装**
- 直近 300 レコードで `trending_up` はわずか 6 件、うち confidence ≥ 0.55 は 3 件のみ

**妥当性判断**: CRITICAL は適切。trend_min_confidence がボトルネック。  
ただし「閾値を下げれば良い」のではなく、**ヒステリシス化が本質解**（185# §5.2 に同意）。

### 所見 3 (HIGH): 参加率不足へのシフト — ✅ データ支持

直近 run の fill rate 36.8% に対し、平均 PnL は +0.4938 bps。  
**「損しないが取れない」状態** = 183# 厳格化の意図通りだが限界。

**妥当性判断**: HIGH は適切。次フェーズは「参加率改善」に比重を移すべき。

### 所見 4 (HIGH): Macro Regime 不在 — ⚠️ 方向は正しいが実装コストに注意

**178# との関係**: 178# は `CycleStrategy` Protocol + `RegimePolicyConfig` を提案し、  
179# で**実装済み**。つまり「regime 別の制御量分岐」の骨格は既にある。

185# が求める「Macro Regime」は、現行の `RegimeDetector` (60s velocity ベース) とは別の  
5m/15m 時間軸のトレンド判定レイヤー追加を意味する。

**妥当性判断**: 中長期では正しいが、**まず既存 regime の発火率を改善するのが先**。  
Macro Regime 追加はデータパイプライン拡張 + 新モデル学習が必要で 1 セッションでは完了しない。

### 所見 5 (HIGH): Buy/Sell ホライズン非対称 — ✅ 確認済み・要修正

- buy: `skip_gate_lgbm_pnl30_buy.pkl` (短期)
- sell: `skip_gate_lgbm_pnl120_sell.pkl` (中期)

**妥当性判断**: 非対称性は意図的 (125# で sell の AS パターンが長期に表れるため)。  
しかし、185# の指摘通り、buy 側が短期ノイズに過敏でトレンド初動を取り逃す副作用がある。  
**ev_weighted (0.4*pnl30 + 0.6*pnl120) への統合が合理的**。  
ただし buy 用 pnl120 モデルが存在するか要確認。

### 所見 6 (MEDIUM): 説明可能性不足 — ✅ 事実

`regime_confidence=None` が 51.3% は問題。  
`gated_regime` (降格後のレジーム) と `guard_trace` (なぜスキップしたか) の記録追加は正当。

### 所見 7 (MEDIUM): 厳格化の相殺 — ✅ コード確認済み・最重要修正項目

`_total_offset = _hour_offset + _spread_offset` に**クランプなし**。  
さらに `regime_threshold` offset も加算されるため、最悪ケース:

```
hour_offset(+0.5) + spread_offset(+0.2) + regime_threshold(+0.2 for high_vol)
= +0.9 → PnL 予測が +0.9 bps 以上でないと通過不可 → 事実上全ブロック
```

**→ clamp 導入は即時必須。**

### 所見 8 (MEDIUM): Chase 順方向限定 — ✅ コード確認済み

現行 Chase は `is_drifting_away` (注文価格からの乖離) のみ。  
マクロトレンド方向との整合性チェックは**未実装**。

**妥当性判断**: 正しい。trending_up で sell 注文を chase するのは逆選択を増幅する。

---

## 2. Gemini セカンドオピニオン (§9) への評価

| # | 主張 | 判定 | 理由 |
|---|------|------|------|
| §9.1 | Macro Regime は**絶対条件** | ⚠️ 方向正しいが段階的に | 5m/15m 系データの取得・特徴量化・モデル学習が前提。既存 regime の改善が先 |
| §9.2 | 順方向 Chase 完全解放 + 逆方向 Cancel-only | ✅ 賛成 | 178# §2.3 で Chase = stale reprice 拡張として採用済み。方向制限は追加実装 |
| §9.2 | ヒステリシス化 | ✅ 賛成 | 185# Codex / 178# Copilot 全て同意の最優先事項 |
| §9.3 | RiskRuleEngine 即時投入 | ❌ 過剰 | ztb の RiskRuleEngine は v459 のバックテスト用。fill_test のリアルタイム skip chain とは設計が異なる。統合コストが高すぎる |
| §9.3 | Reconciliation でトレンド時の在庫 Skew 許容 | △ 一理ある | 178# §2.4 と同じ結論:「完全緩和は却下、regime 別緩和は検討」。`deadlock_limit_trending` (182#) で既に regime 別対応済み |
| §9.4 | 「1 run=1 変更」を即刻捨てよ | ❌ 不同意 | 複数変更の同時投入はデバッグ困難。ただし「相互依存する変更群」は同時投入が合理的 |
| §9.4 | 3施策同時投入 | △ 条件付き | ヒステリシス + clamp は相互依存するため同時投入可。model 変更は独立なので分離 |

**Gemini の最大の問題点**: 178# で指摘した「手数料 0% でもスプレッドコストが dominant」「IOC のリスク」を**完全に無視して再び同じ主張を繰り返している**。  
また「RiskRuleEngine 即時投入」は v459 のバックテスト基盤を fill_test に直輸入する提案だが、  
アーキテクチャが全く異なるため統合コストと破壊リスクが高い。

---

## 3. 178# 未達事項の棚卸し

178# で計画した Phase 1–5 のうち、現在の達成状況:

| Phase | 計画内容 | 状態 | 備考 |
|-------|---------|------|------|
| **P1-S1** | `_effective_sleep()` 抽出 | ✅ **179# で完了** | 6箇所以上で使用 |
| **P1-S2** | `RegimePolicyConfig` 分離 | ✅ **179# で完了** | `regime_policy.py` に配置 |
| **P1-S3** | `CycleStrategy` Protocol 定義 | ✅ **179# で完了** | `DefaultCycleStrategy` も実装 |
| **P2-C1** | Dynamic Cycle (C) | ✅ **179#/181# で完了** | `enabled: true` で稼働中 |
| **P2-D1** | Regime-linked Post-Fill Wait (D) | ✅ **179#/181# で完了** | regime×side 別 wait 設定済 |
| **P2-CD2** | 停止条件 3 つ | ✅ **181# で完了** | StopConditionMonitor |
| **P2-CD3** | hot-reload 対応 | ✅ **完了** | RegimePolicyConfig の hot-reload 対応 |
| **P3-CH1** | Chase ロジック (stale reprice 拡張) | ✅ **179# で完了** | `order_monitor.py` に統合 |
| **P3-CH2** | Chase 発動条件 (regime + drift) | ✅ **179# で完了** | ただし**方向制限なし** |
| **P4-T1** | Coincheck IOC サポート確認 | ❌ **未実施** | 優先度低として保留中 |
| **P4-T2/T3** | IOC 実装 | ❌ **未実施** | T1 に依存 |
| **P5-E1** | EV_weighted 計算 | ✅ **181#/182# で完了** | `ev_weighted_w30/w120` 設定済 |
| **P5-E2** | 重み係数 YAML 外部化 | ✅ **182# で完了** | RegimePolicyConfig に配置 |

### 178# の真の未達事項

| # | 項目 | 性質 | 185# での再指摘 |
|---|------|------|----------------|
| U1 | Trend Mode ヒステリシス | 設計提案のみ、未実装 | ✅ 185#-§5.2 で再度推奨 |
| U2 | Chase の方向制限 | Codex/178# で推奨、未実装 | ✅ 185#-§5.4 で再度推奨 |
| U3 | IOC 実現可能性調査 | 保留のまま | △ 185# では深追いせず |
| U4 | Buy model horizon 整合 | 認識のみ、対応なし | ✅ 185#-§5.3 で再度推奨 |
| U5 | Strictness clamp | 認識なし (183# で問題発生) | ✅ 185#-所見7 で新規指摘 |
| U6 | guard_trace 記録 | 認識のみ | ✅ 185#-所見6 で再度推奨 |

---

## 4. 統合実装計画

### 優先順位の根拠

現在のシステム状態:
- **正転気味** (直近 run +0.49 bps/fill) だが **fill rate 36.8%** で機会損失大
- Trend Mode がほぼ死亡 (confidence gate で 50% 降格)
- 厳格化の clamp がなく、更なる厳格化は危険

→ **「参加率改善」が最優先、「逆選択削減」は現状で十分**

### Phase A: 即時修正 (本セッション) — 推定 2h

相互依存する 2 施策を同時実装。

| # | 施策 | 根拠 | 変更ファイル |
|---|------|------|-------------|
| **A-1** | **Trend Mode ヒステリシス化** | 185#-§5.2, 178#-U1 | `regime_policy.py` |
| | enter: `confidence ≥ 0.45` | 0.55→0.45 に緩和 (trending 発火率向上) | |
| | exit: `confidence < 0.30` | ranging に戻るハードルを下げる (粘着性) | |
| | min_dwell: 3 cycles | 最低 3 サイクルは trend 維持 | |
| **A-2** | **Strictness clamp 導入** | 185#-所見7, 178#-U5 | `skip_gate_evaluator.py` |
| | `_total_offset = clamp(_total_offset, -0.3, 0.5)` | 過剰厳格化を防止 | |

### Phase B: 方向認識改善 (次セッション) — 推定 2h

| # | 施策 | 根拠 | 変更ファイル |
|---|------|------|-------------|
| **B-1** | **Chase 順方向限定** | 185#-§5.4, 178#-U2 | `order_monitor.py`, `fill_cycle_executor.py` |
| | trending_up: buy chase ✅ / sell chase ❌ (cancel-only) | | |
| | trending_down: sell chase ✅ / buy chase ❌ | | |
| **B-2** | **guard_trace 全レコード記録** | 185#-所見6, 178#-U6 | `fill_record_helpers.py`, `fill_cycle_executor.py` |
| | `gated_regime`, `effective_interval`, `skip_reasons[]` を FillRecord に追加 | | |

### Phase C: Model Horizon 整合 (次々セッション) — 推定 3h

| # | 施策 | 根拠 | 変更ファイル |
|---|------|------|-------------|
| **C-1** | **Buy SkipGate を ev_weighted 評価に変更** | 185#-§5.3, 178#-U4 | `skip_gate_evaluator.py`, YAML |
| | 既存 `ev_weighted_w30/w120` を SkipGate 評価に適用 | | |
| | 両側とも ev_weighted で評価し、horizon 非対称性を解消 | | |

### Phase D: Macro Regime 調査 (将来) — 推定 5h+

| # | 施策 | 根拠 | 変更ファイル |
|---|------|------|-------------|
| **D-1** | 5m/15m slope の取得・計算パイプライン構築 | 185#-§5.1 | 新規 module |
| **D-2** | Macro Regime 判定ロジック + micro regime との統合 | 185#-§5.1 | `regime_detector.py`, `regime_policy.py` |

### 保留・却下

| 施策 | 判定 | 理由 |
|------|------|------|
| IOC/Taker 実装 | 保留 | スプレッドコスト > 期待PnL の現状では不合理。API 確認すら不要 |
| RiskRuleEngine 投入 | ❌ | v459 バックテスト用。fill_test 統合コスト大 |
| 在庫 Skew 完全解放 | ❌ | regime 誤判定リスク。182# の `deadlock_limit_trending` で既に緩和済み |
| hour_offset 01h JST 緩和 | △ | 185# は +0.2〜0.3 推奨。A-2 の clamp (+0.5 上限) で自動的に制限される |
| 「1 run=1 変更」の撤廃 | ❌ | 相互依存する変更群は同時投入するが、独立変更まで混ぜるのは品質劣化 |

---

## 5. Phase A 実装の技術仕様

### A-1: Trend Mode ヒステリシス

`DefaultCycleStrategy` に以下を追加:

```python
@dataclass
class RegimePolicyConfig:
    # 既存フィールド
    trend_min_confidence: float = 0.55  # 既存 (enter に使用)
    # 新規フィールド
    trend_exit_confidence: float = 0.30  # ranging に戻る閾値
    trend_min_dwell: int = 3             # 最低保持サイクル数
```

`gated_regime()` の変更:

```python
def gated_regime(self, regime: str | None, confidence: float | None = None) -> str | None:
    c = confidence if confidence is not None else self._current_confidence
    if regime is None:
        return regime

    is_trending_input = regime.startswith("trending")

    if self._in_trend_mode:
        # Exit 条件: confidence 低下 AND min_dwell 経過
        if not is_trending_input or (
            c < self._policy.trend_exit_confidence
            and self._trend_dwell >= self._policy.trend_min_dwell
        ):
            self._in_trend_mode = False
            self._trend_dwell = 0
            return "ranging" if is_trending_input else regime
        self._trend_dwell += 1
        return regime  # trend 維持
    else:
        # Enter 条件: confidence ≥ enter threshold
        if is_trending_input and c >= self._policy.trend_min_confidence:
            self._in_trend_mode = True
            self._trend_dwell = 1
            return regime
        return "ranging" if is_trending_input else regime
```

### A-2: Strictness Clamp

`skip_gate_evaluator.py` L693 付近:

```python
_total_offset = _hour_offset + _spread_offset
# 185# clamp: 過剰な厳格化/緩和を防止
_OFFSET_FLOOR = -0.3  # 最大緩和
_OFFSET_CEIL = 0.5     # 最大厳格化
_total_offset = max(_OFFSET_FLOOR, min(_OFFSET_CEIL, _total_offset))
```

定数は YAML 外部化を検討するが、初回は安全なハードコードで投入。

---

## 6. 185# への Q&A 回答 (この評価に基づく)

| Q | 185# の問い | 回答 |
|---|------------|------|
| 185-A | trend_min_confidence ヒステリシス化 | ✅ **本計画 Phase A-1** で即時実装 |
| 185-B | gated_regime / guard_trace の全レコード記録 | ✅ **Phase B-2** で次セッション |
| 185-C | buy SkipGate ev_weighted 統合 | ✅ **Phase C-1**。ただしモデル再学習要否を先に確認 |
| 185-D | Chase 順方向限定 | ✅ **Phase B-1** で次セッション |
| 185-E | strictness clamp | ✅ **Phase A-2** で即時実装 |

---

## 7. リスク評価

| リスク | 影響 | 緩和策 |
|--------|------|--------|
| ヒステリシスにより「偽トレンド」に長く居座る | 低品質 fill の累積 | `trend_exit_confidence=0.30` + `min_dwell=3` で最低限の品質保証。hot-reload で即時調整可 |
| clamp が VG/regime offset にも影響 | 意図的な厳格化を制限してしまう | clamp は `_hour_offset + _spread_offset` のみに適用。regime_threshold は `evaluate()` 内部で別途加算されるため影響なし |
| 179#–183# の 5 セッション分 + A-1/A-2 の同時デプロイ | デバッグ困難 | A-1/A-2 は hot-reload 可能パラメータ中心。コード変更部分は限定的 (2ファイル) |
