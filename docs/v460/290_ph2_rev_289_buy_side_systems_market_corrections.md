# 290# ph2 レビュー: 289 Buy 側分析のシステム工学・市場理論補正

> **目的**: 289# の妥当性を再点検し、`buy` 側不振の真因を「実装挙動」「観測設計」「市場理論」の3層で補正する  
> **日付**: 2026-03-06  
> **対象**: `docs/v460/289_ph2_analysis_buy_side_improvement.md`  
> **制約**: G1.2-full 計測中のため、即時変更は最小化し、まず判定ロジックの誤認を排除する

---

## 1. 総評 (Codex 観点)

289# は、`buy` 側の損失クラスタを夜間・ranging に切り分けた点は正しい。  
ただし、次の1点は解釈を修正しないと意思決定を誤る。

- **修正必須**: `ev_weighted` 利用率を `skip_gate_model_used` で判定している点
  - 現在は `ev_as_offset_enabled=true` なので、`ev_score` を使っても `model_used` は primary のまま残る実装が標準挙動。
  - したがって「`ev_weighted` が 9.6% しか使われていない」は、少なくとも現行モードでは成立しない。

---

## 2. 妥当だった論点 (維持して良い)

1. **Simpson's Paradox 指摘**
   - 旧/新モデル期混在で score 解釈が歪む問題は妥当。
2. **`ev_weighted_pnl` tautology 指摘**
   - `ev_weighted_pnl=0.4*pnl30+0.6*pnl120` の ex-post 値であり、予測力根拠に使えない。
3. **損失クラスタの所在**
   - 新モデル期で夜間寄与が過大、かつ ranging が主損失源という方向性は維持可能。

---

## 3. 修正が必要な論点

### 3.1 `ev_weighted` 利用率の測り方が不適切

現行コードでは次の流れ:

- `scripts/v460/lib/skip_gate_evaluator.py:1196` で `_try_ev_weighted_decision(...)` 実行
- `scripts/v460/lib/skip_gate_evaluator.py:1203-1207` で `ev_as_offset_enabled=true` の場合
  - `result.ev_score` のみセット
  - emergency skip のときだけ decision 上書き
- `scripts/v460/lib/fill_cycle_executor.py:922-956` で `sg.ev_score` に基づき価格オフセットを実適用
- ログにも `[193# ev_offset]` が多数出力されている

結論:

- **`skip_gate_model_used` は利用率プロキシとして不十分**  
- `ev_score` 非 null 率、`[193# ev_offset]` 発火率、offset 乗数分布で評価すべき

### 3.2 「パス利用率 9.6%」主張の再定義

289# の 9.6% は「`model_used` が `ev_weighted:*` の割合」であり、  
**offset モードでは「ev path が動いた割合」ではない**。  
この点は 290# で明確に補正対象とする。

### 3.3 `buy` 不振の交絡 (forced flow)

`balance_forced_switch` 系の取引は、裁量 alpha ではなく在庫修復フロー。  
このフローが混在すると `buy` alpha 評価が過小化される。

補正:

- 分析系を `alpha_trade` と `inventory_repair_trade` で分離評価
- `balance_forced_switch=true` を別母集団として KPI を算出

---

## 4. システム工学補正 (見落とし拾い上げ)

### 4.1 観測スキーマ不足

`FillRecord` には `ev_weighted_pnl` はあるが、ex-ante の `ev_score` が恒久保存されない。  
結果として「モデルが何を予測したか」と「実績」の対応が追えない。

推奨:

1. `ev_score_pretrade` (ex-ante) を FillRecord 保存
2. `ev_offset_mult_applied` 保存
3. `decision_path` (`primary_only` / `ev_offset` / `ev_emergency_skip`) 保存

### 4.2 評価設計不足 (時系列依存)

現状の単純 t 検定/層別比較は、連続取引の自己相関を十分に扱えていない。  
特に夜間連続局面では iid 仮定が崩れる。

推奨:

1. block bootstrap (時系列ブロック単位) で CI 再推定
2. run_id + git_sha + market_day 固定効果で差分検証
3. side 別に `attempted` と `forced` を分けた Gate 補助指標を追加

### 4.3 再現性リスク

履歴に `git_sha` 長さ混在 (短縮/完全) があり、比較軸を誤る余地がある。  
同一 SHA 判定は `startswith` ではなく正規化比較に統一したほうがよい。

---

## 5. 市場理論補正 (buy 不振の構造解釈)

### 5.1 低 VPIN 環境の損失は「静かな相場」ではなく「情報優位不在」

VPIN 低位帯での `buy` 悪化は、短期 alpha が希薄なまま maker で先回りされる典型。  
「静かだから安全」ではなく「優位がないのに板に置くと逆選別される」が実態。

### 5.2 Ranging×Night は「方向予測」より「在庫・価格提示問題」

この領域で効く順序は:

1. 方向予測強化より先に、提示価格の保守化/休止を最適化
2. 在庫歪み時は alpha 発注を止め、修復発注のみ許可
3. 回復後に通常 alpha フローへ復帰

### 5.3 「高値売り・安値買い」への距離

現状は、理想のタイミング最適化よりも「不利局面で参加し過ぎ」が主損失。  
まずは **参入しない最適化** (toxic veto / session veto / inventory segregation) が先行課題。

---

## 6. 優先アクション (G1.2 後)

| 優先 | アクション | 目的 | 期待効果 |
|---|---|---|---|
| P0 | `ev_score_pretrade` と `ev_offset_mult` を FillRecord 保存 | ev path 可観測化 | 誤判定の再発防止 |
| P0 | `buy` KPI を `alpha` vs `forced` で分離 | 交絡除去 | 真の改善対象特定 |
| P1 | night+ranging の価格提示を追加保守化 (YAML) | 毒性時間の参加抑制 | DD/AS の即時低減 |
| P1 | `ev path` 評価指標を `model_used` 依存から置換 | 利用率の正測定 | 改善効果測定が成立 |
| P2 | ranging 専用 buy policy (予測より execution 重視) | 構造損失対策 | 中期の mean 改善 |

---

## 7. 290# 結論

289# は分析方向としては有効だが、`ev_as_offset` 前提での解釈補正が必須。  
現状の `buy` 不振は「モデル性能だけの問題」ではなく、**在庫修復フロー混在 + 低情報帯参加 + 観測不足**の複合問題である。

次の判断基準は以下に一本化する:

1. `ev` が実際に何回作用したかを正しく測る
2. `alpha` と `repair` を分離して評価する
3. 夜間/ranging では「当てる」より「参加しない・保守化する」を優先する
