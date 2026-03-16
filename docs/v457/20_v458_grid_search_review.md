# 20. v458 Grid Search Results Review (Next Strategy)

対象: `docs/v457/19_v458_grid_search_results.md`

## 1. 総評
結果は一貫して「**取引回数の増加 → 手数料負け**」を示しており、`min_delta`調整だけでは黒字化に届かない。  
Case Cのように**hold_rampを緩めた方が改善**する傾向が見えるため、次は「頻度抑制 + エッジ確保」を同時に掛けるフェーズに移るべき。

## 2. 重大な確認ポイント（前提の整合）
1) **手数料の定義が曖昧**  
   - 記載は「0.1% 往復」だが、実運用の `transaction_cost=0.001` は**片道0.1%**の可能性がある。  
   - 往復なら **0.2%** で、コスト推定がズレる。要確認。

2) **Net PnLが推定値か実測か**  
   - `Avg Cost` を使った推定なら、**実際のバックテストと乖離**する可能性がある。  
   - `trades.json` に手数料・スリッページを実数記録して算出する方が安全。

3) **Case Bの頻度逆転は「制約発火」の可能性**  
   - `min_delta=0.02` で頻度が下がったのは  
     `max_delta_per_step` / `cooldown_steps` / `TTL` の発火で  
     **実効トレードが削られている**可能性がある。  
   - ここは「行動分布」「実効エントリー数」「TTL終了数」を確認して原因分離すべき。

## 3. 追加の診断が必要な理由
現在の結果は「平均利益 < 平均コスト」で、**構造的に負ける設定**。  
`Avg Win` と `Avg Cost` の差を見る限り、**勝率が上がっても黒字化は難しい**。

次は「**1トレードの期待値を改善する**」か「**取引頻度を抑制してコスト合計を減らす**」の二択。

## 4. 次に取るべき方策（優先順）

### A. 取引頻度の抑制（優先度: 高）
- **cooldown_steps を 30〜60 に引き上げる**  
  目標: 100〜150 trades / 10k steps 程度に減らす。  
- `min_delta` は 0.03 を維持し、頻度は cooldown で抑える方が安全。

### B. エッジ判定の強化（優先度: 中）
- `min_edge_mult` を 1.5 以上で固定し、  
  **期待値の薄いトレードを機械的に排除**。  
- `vol_floor` を 0.001〜0.002 で再検証。

### C. 評価指標の追加（必須）
- **Profit Factor / Expectancy / Avg Win vs Avg Loss** を必ず出す。  
- `edge_shortfall / trade_cost / vol_ratio` をログに出し、  
  「低ボラでの無駄打ち」なのか「閾値ノイズ」なのかを分離する。

### D. Dynamic Thresholdは後回し
- ATRベースの動的 `min_delta` は効果検証する価値はあるが、  
  その前に **cooldown+edge判定** で損失構造を潰すのが先。

## 5. 次の実験提案（Case D 以降）

**Case D（提案修正版）**  
- Base: Case C  
- `cooldown_steps`: 60  
- `min_edge_mult`: 1.5  
- 目標: trades 100〜150 / 10k steps, Profit Factor > 1.05

**Case E（低ボラ回避）**  
- Base: Case C  
- `vol_floor`: 0.002  
- 目標: 低ボラ局面の無駄打ち削減

## 6. 結論
「回転数の回復」は達成できたが、**現状は“取引するほど損をする状態”**。  
次フェーズは**回転数を落とし、1回あたりの期待値を上げる**方向へ移行すべき。

最短ルートは:
1) cooldown強化  
2) min_edge_mult + vol_floor の再検証  
3) Profit Factor / Expectancy で評価固定化
