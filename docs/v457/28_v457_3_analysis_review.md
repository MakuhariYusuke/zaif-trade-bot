# 28. v457.3 Analysis Review: Plan Hardening (Multi-Angle Critique)

対象: `docs/v457/27_v457_3_analysis.md`

## 1. 総評
TTL 固定で収益が出た点は「環境が解ける」ことの証明になっており、方向性は良い。  
ただし、**結果の多くが “Bull 区間での Buy & Hold 効果” に依存している可能性が高く、汎化の裏付けが不足**している。  
計画を強靭化するには「TTL 固定だけでなく、評価手順・ログ・報酬/コスト設計」を同時に固める必要がある。

## 2. 良い点（継続すべき判断）
1) **TTL が売買頻度の主因だった仮説が裏付けられた**  
   - 676 trades まで減少し収益化。これは重要な発見。
2) **2D Action の問題を切り分けた**  
   - Wrapper で先に原因特定 → リファクタリングへ、という順序は妥当。
3) **“環境は解ける” という事実を得た**  
   - まずはこのベースラインを活用すべき。

## 3. リスク・反証ポイント（見落としがちな点）
1) **Buy & Hold の市場依存性**  
   - 収益の主因が「Bull 区間の持ちっぱなし」なら、  
     **レンジ/下落局面では逆効果**になる可能性が高い。  
   - Buy & Hold のベースラインと **アウト・オブ・サンプル評価**が必須。

2) **TTL 固定でも強制クローズは残る**  
   - `max_ttl_steps` 到達で強制クローズ → cooldown → 再入場が起きる。  
   - “Hold” の見た目でも **実際は周期的な再エントリー**になり得る。

3) **676 trades/10k steps はまだ高頻度**  
   - 15.8% → 6.8% に減ったが、  
     **実運用では依然としてコスト過多の領域**。

4) **Profit Factor 5.35 の信頼性**  
   - 10k steps は統計的に弱い。  
   - “fee/slippage 実測ログなし” だと **見かけの PF が過大評価**される。

5) **報酬コンポーネント統合の難易度**  
   - `UltraProfitReward` 等は HeavyEnv 系に設計されている。  
   - `FastIntradayEnvV456` に直接差し込むには **インタフェース差分の吸収が必要**。

6) **ActionExecutor の所在が異なる**  
   - 記載の “`ztb/components`” ではなく、  
     実体は `ztb/trading/environment/components/action_executor.py`。

## 4. 計画の強化提案（多角的）
### A. 評価設計（最優先）
- **Train/Test を完全に分離**し、  
  “Bull / Range / Bear” を含む 3 区間で評価。  
- **Buy & Hold ベースライン**と比較し、  
  「超過収益があるか」を必ず確認。  
- PF / Expectancy / Avg Win-Loss を固定出力化。

### B. 行動設計
- **1D Action をネイティブに実装**し、  
  TTL ロジックを完全に切り離す。  
- TTL を使う場合でも、**max_ttl_steps を大きくし強制クローズ頻度を下げる**。  
- `min_delta` / `cooldown_steps` を組み合わせて “実際の売買頻度” を抑制。

### C. 報酬設計
- `fast_intraday.py` を維持しつつ、  
  **`min_edge_mult` と `edge_penalty_rate` の強化**で  
  “値幅が足りない取引は不利” を学習に反映。  
- `reward_scale` / `reward_clip` を再調整し、  
  **負の報酬が潰れない範囲**へ。

### D. ログ/観測の強化
- **action[0]/action[1] の分布ログ**  
- TTL 強制クローズ回数、cooldown 発生回数  
- trades.json に fee/slippage 実測

### E. データと運用
- データ源の統一（Yahoo/bitFlyer/coincheck）と時刻ズレ検証  
- “低ボラ期間” の定義を ATR/price で固定し、  
  重み付け学習 or サブセット学習に反映

## 5. 改訂ロードマップ案（簡潔）
1) **Phase 0**: 1D Action をネイティブ実装 + TTL 切離し  
2) **Phase 1**: 評価基盤（OOO 分割・ベースライン比較・固定指標）  
3) **Phase 2**: 取引頻度最適化（min_edge / cooldown / min_delta）  
4) **Phase 3**: Reward 追加最適化（ペナルティ・スケール調整）

## 6. 結論
v457.3 の発見は有力だが、**市場レジーム依存の可能性が高い**。  
“TTL 固定”は有効な応急処置であり、  
今後は **評価設計と頻度制御を同時に堅牢化**することで、  
再現性の高い計画へ進化させられる。
