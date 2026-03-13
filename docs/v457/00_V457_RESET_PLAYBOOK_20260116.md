# v457 Reset Playbook (2026-01-16)

> v456で「同じループを繰り返す」状態から抜けるための、v457の開始ドキュメント。
> 過去vXXXシリーズ（v455/v456）の教訓を明文化し、次の一手を固定する。

---

## 1. 問題の核心 (いま陥っているループ)
1. **機能追加 → 期待先行 → 検証不足 → 収益失敗 → さらに機能追加** の循環。
2. **報酬関数の過剰設計**でPnL信号が潰れ、「取引しない/偏った行動」が最適解化。
3. **バックテストの信頼性不足**により、良すぎる結果を真に受けて誤判断。

根拠: `docs/v456/59_V456_FINAL_RETROSPECTIVE.md` `docs/v456/58_BACKTEST_FINAL_SUMMARY_20260115.md`

---

## 2. v454/v455/v456からの教訓 (やるべきこと / やめるべきこと)

### 継続すべき成果
- 88次元観測設計と環境ファクトリーの統一 (`docs/v456/59_V456_FINAL_RETROSPECTIVE.md`)
- 型安全化・訓練インフラの安定化
- ロギング・チェックポイント・マルチスケール検証の仕組み (`docs/v456/55_MULTI_SCALE_VALIDATION_FINAL_20260115.md`)

### 明確にやめるべきこと
- **報酬へのペナルティ過多** (PnLより罰が支配)  
  → `docs/v456/59_V456_FINAL_RETROSPECTIVE.md` と `docs/v455/10_reward_function_adjustments.md`
- **バックテストの“良すぎる”結果を鵜呑みにする**  
  → 初期63.2%勝率報告が後に0.3%勝率に修正 (`docs/v456/59_V456_FINAL_RETROSPECTIVE.md` `docs/v456/58_BACKTEST_FINAL_SUMMARY_20260115.md`)
- **設定値をコードで上書きして検証不能にする**

### 反省点 (v455/v456共通)
- v455は安定化達成だが「Alpha不足」 (`docs/v455/15_v455_summary_and_handover.md`)
- v456は「報酬の過剰設計」で収益性失敗 (`docs/v456/59_V456_FINAL_RETROSPECTIVE.md`)

## 2.5 v453以前からの再発掘 (Lost Alpha)
> v456等の複雑化の過程で失われた、シンプルだが強力だったロジックを復刻する。

- **「素朴な特徴量」の再評価**: 
  - 複雑な正規化や変換を入れる前の、生に近いPrice Actionや単純なMA乖離などがAlphaを持っていた可能性。
  - v444/v451時代のバックテストログ (`backtest_v451_optimized.py` 等) から設定値を抽出。
- **報酬関数の「原点回帰」**:
  - 複雑なシャープ・レシオ報酬などではなく、初期の単純な「利確/損切り」ベースの成功体験を掘り起こす。

### v454の軽い振り返り (High Win Rate / Low Return)
- **高勝率=収益ではない**: 勝率97.5%でもReturnがほぼ0%に張り付く麻痺状態が発生。  
  → 取引件数と総リターンの整合性を常に検証する。  
  根拠: `docs/v454/02_hybrid_strategy_analysis.md`
- **TP/SLが機能しない問題**: モデルが早期クローズしTP/SLが発火しない。  
  → 退出ルールの強制や`pnl_mode="trade"`で行動を整合させる。  
  根拠: `docs/v454/02_hybrid_strategy_analysis.md`
- **Z-Score補助輪の有効性**: `entry_action_source="zscore"`で初めてプラス化。  
  → ルールベースを教師として使い、RLは後から自律化する。  
  根拠: `docs/v454/03_scaling_and_next_steps.md`
- **手数料ショック**: feeが入ると勝ち筋が崩壊し、教師戦略自体が損失化。  
  → “Fee-Safe”パラメータ探索が前提。  
  根拠: `docs/v454/04_retraining_plan.md`
- **レジーム分布の罠**: 分類閾値の不整合で「強トレンドが存在しない」誤判定。  
  → レジーム分布のサニティチェックを必須化。  
  根拠: `docs/v454/04_retraining_plan.md`

---

## 3. v457の基本方針 (最小差分・最大検証)

### 原則
1. **報酬関数はPnL中心で単純化** (PnL - Costs)  
2. **検証を先に固定** (データ・手順・期待値)  
3. **新機能は“必ず”効果検証してから追加**

### v457で“やらない”こと
- 新しい巨大特徴量群の追加
- ブラックボックス型の報酬シェーピング強化
- 検証前の本番想定議論

---

## 4. v457のスコープ (最小セット)

### 必須テーマ
- **Reward Reset**: 報酬関数を単純化して基準を作る  
  - `reward = (pnl - fee - slippage) / max_position`
- **Config一本化**: 訓練/環境/報酬が必ずconfigに従う  
  - 例: `config/v457/base/config.yaml` のみを真実にする
- **Backtest検証強化**: 取引件数・勝率・PnLを必ず相互検証

### 継承するもの
- v456の88次元観測空間（ただし、Phase 2で旧特徴量との入れ替えを検討）
- EnvironmentFactoryの構造
- 訓練スクリプトの安定化パッチ (ログ・チェックポイント)

### 復活させるもの (v453以前)
- シンプルなTrend Followingロジック
- 過剰なフィルタリング（Regimeなど）のない素直なEntry判断
- `backtest_v451_optimized.py` 等に見られる成功パラメータ

---

## 5. 実施計画 (v457立ち上げ)

### Phase 0: 事前固定 & 考古学 (1-2日)
- v456の訓練/評価/バックテストを**再現可能**に固定 (Baseline確保)
- **Legacy Review**: v453以前 (v444, v451等) のコード・Configを解析し、「当時何が機能していたか」をリストアップする。
- 過去の成功パターンの特徴量セットを v457 の候補として登録。

### Phase 1: Reward Reset (2-3日)
- **シンプルPnL報酬版を先に完成**
- 旧報酬とのA/B比較を行い、PnL支配を確認

### Phase 2: Alpha探索 & 統合 (3-5日)
- **Legacy Injection**: Phase 0で特定した過去の有効特徴量を投入し、PnLへの寄与を測定。
- MTF特徴量とSignal統合を“オフ/オン比較”
- 追加した特徴量がPnL改善に寄与するかを数値で示す

### Phase 3: 検証 (2-3日)
- バックテストは「必須指標セット」を固定  
  - 取引件数, 勝率, 平均PnL, 最大DD, 収益曲線

---

## 6. 成功基準 (v457での最低条件)

### 学習面
- 10K, 50Kで報酬の一貫性が維持されること
- rewardトレンドが「単純PnL」で改善傾向を示すこと

### 実運用面
- 勝率 > 50% かつ 平均PnL > 0
- 最大DD < 10%
- バックテスト結果が“取引件数”と整合すること

---

## 7. v457で必ずやる検証

1. **「良すぎる結果は疑う」テスト**  
   - 取引件数と損益が一致しない場合は即中断
2. **Reward Ablation**  
   - PnL only → +Cost → +軽微なshape  
3. **Config一致チェック**  
   - 訓練ログにconfig hashを残す
4. **Fee On/Offの感度確認**  
   - fee=0 と fee=0.1% の両方で方向性が一致するか
5. **レジーム分布の検証**  
   - regime step countsを必ずログ化し、偏りを検出

---

## 8. v457開始時の具体アクション

1. v457用config作成 (v456から最低限コピーして削る)
2. 報酬関数の“PnLベース”実装
3. Backtest検証のチェックリスト作成
4. 10K訓練 + 最小バックテストの再現

---

## 9. まとめ

v456は「技術基盤の完成」には成功したが、**報酬設計と検証の甘さで利益を失った**。  
v457は「過剰設計を捨て、PnL一本で検証する」フェーズに戻る。  
このドキュメントの主眼は **“二度と同じループに戻らない”** ことであり、  
v457は**小さく、測れる変更だけ**を積み上げる。

---

### 参照ドキュメント
- `docs/v456/59_V456_FINAL_RETROSPECTIVE.md`
- `docs/v456/58_BACKTEST_FINAL_SUMMARY_20260115.md`
- `docs/v456/55_MULTI_SCALE_VALIDATION_FINAL_20260115.md`
- `docs/v455/15_v455_summary_and_handover.md`
- `docs/v455/10_reward_function_adjustments.md`
- `docs/v454/02_hybrid_strategy_analysis.md`
- `docs/v454/03_scaling_and_next_steps.md`
- `docs/v454/04_retraining_plan.md`

---

## 10. ステータス更新 (2026-01-16)

### スクリプト基盤の標準化
ユーザ指示に基づき、独自実装スクリプトを廃止し、以下のv456/v455資産をベースとした標準構成へ移行済み。

- **`scripts/v457/train.py`**:
  - ベース: `scripts/v456/train_v456_production.py`
  - 環境構築: `ztb.trading.environment.factory_v456.EnvironmentFactory` を使用
  - 観測空間: v456準拠の88次元 (Base 30 + MTF 27 + Regime 13 + Etc)
  - 特徴量: `FeaturePipeline` による自動計算（型安全）
  - 報酬: `compute_hft_reward` (SIMPLIFIED VERSION: PnL - Costs) を使用してv457方針に準拠

