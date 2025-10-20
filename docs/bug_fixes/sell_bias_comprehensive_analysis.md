# SELL Action Bias - 徹底調査と修正計画の完全版

## エグゼクティブサマリー

### 現状

PPOモデルが極端な報酬設定(`has_position_sell_reward: 20.0`, `no_position_buy_reward: -1.0`)で訓練されたにもかかわらず、推論時にSELLアクションを選択せず、HOLDまたはBUYのみを選択する問題が発生しています。

### これまでの実施内容

1. **MaskablePPO統一**: モデル読み込みエラー解消、`use_sde`無効化
2. **ハイパーパラメータ調整**: エントロピー係数、報酬スケール、学習率、`n_steps`等を段階的に調整
3. **ポジション依存報酬**: 多数の組み合わせを試行(BUY/SELL/HOLDの奨励・ペナルティ変更)
4. **カリキュラム学習**: 過学習抑制のための段階的学習
5. **紙上取引テスト**: アクション分布と収益性の確認(複数回実施)
6. **アクションマスク修正**: `paper_trade.py`で`action_masks`を`predict()`に渡すように修正
7. **デバッグログ追加**: verbose出力試行(未完了)

### 今回の追加実施内容(2025-10-06)

1. **診断ツール作成** ✓
   - `ztb/utils/diagnostics/action_diagnostics.py`
   - マスク適用前後のlogits/probs、temperature、エントロピー、KL、lossを可視化
   
2. **強制アクションテスト作成** ✓
   - `tests/unit/environment/test_forced_actions.py`
   - 既知価格列でBUY→SELL実行時のPnL/fee/在庫を理論値と突合
   - 1テスト通過確認(`test_hold_only_sequence`)

3. **包括的ドキュメント作成** ✓
   - `docs/fix_sell_bias.md`: 問題の経緯、技術分析、修正計画
   - `docs/action_bias_implementation_guide.md`: 具体的なファイル・関数・コード例

## 依然として収益化できない原因の洗い出し

### Copilot分析に**欠けている視点**(追加)

#### 1. **学習時と評価時のAction Mask不一致** 🔴 CRITICAL
**状況**: 
- 学習時: 全アクション許可(mask未適用)
- 評価時: mask適用でillegalアクション除外
- 結果: 学習分布と評価分布が乖離 → deterministic選択が崩壊

**影響**:
- 学習中にillegalアクションも学習 → 評価でそれが除外 → 確率分布が大幅変化
- 特にBUYが学習で多く選ばれていた場合、評価でも残存

**修正策**:
- 学習時も評価と**同一のaction mask**を適用
- policy forward内でillegalを`-inf`に設定
- sampling/loss計算はmask後分布のみ使用

#### 2. **決定論デコード順序バグ** 🟡 HIGH
**正しい順序**: `mask → softmax(temperature) → argmax`

**誤った順序の影響**:
- `softmax → mask → argmax`: illegalアクションに確率配分 → 除外で分布歪み
- `argmax → mask`: maskより先に選択 → illegal選択の可能性
- temperature誤適用: 過信頼/過探索

**修正策**:
- 順序を明示的に固定
- `T=0.7`のsoft-greedy評価も併用
- 中間値をログして検証

#### 3. **policy headバイアス初期値の傾き** 🟡 MEDIUM
**メカニズム**:
- 最終層`bias`が微妙に正に偏る → BUY優先
- 勾配更新が不十分だと偏りが残存

**修正策**:
- `action_net.bias.zero_()` で再初期化
- または学習可能な`LogitBiasLayer`挿入(初期値=0)

#### 4. **アクション不均衡による勾配スタベーション** 🟡 HIGH
**メカニズム**:
- SELLが稀 → SELLのadvantage勾配が小
- policy更新がBUY/HOLD優先 → フィードバックループ

**修正策**:
- **逆頻度重み**: `w_a = max(0.5, 1/freq(a))`でpolicy lossを重み付け
- **レジーム分層サンプリング**: trend/range/high_vol/low_volを均等にサンプル

#### 5. **価値関数の過支配(vf_coef高すぎ)** 🟡 MEDIUM
**現状**: `vf_coef: 0.5`

**影響**: `value_loss >> policy_loss` → 方策が硬直

**修正策**:
- `vf_coef: 0.5 → 0.3`
- `gae_lambda: 0.95 → 0.9`
- `clip_range: 0.2 → 0.1`
- `target_kl: 0.02`導入

#### 6. **正規化統計のズレ(学習≠評価)** 🟢 LOW
**影響**: 学習時のscalerを評価で再現しない → 特徴量スケールがズレる

**修正策**:
- 学習時にscaler保存(`joblib.dump`)
- 評価時に固定ロード
- 0分散列を前処理で除去

#### 7. **手数料・スリッページの非対称** 🟢 LOW
**影響**: BUY/SELLで控除タイミングがズレる → SELL常に不利

**検証**: 強制アクション単体テストで1往復(BUY→SELL)の理論値突合

#### 8. **SELLの意味論(スポット制約)** 🟢 INFO
**前提**: スポット取引で空売り不可 → SELLはエグジット専用

**重要**: 
- KPIは「保有時のSELL実行率」であり、絶対SELL率ではない
- action maskで`position > 0`時のみSELL合法化が必須

### Copilot既存分析(再掲)

#### 環境・報酬ロジック
- ✓ 売買成立条件やアクションマスク検証済み(SELL合法確認)
- ✓ 報酬スケール確認済み(20.0 → 2.0)
- ⚠️ 手数料設定の影響(要検証)

#### データ・特徴量
- ⚠️ 学習データの片側バイアス(上昇相場中心の可能性)
- ⚠️ 特徴量がBUYシグナル偏重
- ✓ 価格データ整合性確認済み

#### ハイパーパラメータ
- ✓ エントロピー係数調整済み
- ⚠️ `gamma`/`gae_lambda`の短期行動強化の可能性
- ⚠️ バッチサイズ/更新頻度の不整合

#### モデル構造
- ⚠️ ネットワーク容量不足の可能性
- ⚠️ 過学習による評価時バランス崩壊
- ⚠️ 価値関数と政策の非同期

#### 評価・実装差異
- ✓ トレーニング/評価環境の設定確認済み
- ⚠️ モデル保存・読み込み時の正規化統計(要対応)
- ✓ 評価指標整備済み

#### マーケット特有要因
- ⚠️ 学習期間と評価期間のregimeずれ
- ⚠️ 高頻度急変動への追随不足
- ⚠️ 単一ペア依存のボラティリティ偏り

## 修正計画の優先順位

### Phase 1: 即効性高(1-2日) 🔴
1. **学習時action mask厳密適用** (CRITICAL)
   - `StrictMaskedPolicy`作成
   - policy forward内でillegal → `-inf`
   - 実装ガイド: `docs/action_bias_implementation_guide.md` §2

2. **決定論デコード順序修正** (HIGH)
   - `mask → softmax(T) → argmax`順序固定
   - `T=0.7`評価追加
   - 実装ガイド: §3

3. **強制アクションテスト完全実行** (HIGH)
   - 全テストケース実行
   - 手数料対称性確認
   - `pytest tests/unit/environment/test_forced_actions.py -v`

### Phase 2: 学習安定化(3-4日) 🟡
4. **アクション不均衡補正** (HIGH)
   - `ActionFrequencyWeighter`実装
   - `RegimeSampler`実装
   - 実装ガイド: §4

5. **policy head中立化** (MEDIUM)
   - `action_net.bias.zero_()`
   - `target_kl=0.02`導入
   - エントロピー係数cosine減衰(`0.6→0.2`)
   - 実装ガイド: §5, §6

6. **ハイパーパラメータ最適化** (MEDIUM)
   - `vf_coef: 0.5→0.3`
   - `clip_range: 0.2→0.1`
   - `gae_lambda: 0.95→0.9`

### Phase 3: 評価・検証(5-7日) 🟢
7. **正規化統計固定** (LOW)
   - scaler保存/ロード
   - 0分散列フィルタ
   - 実装ガイド: §7

8. **50k×3seed検証** (REQUIRED)
   - 変更前後で比較
   - BUY/SELL/HOLD分布
   - Sharpe、entropy、KL
   - レジーム別性能
   - 結果を`docs/fix_sell_bias.md`に記録

## 実装成果物

### 作成済みファイル ✓

1. **診断ツール**
   - `ztb/utils/diagnostics/action_diagnostics.py`: 504行
   - `ztb/utils/diagnostics/__init__.py`: 3行

2. **テストスイート**
   - `tests/unit/environment/test_forced_actions.py`: 268行
   - 10テストケース実装(1テスト通過確認済み)

3. **ドキュメント**
   - `docs/fix_sell_bias.md`: 包括的問題分析・修正計画
   - `docs/action_bias_implementation_guide.md`: 具体的実装手順

### 未実装(次ステップ)

1. **`ztb/training/policies/masked_policy.py`**
   - `StrictMaskedPolicy`クラス
   - 学習時mask厳密適用

2. **`ztb/training/utils/action_weighting.py`**
   - `ActionFrequencyWeighter`クラス
   - 逆頻度loss重み付け

3. **`ztb/training/data/regime_sampler.py`**
   - `RegimeSampler`クラス
   - レジーム分層サンプリング

4. **`ztb/training/unified_trainer.py`への統合**
   - 診断ログ統合
   - policy bias再初期化
   - entropy cosine減衰
   - hyperparameter調整

5. **`ztb/training/paper_trade.py`への統合**
   - `_get_action_with_temperature()`メソッド
   - scaler読み込み

## 検証基準

### 必須条件 ✅
1. **合法アクション率** ≥ 99.9% (学習/評価とも)
2. **アクション分布**: 極端な偏りなし
   - スポット前提: 「保有時SELL実行率」で評価
3. **エントロピー**: 早期枯渇なし(初期高→自然減衰)
4. **手数料対称テスト**: 全テストグリーン ✓(1/10)
5. **Sharpe > 0**: 50k×3seed平均

### 推奨条件 🎯
1. レジーム別分析(bull/bear/high-vol/low-vol)
2. 取引頻度が許容範囲内
3. ドローダウンが限度内
4. seed間の収束安定性

## 次のステップ(他AIエージェントとの相談用)

### 即座に確認すべき点

1. **現在のトレーニングコードでaction maskが適用されているか**
   - `MaskablePPO`のforward pass確認
   - loss計算でmaskが考慮されているか

2. **決定論推論の実装順序**
   - `paper_trade.py`の`predict()`呼び出しフロー
   - mask/softmax/argmaxの順序

3. **policy head初期化**
   - 最終層biasの初期値確認
   - 学習前のlogits分布

### 追加調査が必要な点

1. **学習データのregime分析**
   - 上昇/下降/レンジの比率
   - SELLが有利な状況の頻度

2. **特徴量のSELLシグナル表現力**
   - SELL優位を示す特徴の有無
   - 特徴量重要度分析

3. **モデル容量の妥当性**
   - ネットワークサイズと問題複雑度のバランス
   - 過学習の兆候

### 実装の優先順位相談

1. **最優先**: 学習時mask適用(Phase 1-1)
   - 効果が最も高い可能性
   - 実装比較的シンプル

2. **次点**: デコード順序修正(Phase 1-2)
   - バグの可能性が高い
   - 即効性あり

3. **並行**: 強制テスト完全実行(Phase 1-3)
   - 環境の正しさを保証
   - 他修正の前提条件

## 総合診断チェックリスト

### 環境検証 ✓
- [x] reward計算ロジック確認(SELL=20.0適用確認)
- [x] action mask実装確認(position>=0でSELL合法)
- [x] 基本テスト作成(test_hold_only_sequence通過)
- [ ] 全強制テスト実行
- [ ] 手数料対称性検証

### 学習パイプライン検証 ⏳
- [ ] 学習時mask適用確認
- [ ] 決定論デコード順序確認
- [ ] policy bias初期値確認
- [ ] アクション分布ログ確認
- [ ] エントロピー推移確認

### 評価パイプライン検証 ⏳
- [ ] mask適用確認(`paper_trade.py`修正済み)
- [ ] 正規化統計一致確認
- [ ] temperature評価追加
- [ ] verbose出力動作確認

### 診断・可視化 ✓
- [x] `ActionDiagnostics`ツール作成
- [ ] 学習ループ統合
- [ ] 評価時診断ログ
- [ ] プロット生成・確認

## まとめ

### 現時点の状況

- **問題の本質**: 学習と評価のaction mask適用不一致が主因の可能性が高い
- **実施済み**: 診断ツール、テストスイート、包括的ドキュメント作成完了
- **次のステップ**: Phase 1(即効修正)の実装とテスト

### 他AIエージェントへの引き継ぎポイント

1. **作成済みファイル**を確認
   - `ztb/utils/diagnostics/action_diagnostics.py`
   - `tests/unit/environment/test_forced_actions.py`
   - `docs/fix_sell_bias.md`
   - `docs/action_bias_implementation_guide.md`

2. **実装ガイド**(`docs/action_bias_implementation_guide.md`)の§2-§7を順次実装

3. **検証**: 50k×3seed短距離学習で変更前後を比較

4. **報告**: 結果を`docs/fix_sell_bias.md`に追記

---

**作成日**: 2025-10-06  
**ステータス**: 診断・計画フェーズ完了、実装フェーズ開始準備完了
