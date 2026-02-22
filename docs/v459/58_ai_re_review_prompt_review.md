# 58_ai_re_review_prompt_review

## 📋 概要
**目的**: 57番の再レビュー（報酬設定注入・behavior_optimization の伝播確認）で使用したプロンプトをまとめ、レビュー担当者が「様々な角度」から観察・報告できる仕組み（チェックリスト・再現手順・観察テンプレ）を組み込みます。✅

---

## 🔧 再レビュー依頼（送信テンプレ）
件名: 【再レビュー依頼】Day6 報酬設計修正・環境注入の修正（behavior_optimization の伝播含む）/ ユニットテスト追加

本文（そのまま送信可）:

> 概要：先の指摘（#51 / #55 / #57）に対応し、報酬設定の注入経路とスキーマ、不具合に関する修正を行いました。自動テストと最小スモーク実行を行い、ローカルでの検証を完了しています。以下に変更点と確認済み項目をまとめます。ご確認のうえ再レビュー（go/no-go）をお願いします。

短い確認リスト:
- RewardConfigSchema.load_and_validate() による辞書注入が `training.environment.reward_settings` に反映されているか
- `behavior_optimization` が `training.environment.behavior_optimization` に分離注入され、ランタイムで参照されるか
- `tests/test_reward_config_integration.py` が全てパスしているか
- スモーク（`--limit 1`）でトレーニングが完了し、レポート（JSON）が保存されるか

---

## 🧭 観察を促すチェックリスト（多角的）
- スキーマ・静的観察
  - [ ] YAML に必須キー (name/description/curriculum_stage/reward_scale) が存在する
  - [ ] 未知キーが無視される or 警告されるか（期待される挙動の確認）

- 注入経路・構造
  - [ ] `config["training"]["environment"]["reward_settings"]` が dict である
  - [ ] `config["training"]["environment"]["behavior_optimization"]` が存在する（キーと値を確認）
  - [ ] EnvironmentConfig.from_dict が該当キーを `RewardSettings` にマップしているか

- ランタイム（スモーク実行）
  - 実行コマンド（短縮）:
    - `pytest -q tests/test_reward_config_integration.py`
    - `python scripts/v459/run_day6_reward_tuning.py --limit 1`
    - `python scripts/v459/run_ab_reward_experiments.py --limit 1`
  - ログで確認するキーワード: `TRAINING COMPLETED`, `saved to`, `Final Reward`, `High memory usage`
  - [ ] トレーニングが正常に終了する
  - [ ] レポート JSON が生成され、`behavior_optimization` に関連する出力が反映されている（もしログに出ていれば確認）

- シリアライズ／保存
  - [ ] `save_results` の出力 JSON が dataclass / numpy / Path / datetime を破綻なく含む
  - [ ] 該当ファイルを開いて主要フィールド（reward_settings.name, behavior_optimization）を確認

- パフォーマンス・安定性
  - [ ] 実行中に致命的な例外がない
  - [ ] メモリ使用が急増していないか（ピーク値を記録）
  - [ ] 大きなメモリスパイクや頻繁な GC 警告が出ていないか

- 再現性
  - [ ] seed を固定して 2 回実行したとき挙動が概ね一致するか（重要メトリクスの差が小さいか）

---

## 🧾 レビュー時に記録して返却する情報（テンプレ）
- 実行コマンド: e.g., `python scripts/v459/run_day6_reward_tuning.py --limit 1`
- pytest 出力: 成功/失敗 + 主要失敗のスタックトレース
- スモーク出力（最後の 30 行）: （貼り付け）
- 生成されたレポートのパス & 抜粋（reward_settings.name, behavior_optimization の中身）
- メモリの最大値（MB）: （例: 1500MB）
- 最終的な判定: GO / NO-GO + 短い理由（1-2 文）

---

## ⚠️ レッドフラグ（即時対応推奨）
- ユニットテストが失敗する
- `behavior_optimization` が environment に存在しない
- トレーニングが途中で止まる / 例外を吐く
- JSON シリアライズで TypeError/ValueError が出る
- メモリ使用量が想定を大幅に超える（例: > 4GB on small worker）

---

## ✅ ゴー判定の簡易ルール（レビュワー向け）
- 全ユニットテスト通過 AND スモーク完了 AND `behavior_optimization` が環境に存在 → GO
- いずれかが失敗 or 不明瞭 → NO-GO（詳細を添えて返却）

---

## 💡 観察を促す追加仕組み（自動化・短期的運用）
- テスト強化案:
  - tests に `test_behavior_optimization_is_propagated` を追加し、create_experiment_config 結果で `environment["behavior_optimization"]` があることを assert する
  - JSON の round-trip テスト: `save_results` の出力を読み込み、主要キーが存在することをアサート
- 自動スモークジョブ（CI）:
  - PR マージ前に `--limit 1` スモークを CI 上で実行（時間制限付き）し、ログを自動添付
- 観察フォーム (小さな JSON): レビュワーはフォームに Yes/No/Notes を記入して返却できる（標準化されたフィードバック）

---

## 最後に（送信時の一言テンプレ）
> 以上が今回の修正点と確認事項です。特に **behavior_optimization の環境への伝播** と **ユニットテストの結果** を重点的にご確認ください。問題なければ GO のご判断をお願いします。

---

ファイル作成日時: 2026-01-29


---

*補足*: 必要があればこの文書をベースに `docs/v459/59_...` として「メモリ監視手順」「CI スモークの実装手順」などを分割して追加できます。💡