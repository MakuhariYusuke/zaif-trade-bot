# AI コーディングエージェントへのレビュー依頼プロンプト (54)

**目的** ✅
- 本番実行前に `scripts/v459/run_day6_reward_tuning.py` と関連設定（`configs/rewards/*.yaml` 等）を完全にレビューし、実行の可否判定、リスク特定、迅速な修正案を出してください。

---

## 検査対象ファイル（優先度順） 🔍
- `scripts/v459/run_day6_reward_tuning.py`  (主対象)
- `scripts/v459/run_ab_reward_experiments.py` (関連)
- `configs/rewards/*.yaml` (stage1_*.yaml 含む)
- `docs/v459/52_phase4_week2_implementation_plan_revised.md`（実行手順確認）
- `docs/v459/53_time_optimization_strategies.md`（時間短縮論拠）

---

## 依頼事項（必須チェックリスト） ✅
1. コード品質: 明示的なバグ、例外ハンドリング漏れ、型誤り、潜在的メモリリークを指摘してください。
2. 再現性: seed管理、ログと結果保存（numpy→Python型変換）の妥当性を確認してください。
3. ハイパーパラメータ安全性: `buffer_size`, `learning_starts`, `batch_size`, `gamma`, `ent_coef` などが実行環境・目的に適合するか評価してください。
4. 実行時間見積り: 各構成（seeds × configs × timesteps）について推定所要時間と最悪ケースを提示してください。
5. 時間短縮策の副作用: 推奨案（buffer最適化等）が性能に与える影響とリスクを簡潔に評価してください。
6. テスト提案: 最小限の単体/統合テスト（失敗再現ケース含む）と追加すべきテスト名を列挙してください。
7. ログ・出力: 失敗時のデバッグに必要なログ出力箇所を提案してください。
8. セキュリティ/安全性: 長時間実行中のリソース監視や中断ポリシーのチェック項目を提示してください。

---

## 出力フォーマット（JSON）📦
返答は以下 JSON 構造でお願いします。

{
  "summary": "短い要約（1-2文）",
  "issues": [
    {"severity": "critical|high|medium|low", "file": "path", "line": number_or_null, "message": "説明", "suggested_fix": "patch or snippet"}
  ],
  "quick_fixes": [ {"file": "path", "patch": "差分 or diff snippet", "reason": "説明"} ],
  "tests_to_add": [ {"file": "tests/...", "name": "test_name", "purpose": "何を検証するか"} ],
  "runtime_estimate": {"per_experiment_min": "xxm", "per_experiment_max": "xxm", "total_estimate": "yyh"},
  "go_decision": "go|no-go|go-with-conditions",
  "notes": "追加コメント（任意）"
}

---

## 制約事項 / 背景
- 実行環境: Windows, 仮想環境あり。.venv を使っています。
- 現行設定: seeds=[42,123], total_timesteps=50_000（維持予定）、推奨: buffer_size=25000, learning_starts=500。
- 目的: ROI の改善（50,000stepで-5%→0%以上）と51番レビュー対応の遵守。

---

## 優先アクション（期待する成果） 🎯
1. 重大なバグがある場合は即時 "no-go" とし、最短パッチを提示する。
2. 重大問題が無ければ "go-with-conditions"（例: 追加監視ログ、1つのショートラン検証）を推奨する。
3. 追加の小修正 (low/medium) はパッチとテストコードを提供してください。

---

**備考**: 出力は簡潔に、技術的根拠と推奨度（★〜☆☆☆）を付けてください。時間見積りは実測ベースでなくても構いませんが、根拠を明記してください。

---

作業が終わったら、`go_decision` に基づく実行可否と、実行時に監視すべきログ行（キーワード）を3つ提示してください。