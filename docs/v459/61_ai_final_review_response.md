{
  "summary": "報酬設定の注入経路とbehavior_optimization伝播は修正され、YAML検証も通過しました。未使用キーの整理とreward_scaleの実効反映だけ注意すれば長時間実行は概ねGOです。",
  "status_checks": [
    {
      "name": "yaml_validation",
      "status": "pass",
      "details": "実行確認済み: RewardConfigSchema.load_and_validate が stage1_* 4件 모두 OK。"
    },
    {
      "name": "config_injection",
      "status": "pass",
      "details": "create_experiment_config で reward_settings 注入 + behavior_optimization を environment へ分離注入済み（コード確認）。"
    },
    {
      "name": "runtime_smoke",
      "status": "pass",
      "details": "ユーザー報告ベース（--limit 1 完了・JSON保存確認）。本レビューでは再実行していません。"
    },
    {
      "name": "json_serialization",
      "status": "pass",
      "details": "save_results が numpy/dataclass/Path/datetime を処理。静的確認で問題なし。"
    },
    {
      "name": "unit_tests",
      "status": "pass",
      "details": "ユーザー報告ベース（tests/test_reward_config_integration.py 4/4）。本レビューでは再実行していません。"
    }
  ],
  "go_decision": "go-with-conditions",
  "quick_fixes": [
    {
      "file": "ztb/trading/environment/components/calculators/reward_calculator.py",
      "patch": "# 参考: reward_scale を有効化したい場合\n# reward_scaling = self.get_setting_float(\"reward_scaling\", 1.0)\n# reward_scaling = self.get_setting_float(\"reward_scaling\", self.get_setting_float(\"reward_scale\", 1.0))",
      "reason": "reward_scale が実効スケールに反映されないリスクを低減。現状は既知制限として運用でも可。"
    }
  ],
  "notes": "behavior_optimization は environment に伝搬済みで action_smoothing が効く前提になりました。未使用キー（max_drawdown_penalty_weight等）は RewardSettings/RewardCalculator で参照されないため、影響評価するなら対応キーへ寄せるか削除推奨。長時間実行はメモリ監視（>1GB時は再検討）を条件に推奨。"
}
