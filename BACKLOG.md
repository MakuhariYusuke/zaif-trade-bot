# Backlog

## Position Store Recovery Enhancements

- B案: 履歴ベースの完全再構築
  - アプローチ: 追記型ログ (append-only WAL) から最新スナップショットを再合成
  - Pros: 破損時に累積精度を最大化 / デバッグ容易
  - Cons: IO 増, ログ肥大のローテーション戦略必要

- C案: .bak ファイルからの復旧機能
  - アプローチ: 各保存時に atomic write + 直前バージョンを `<name>.bak` へコピーし二段階フォールバック
  - Pros: 実装容易, 破損発生時の直前状態ロールバック
  - Cons: 単一直前しか保持しないため連続破損には弱い

現状優先度: Low (運用上致命的インシデント発生時に再評価)
