# Zaif Trade Bot Operations

## 静的ダッシュボード生成

- `scripts/make_dashboard.py` で artifacts/index.json と各 summary.json を集約して HTML ダッシュボード生成
- セッション一覧（ID/開始時刻/最新step/ETA/ステータス）と主要メトリクス（Sharpe/DSR/p-value/メモリ/RSSピーク）を表形式で表示
- `--out artifacts/dashboard.html` で出力先指定

## セッション横断トレンド集計

- `scripts/aggregate_trends.py` で artifacts/*/summary.json から global_step, rl_sharpe, dsr, p_value を CSV 出力
- 欠損値は空欄扱い、session_id でソート
- `--out artifacts/trends.csv` で出力先指定

## "ops doctor"（一次切り分けワンショット）

- `scripts/ops_doctor.py` で validate_artifacts → collect_last_errors → progress_eta → disk_health を順次実行
- 失敗しても継続、要約（OK/WARN/FAIL数）を1行表示、詳細は artifacts/`<ID>`/reports/doctor.txt に保存
- `--correlation-id <ID>` で対象セッション指定

## 毎晩のレポート生成

- `scripts/make_trends_md.py` で trends.csv から Markdown レポート生成
- メトリクス別テーブルと ASCII スパークラインでトレンド可視化
- `--out reports/trends.md` で出力先指定

## コスト見積り（GPU/電力/クラウド）

- `scripts/cost_estimator.py` で run_metadata + steps/sec + 単価からコスト見積り
- 出力: reports/budget_estimate.json または .md
- パラメータ例: --gpu-rate 300 (GPU時間単価, 円/時), --kwh-rate 35 (電力単価, 円/kWh)
- ENV変数: GPU_RATE_JPY_PER_HOUR, KWH_RATE_JPY, CLOUD_RATE_JPY_PER_HOUR でデフォルト設定

## コスト算出：日本の従量電灯（40A）

- --tariff jp_residential_tiered で三段階料金適用（基本料金1246.96円/月込み）
  - 0–120 kWh: 29.70 円/kWh
  - 120–300 kWh: 35.69 円/kWh  
  - 300+ kWh: 39.50 円/kWh
- 月間消費kWhは run_metadata/tb_summary/steps_per_sec から推定、--kwh で手動上書き可
- CLI: python -m ztb.ops.cost_estimator --tariff jp_residential_tiered --kwh `<override>`

## ゲート失敗時の自動通知

- `scripts/gates_to_alerts.py` で `artifacts/<ID>/reports/gates.json` を読み、失敗を webhook に通知
- gates.json が無い場合は SKIP、失敗時のみ WARN/ERROR レベルで通知
- --webhook で URL 指定、ENV $SLACK_WEBHOOK_URL も可

## 通知設定：Discord

- Discord Webhook 対応で Slack 未設定でも通知可能
- Webhook 作成: Discordサーバー設定 → 連携サービス → Webhook → URLコピー
- ENV優先順位: --discord-webhook > ZTB_DISCORD_WEBHOOK > --webhook > SLACK_WEBHOOK_URL
- カスタム設定: ZTB_DISCORD_USERNAME, ZTB_DISCORD_AVATAR_URL でボット表示変更
- レート制限429時は指数バックオフで自動リトライ（最大3回）

## スケジュールテンプレート生成

- `scripts/schedule_templates.py` で cronish.py のプリセットJSONを生成
- クロスプラットフォーム対応 (Windows/Linux cron 形式)
- テンプレート: daily, hourly, weekly, monthly, training-daily, backup-hourly
- --list で利用可能テンプレート一覧、--template で指定生成

## スケジューリング：キャッチアップと常駐運用

- `scripts/cronish.py` で定期コマンド実行、--max-catchup で起動時取りこぼし補填（最大回数制限）
- --catchup-cooldown-sec で補填実行間クールダウン
- 24/7運用: ops/systemd/ztb-live.service (Linux systemd) または ops/windows/task_scheduler.xml (Windows タスクスケジューラ)
- PC再起動後に1回だけ取りこぼしを実行、systemd/Windows の雛形を提供

## TensorBoard サマリースクレイパー

- `scripts/tb_scrape_summary.py` で runs/ のTBログから最新スカラー値をJSON抽出
- 各実験ディレクトリをスキャン、scalars_summary.json がある場合は読み込み
- --run-dir で対象ディレクトリ指定、--output でJSON出力ファイル指定

## 予算ロールアップレポート

- `scripts/budget_rollup.py` で run_metadata.json と cost_estimate.json を集計
- 日次コストレポートを reports/budget_daily.md にmarkdown出力
- --date で特定日付フィルタ、総コスト/GPU時間/実行回数を集計

## 外部統合互換性ラッパー

- `scripts/compat_wrapper.py` で外部ツールからのJSONベース統合を提供
- stdinからJSONリクエストを受け取り、stdoutにJSONレスポンスを返す
- action: "status" でライブステータス取得、 "run" でコマンド実行
- 例: echo '{"action": "status"}' | python scripts/compat_wrapper.py
- 外部監視ツールやCI/CDパイプラインとの統合に適する

## ライブステータススナップショット（Discord配信）

- `scripts/live_status_snapshot.py` で trade-state.json, trade-config.json, stats.json から現在のボットステータスを収集
- Discord Webhook に embed 形式で配信（フェーズ/連勝日数/P&L/設定値など）
- コマンド: python scripts/live_status_snapshot.py `<discord_webhook_url>`
- ファイルが存在しない場合はスキップ、部分データでも配信可能
