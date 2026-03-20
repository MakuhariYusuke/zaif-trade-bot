# 511# docs: 504-510 命名規約適合リネーム + index.md 更新

## 概要
ドキュメント 504#-510# のファイル名を v460 命名規約 (`NNN_phX_type_description.md`) に適合させるリネームと、index.md の更新。

## 変更内容

### ファイルリネーム
| 旧名 | 新名 |
|---|---|
| (旧名不明) | `509_ph2_fix_sell_age_cap_micro_timeout_guard.md` |
| (旧名不明) | `510_ph2_impl_inv_skew_periodic_summary_vg_reason.md` |

### index.md 更新
- 497#-510# の 12 エントリを index.md に追加
- 最終更新日付を `500# 497#-499#/503#レビュー` に更新
- 追加されたドキュメント:
  - 497# rpt: config 変更影響検証
  - 498# fix: hot-reload 対象拡張 (348→409 fields)
  - 499# fix: hard_loss_cap crash loop 修正
  - 500# rev: sell 崩壊と非対称実行の再整理
  - 503# rpt: Buy/Sell 損益要因分析
  - 504# rev: lib→ztb 移行計画レビュー
  - 505# resp: 504# 応答
  - 506# verify: 500#/501# レビュー検証
  - 507# fix: Confidence/Velocity De-meaning + Recovery Skew
  - 508# impl: ログ・可観測性改善
  - 509# fix: sell_age_cap × micro_timeout ガード
  - 510# impl: inv_skew_factor + 周期サマリ + VG reason

## 変更ファイル
| ファイル | 変更内容 |
|---|---|
| `docs/v460/509_*.md` | リネーム |
| `docs/v460/510_*.md` | リネーム |
| `docs/v460/index.md` | 12 エントリ追加 + 更新日付変更 |
