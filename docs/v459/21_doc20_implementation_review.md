# Doc20 実装レビュー対応 再レビュー (21)

**Date**: 2026-01-24  
**Status**: 📝 Review  
**Targets**: `docs/v459/20_doc19_implementation_review_response.md`, `ztb/trading/environment/fast_intraday_env_v456.py`, `ztb/evaluation/walk_forward/evaluator.py`, `ztb/evaluation/walk_forward/reporter.py`, `tests/unit/trading/test_close_reason.py`

---

## Findings (Critical -> Major -> Minor)
- [Major] `prev_entry_price`はinfoに追加されたものの、Evaluatorは依然`entry_price`のみを使用しており、反転クローズ側のentry_price誤りが解消されていません。Doc20の「Evaluatorがprev_entry_priceを使用してPnL計算」は未実装です。`docs/v459/20_doc19_implementation_review_response.md:130` `ztb/evaluation/walk_forward/evaluator.py:440` `ztb/trading/environment/fast_intraday_env_v456.py:844`
- [Major] 反転取引のPnL配賦は依然として50/50分割で、Doc20の「decompose_reversal実装済み」や「クローズ側全PnL・新規側コストのみ」ルールと不一致です。`docs/v459/20_doc19_implementation_review_response.md:156` `ztb/evaluation/walk_forward/reporter.py:387`
- [Major] Add/Reduce時にも`entry_price`が更新されるため、残存ポジションの基準価格が上書きされます。PnLを確定しない設計と整合せず、将来のクローズPnLやTP/SL判定が歪みます。`ztb/trading/environment/fast_intraday_env_v456.py:667` `ztb/trading/environment/fast_intraday_env_v456.py:686`
- [Major] Add/Reduceの手数料・スリッページが`trade_pnl`に反映されず、NET PnLが過大評価されます（記録上は0のまま）。`ztb/trading/environment/fast_intraday_env_v456.py:667` `ztb/trading/environment/fast_intraday_env_v456.py:696`
- [Minor] Doc20は`info['position_before']`/`info['position_after']`が既存と記載していますが、info辞書には存在しません。`docs/v459/20_doc19_implementation_review_response.md:257` `ztb/trading/environment/fast_intraday_env_v456.py:821`
- [Minor] Doc20は`decompose_reversal()`という関数名で記載していますが、実装は`decompose_reverse_trade()`です。記載と実装の不一致が残っています。`docs/v459/20_doc19_implementation_review_response.md:150` `ztb/evaluation/walk_forward/reporter.py:65`
- [Minor] 追加変更のテストがなく、`prev_entry_price`伝搬や反転PnL配賦ルールを検証できません（既存テストのアサーション緩和のみ）。`tests/unit/trading/test_close_reason.py:106`

---

## Open Questions / Assumptions
- 反転時のクローズ側`entry_price`は、Evaluatorで`prev_entry_price`を使う方針で確定ですか？（infoに`entry_price_close`/`entry_price_open`を分けて渡す案もあり）
- Add/Reduceを許容するなら、加重平均entry_priceと部分決済PnLの設計をPhase 2で固定しますか？それともAdd/Reduce自体を抑止しますか？
- Add/Reduceの取引コストは`trade_pnl`に含める前提で良いですか？（NET PnL規約との整合）

---

## Change Summary (Doc20の実績整理)
- env側の`prev_entry_price`保存とrecorder呼び出し停止は実装済み。
- PnL計算は「完全クローズ/反転時のみ」に変更済み。
- テストは反転時`close_reason`アサーションを緩和。
- ただし、Evaluator/Reporterの配賦・entry_price取り扱いが未整合のため、修正完了とは言い切れません。
