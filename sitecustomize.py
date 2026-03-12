"""Project-level import fixes executed very early on interpreter startup.

379# 修正: ローカル stable_baselines3 スタブ (_sb3_test_stub/) が
本物の pip 版 SB3 をシャドウしていたため、SAC.learn() が何もせず
SAC.predict() が常に int(0) を返していた。

384# cleanup: SB3 スタブ関連の dead code を削除。
pip 版 SB3 2.7.0 が正常にロードされるため、ローカルスタブ回避ロジックは不要。
テスト用スタブが必要な場合は tests/conftest.py で注入する。
"""
