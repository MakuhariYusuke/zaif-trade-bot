"""136# P1-01/02/03 テスト.

- RetrainTrigger: fill_records mtime チェック、trades health ガード、バックオフ
- §9 #1 回帰テスト: unhealthy→healthy 同一 mtime で retrain が走ること
- §9 #2: feature freshness → trigger 統合
- FeatureFreshness: trades/OB 鮮度チェック
- SellDynamicKillManager: rolling kill、cooldown、レジーム別閾値、テレメトリ
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ===== P1-01: RetainTrigger =====

class TestRetainTrigger:
    """RetrainTriggerの事前チェック (§9 #5: 後方互換エイリアスも確認)."""

    def test_skip_when_no_fill_records_updated(self, tmp_path: Path) -> None:
        """fill_records 存在するが mtime 変化なし → 2回目はスキップ."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(check_trades_health=False)
        trigger = RetrainTrigger(results_dir=tmp_path, config=cfg)

        # ファイル作成
        fr = tmp_path / "fill_records_test.jsonl"
        fr.write_text('{"x":1}\n')

        # 初回: mtime 取得 → 通過
        ok, reason = trigger.should_retrain()
        assert ok is True

        # 2回目: mtime 変化なし → スキップ
        ok2, reason2 = trigger.should_retrain()
        assert ok2 is False
        assert "unchanged" in reason2

    def test_pass_when_fill_records_updated(self, tmp_path: Path) -> None:
        """fill_records 更新 → 通過."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(check_trades_health=False)
        trigger = RetrainTrigger(results_dir=tmp_path, config=cfg)

        # ファイル作成
        fr = tmp_path / "fill_records_test.jsonl"
        fr.write_text('{"x":1}\n')
        ok, _ = trigger.should_retrain()
        assert ok is True

        # ファイル更新
        import os
        os.utime(str(fr), (time.time() + 10, time.time() + 10))
        ok2, _ = trigger.should_retrain()
        assert ok2 is True

    def test_skip_when_trades_unhealthy(self, tmp_path: Path) -> None:
        """trades 欠損 → blocked."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            check_fill_records_mtime=False,
            check_trades_health=True,
            trades_lookback_days=1,
        )
        trigger = RetrainTrigger(results_dir=tmp_path, raw_dir=tmp_path, config=cfg)
        # trades ディレクトリなし → unhealthy
        ok, reason = trigger.should_retrain()
        assert ok is False
        assert "unhealthy" in reason

    def test_trades_max_missing_days_tolerance(self, tmp_path: Path) -> None:
        """158# max_missing_days=1 で 1 日欠損を許容."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        # lookback_days=3 で 3 日中 1 日欠損のセットアップ
        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        for i in [1, 3]:  # yesterday と 3 日前は存在、2 日前は欠損
            day = (now - timedelta(days=i)).strftime("%Y%m%d")
            f = tr_dir / f"{day}.jsonl.gz"
            import gzip as _gz
            f.write_bytes(_gz.compress(b'{"t":1}\n'))

        # max_missing_days=0 (厳密) → UNHEALTHY
        cfg_strict = RetrainTriggerConfig(
            check_fill_records_mtime=False,
            check_trades_health=True,
            trades_lookback_days=3,
            trades_max_missing_days=0,
        )
        trigger_strict = RetrainTrigger(
            results_dir=tmp_path, raw_dir=tmp_path, config=cfg_strict
        )
        ok, reason = trigger_strict.should_retrain()
        assert ok is False
        assert "missing" in reason.lower() or "unhealthy" in reason.lower()

        # max_missing_days=1 (寛容) → HEALTHY
        cfg_tolerant = RetrainTriggerConfig(
            check_fill_records_mtime=False,
            check_trades_health=True,
            trades_lookback_days=3,
            trades_max_missing_days=1,
        )
        trigger_tolerant = RetrainTrigger(
            results_dir=tmp_path, raw_dir=tmp_path, config=cfg_tolerant
        )
        ok2, reason2 = trigger_tolerant.should_retrain()
        assert ok2 is True

    def test_backoff_increases_interval(self) -> None:
        """連続スキップでバックオフ倍増."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=100,
            backoff_multiplier=2.0,
            backoff_max_interval_sec=1000,
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        assert trigger.get_effective_interval() == 100

        trigger.record_result("skipped")
        assert trigger.get_effective_interval() == 200  # 100 * 2^1

        trigger.record_result("skipped")
        assert trigger.get_effective_interval() == 400  # 100 * 2^2

        trigger.record_result("skipped")
        assert trigger.get_effective_interval() == 800  # 100 * 2^3

        trigger.record_result("skipped")
        assert trigger.get_effective_interval() == 1000  # capped

    def test_backoff_resets_on_deploy(self) -> None:
        """deploy 成功でバックオフリセット."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(base_interval_sec=100, backoff_multiplier=2.0)
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.record_result("skipped")
        trigger.record_result("skipped")
        assert trigger.consecutive_skips == 2

        trigger.record_result("deployed")
        assert trigger.consecutive_skips == 0
        assert trigger.get_effective_interval() == 100

    def test_unhealthy_to_healthy_same_mtime_retrain_fires(self, tmp_path: Path) -> None:
        """§9 #A 回帰: unhealthy→healthy 同一 mtime で retrain が走る.

        #1 FIX 前は mtime が先に消費されたため、health 復帰後に
        「fill_records unchanged」で false skip していた。
        """
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        # trades ディレクトリ = tmp_path/trades — 作成しない (unhealthy)
        cfg = RetrainTriggerConfig(
            check_fill_records_mtime=True,
            check_trades_health=True,
            trades_lookback_days=1,
            trades_stale_threshold_hours=999.0,  # stale は無視
        )
        trigger = RetrainTrigger(
            results_dir=tmp_path, raw_dir=tmp_path, config=cfg
        )

        # fill_records 作成
        fr = tmp_path / "fill_records_test.jsonl"
        fr.write_text('{"x":1}\n')

        # 1st call: fill_records に変化あり、だが trades unhealthy → blocked
        ok1, reason1 = trigger.should_retrain()
        assert ok1 is False, "trades 不在なので blocked のはず"
        assert "unhealthy" in reason1

        # 2nd call: trades を用意 (同じ mtime のまま healthy に復帰)
        # → #1 FIX により mtime は消費されていないので retrain が通る
        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        import gzip
        from datetime import datetime, timedelta, timezone
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        (trades_dir / f"{yesterday}.jsonl.gz").write_bytes(
            gzip.compress(b'{"ts":1}\n')
        )

        ok2, reason2 = trigger.should_retrain()
        assert ok2 is True, f"健全化後は retrain が走るべき: {reason2}"

    def test_feature_freshness_integrated_in_trigger(self, tmp_path: Path) -> None:
        """§9 #2: check_feature_freshness=True で stale → skip."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            check_fill_records_mtime=False,
            check_trades_health=False,
            check_feature_freshness=True,
            feature_trades_stale_hours=0.001,  # 即時 stale
            feature_ob_stale_hours=0.001,
        )
        trigger = RetrainTrigger(results_dir=tmp_path, raw_dir=tmp_path, config=cfg)
        # trades/OB ディレクトリなし → stale
        ok, reason = trigger.should_retrain()
        assert ok is False
        assert "stale" in reason.lower()

    def test_backward_compat_alias(self) -> None:
        """§9 #5: RetainTrigger/RetainTriggerConfig エイリアスが存在する."""
        from ztb.ml.retrain_trigger import (
            RetainTrigger,
            RetainTriggerConfig,
            RetrainTrigger,
            RetrainTriggerConfig,
        )

        assert RetainTriggerConfig is RetrainTriggerConfig
        assert RetainTrigger is RetrainTrigger


# ===== 158# trades_health max_missing_days =====

class TestTradesHealthMaxMissing:
    """158# check_trades_health の max_missing_days パラメータ."""

    def test_strict_rejects_any_missing(self, tmp_path: Path) -> None:
        """max_missing_days=0 → 1 日欠損で UNHEALTHY."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        # day-1 存在, day-2 欠損
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(
            raw_dir=tmp_path, lookback_days=2, max_missing_days=0,
        )
        assert not result.healthy
        assert len(result.missing_days) == 1

    def test_tolerant_allows_one_gap(self, tmp_path: Path) -> None:
        """max_missing_days=1 → 1 日欠損でも HEALTHY (fresh 条件下)."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(
            raw_dir=tmp_path, lookback_days=2, max_missing_days=1,
        )
        assert result.healthy
        assert "tolerated_gaps" in result.message

    def test_tolerant_rejects_too_many_gaps(self, tmp_path: Path) -> None:
        """max_missing_days=1 → 2 日欠損は UNHEALTHY."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(
            raw_dir=tmp_path, lookback_days=3, max_missing_days=1,
        )
        assert not result.healthy
        assert "max_allowed=1" in result.message

    def test_message_contains_tolerated_info(self, tmp_path: Path) -> None:
        """OK メッセージに tolerated gaps 情報が含まれる."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        for i in [1, 3]:
            day = (now - timedelta(days=i)).strftime("%Y%m%d")
            (tr_dir / f"{day}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(
            raw_dir=tmp_path, lookback_days=3, max_missing_days=1,
        )
        assert result.healthy
        assert "OK" in result.message
        assert len(result.missing_days) == 1


class TestTradesHealthResultFields:
    """159# §2.1: TradesHealthResult のフィールド整合性テスト.

    run_fill_test.py が参照するフィールドが TradesHealthResult に存在すること、
    かつ旧 latest_ts / age_hours が存在しないことを確認。
    """

    def test_result_has_stale_hours(self, tmp_path: Path) -> None:
        """stale_hours フィールドが float で取得可能."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(raw_dir=tmp_path, lookback_days=1)
        assert isinstance(result.stale_hours, float)
        assert result.stale_hours >= -1e-6  # 浮動小数点誤差を許容

    def test_result_has_available_days_list(self, tmp_path: Path) -> None:
        """available_days からインデックスアクセス可能."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        result = check_trades_health(raw_dir=tmp_path, lookback_days=1)
        assert len(result.available_days) >= 1
        latest_day = result.available_days[-1]
        assert len(latest_day) == 8 and latest_day.isdigit()

    def test_no_latest_ts_attribute(self) -> None:
        """159# §2.1: latest_ts は存在しない → hasattr=False."""
        from ztb.data.trades_health import TradesHealthResult

        r = TradesHealthResult(
            healthy=True, available_days=["20260223"],
            missing_days=[], stale_hours=0.5, message="OK",
        )
        assert not hasattr(r, "latest_ts")

    def test_no_age_hours_attribute(self) -> None:
        """159# §2.1: age_hours は存在しない → hasattr=False."""
        from ztb.data.trades_health import TradesHealthResult

        r = TradesHealthResult(
            healthy=True, available_days=["20260223"],
            missing_days=[], stale_hours=0.5, message="OK",
        )
        assert not hasattr(r, "age_hours")

    def test_event_details_format(self, tmp_path: Path) -> None:
        """run_fill_test.py の修正後 event details 形式が構築可能."""
        from ztb.data.trades_health import check_trades_health

        tr_dir = tmp_path / "trades"
        tr_dir.mkdir()
        from datetime import datetime, timedelta, timezone
        import gzip as _gz

        now = datetime.now(timezone.utc)
        day1 = (now - timedelta(days=1)).strftime("%Y%m%d")
        (tr_dir / f"{day1}.jsonl.gz").write_bytes(_gz.compress(b'{"t":1}\n'))

        th = check_trades_health(raw_dir=tmp_path, lookback_days=2)
        # 159# §2.1 fix: 修正後のフィールド参照が例外なしで構築可能
        details = {
            "healthy": th.healthy,
            "latest_day": th.available_days[-1] if th.available_days else None,
            "missing_days": th.missing_days,
            "stale_hours": round(th.stale_hours, 1),
        }
        assert "latest_day" in details
        assert details["stale_hours"] >= 0
        assert isinstance(details["missing_days"], list)


# ===== P1-02: Feature Freshness =====

class TestFeatureFreshness:
    """FeatureFreshnessResult のチェック."""

    def test_fresh_when_recent_files(self, tmp_path: Path) -> None:
        """trades/OB ファイルが新しい → fresh."""
        from ztb.data.trades_health import check_feature_freshness

        trades_dir = tmp_path / "trades"
        ob_dir = tmp_path / "orderbook"
        trades_dir.mkdir()
        ob_dir.mkdir()

        # 最近のファイルを作成
        (trades_dir / "20260222.jsonl.gz").write_bytes(
            gzip.compress(b'{"ts":1}\n')
        )
        (ob_dir / "20260222.jsonl.gz").write_bytes(
            gzip.compress(b'{"ts":1}\n')
        )

        result = check_feature_freshness(raw_dir=tmp_path, trades_stale_hours=24.0, ob_stale_hours=24.0)
        assert result.fresh is True
        assert result.trades_stale_hours < 24.0
        assert result.ob_stale_hours < 24.0
        assert "FRESH" in result.message

    def test_stale_when_no_files(self, tmp_path: Path) -> None:
        """trades/OB ファイルなし → stale."""
        from ztb.data.trades_health import check_feature_freshness

        result = check_feature_freshness(raw_dir=tmp_path)
        assert result.fresh is False
        assert result.trades_stale_hours == float("inf")
        assert "STALE" in result.message

    def test_partial_stale(self, tmp_path: Path) -> None:
        """trades OK、OB なし → stale."""
        from ztb.data.trades_health import check_feature_freshness

        trades_dir = tmp_path / "trades"
        trades_dir.mkdir()
        (trades_dir / "20260222.jsonl.gz").write_bytes(
            gzip.compress(b'{"ts":1}\n')
        )

        result = check_feature_freshness(
            raw_dir=tmp_path, trades_stale_hours=24.0, ob_stale_hours=24.0
        )
        assert result.fresh is False
        assert "OB stale" in result.message


# ===== P1-03: SellDynamicKillManager =====

class TestSellDynamicKillManager:
    """SellDynamicKillManager の単体テスト."""

    def test_not_killed_when_insufficient_data(self) -> None:
        """データ不足 → kill しない."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(window=5, threshold_bps=-1.0))
        for _ in range(3):
            mgr.track(pnl_bps=-2.0)
        killed, tel = mgr.check_kill()
        assert killed is False
        assert tel.rolling_mean is None  # insufficient data

    def test_killed_when_below_threshold(self) -> None:
        """rolling mean < threshold → kill."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(
            window=3, threshold_bps=-0.5, resume_window=2,
        ))
        for _ in range(3):
            mgr.track(pnl_bps=-1.0)

        killed, tel = mgr.check_kill()
        assert killed is True
        assert tel.rolling_mean == pytest.approx(-1.0)
        assert tel.total_kills == 1
        assert tel.cooldown_remaining == 2

    def test_cooldown_decrements(self) -> None:
        """cooldown 中は killed を返し、カウントダウンする."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(
            window=2, threshold_bps=-0.5, resume_window=3,
        ))
        mgr.track(-1.0)
        mgr.track(-1.0)
        killed, _ = mgr.check_kill()  # kill activated, cooldown=3
        assert killed is True

        # cooldown 消化
        k1, t1 = mgr.check_kill()  # cooldown 2
        assert k1 is True
        assert t1.cooldown_remaining == 2
        k2, t2 = mgr.check_kill()  # cooldown 1
        assert k2 is True
        assert t2.cooldown_remaining == 1
        k3, t3 = mgr.check_kill()  # cooldown 0 → 再評価
        assert k3 is True
        assert t3.cooldown_remaining == 0
        # cooldown 終了後、データがまだ悪ければ再 kill
        k4, t4 = mgr.check_kill()
        assert k4 is True  # still below threshold
        assert t4.total_kills == 2

    def test_not_killed_when_above_threshold(self) -> None:
        """rolling mean >= threshold → OK."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(
            window=3, threshold_bps=-0.5,
        ))
        mgr.track(0.1)
        mgr.track(0.2)
        mgr.track(-0.3)

        killed, tel = mgr.check_kill()
        assert killed is False
        assert tel.rolling_mean == pytest.approx(0.0, abs=0.01)

    def test_regime_threshold_override(self) -> None:
        """レジーム別閾値が適用される."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(
            window=3,
            threshold_bps=-0.5,
            regime_thresholds={"volatile": -2.0},  # volatile は緩い
        ))
        for _ in range(3):
            mgr.track(-1.0)

        # default threshold (-0.5) → killed
        killed_default, tel_d = mgr.check_kill(regime=None)
        assert killed_default is True
        assert tel_d.threshold_used == -0.5

        # reset and retry with volatile regime
        mgr.reset()
        for _ in range(3):
            mgr.track(-1.0)
        killed_vol, tel_v = mgr.check_kill(regime="volatile")
        assert killed_vol is False  # -1.0 > -2.0
        assert tel_v.threshold_used == -2.0

    def test_disabled(self) -> None:
        """enabled=False → 常に not killed."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(enabled=False))
        for _ in range(100):
            mgr.track(-5.0)
        killed, _ = mgr.check_kill()
        assert killed is False

    def test_memory_limit(self) -> None:
        """window*3 超過で古いデータが切り捨てられる."""
        from ztb.risk.sell_dynamic_kill import SellDynamicKillManager, SellKillConfig

        mgr = SellDynamicKillManager(SellKillConfig(window=3))
        for i in range(20):
            mgr.track(float(i))
        assert len(mgr._pnl_history) <= 9  # window*3


# =====================================================================
# 145# R-2b: レジーム別 retrain interval テスト
# =====================================================================

class TestRetrainTriggerRegimeInterval:
    """145# R-2b: レジーム別 interval 倍率テスト."""

    def test_high_vol_shortens_interval(self) -> None:
        """high_vol レジーム → 基本 interval の 50%."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=1000,
            backoff_multiplier=1.0,
            regime_interval_multipliers={"high_vol": 0.5, "ranging": 1.5},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.update_regime("high_vol")
        assert trigger.get_effective_interval() == 500  # 1000 * 0.5

    def test_ranging_lengthens_interval(self) -> None:
        """ranging レジーム → 基本 interval の 150%."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=1000,
            backoff_multiplier=1.0,
            regime_interval_multipliers={"ranging": 1.5},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.update_regime("ranging")
        assert trigger.get_effective_interval() == 1500

    def test_regime_with_backoff_combined(self) -> None:
        """バックオフとレジーム倍率の組み合わせ."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=100,
            backoff_multiplier=2.0,
            backoff_max_interval_sec=10000,
            regime_interval_multipliers={"high_vol": 0.5},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.update_regime("high_vol")
        # skip 0 → 100 * 0.5 = 50
        assert trigger.get_effective_interval() == 50

        trigger.record_result("skipped")
        # skip 1 → 100 * 2^1 * 0.5 = 100
        assert trigger.get_effective_interval() == 100

        trigger.record_result("skipped")
        # skip 2 → 100 * 2^2 * 0.5 = 200
        assert trigger.get_effective_interval() == 200

    def test_record_result_updates_regime(self) -> None:
        """record_result で current_regime が更新される."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=1000,
            backoff_multiplier=1.0,
            regime_interval_multipliers={"trending": 0.75},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        assert trigger._current_regime == "unknown"

        trigger.record_result("deployed", current_regime="trending")
        assert trigger._current_regime == "trending"
        assert trigger.get_effective_interval() == 750  # 1000 * 0.75

    def test_unknown_regime_default_multiplier(self) -> None:
        """未知レジーム → 倍率 1.0 (デフォルト)."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=1000,
            backoff_multiplier=1.0,
            regime_interval_multipliers={"high_vol": 0.5},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.update_regime("some_new_regime")
        assert trigger.get_effective_interval() == 1000  # 1.0x

    def test_regime_capped_at_max_interval(self) -> None:
        """ranging でも max_interval を超えない."""
        from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

        cfg = RetrainTriggerConfig(
            base_interval_sec=10000,
            backoff_multiplier=1.0,
            backoff_max_interval_sec=12000,
            regime_interval_multipliers={"ranging": 2.0},
        )
        trigger = RetrainTrigger(results_dir=Path("/nonexistent"), config=cfg)
        trigger.update_regime("ranging")
        # 10000 * 2.0 = 20000, but capped at 12000
        assert trigger.get_effective_interval() == 12000

    def test_default_config_has_regime_multipliers(self) -> None:
        """デフォルト config にレジーム倍率が設定されている."""
        from ztb.ml.retrain_trigger import RetrainTriggerConfig

        cfg = RetrainTriggerConfig()
        assert "high_vol" in cfg.regime_interval_multipliers
        assert cfg.regime_interval_multipliers["high_vol"] == 0.5
        assert cfg.regime_interval_multipliers["ranging"] == 1.5
