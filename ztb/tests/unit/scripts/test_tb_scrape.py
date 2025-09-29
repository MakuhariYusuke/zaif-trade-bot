import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ztb.scripts.tb_scrape import merge_to_metrics, scrape_scalars


def test_scrape_scalars_no_tb():
    with patch("ztb.scripts.tb_scrape.HAS_TENSORBOARD", False):
        scalars = scrape_scalars(Path("fake"), Path("fake.csv"))
        assert scalars == {}


def test_scrape_scalars_with_tb():
    with tempfile.TemporaryDirectory() as tmp:
        tb_dir = Path(tmp) / "tb"
        tb_dir.mkdir()
        csv_path = Path(tmp) / "scalars.csv"

        # Mock EventAccumulator
        with patch("ztb.scripts.tb_scrape.EventAccumulator") as mock_ea:
            mock_instance = MagicMock()
            mock_instance.Tags.return_value = {"scalars": ["test_tag"]}
            mock_instance.Scalars.return_value = [MagicMock(value=1.5)]
            mock_ea.return_value = mock_instance

            # Create fake event file
            event_file = tb_dir / "events.out.tfevents.123"
            event_file.touch()

            scalars = scrape_scalars(tb_dir, csv_path)
            assert "test_tag" in scalars
            assert scalars["test_tag"] == 1.5
            assert csv_path.exists()


def test_merge_to_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        metrics_path = Path(tmp) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"existing": "data"}, f)

        scalars = {"new": 2.0}

        with patch("ztb.scripts.tb_scrape.Path") as mock_path:
            mock_metrics = MagicMock()
            mock_metrics.exists.return_value = True
            mock_path.return_value = mock_metrics

            # Mock open
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file
                mock_file.read.return_value = '{"existing": "data"}'

                merge_to_metrics("corr123", scalars)

                # Check write was called
                assert mock_open.call_count >= 2
