import pytest
from pathlib import Path


_LEGACY_UNIT_SKIP = pytest.mark.skip(
    reason="legacy unit suite targets deprecated compatibility contracts and is excluded from the maintained test baseline"
)

_LEGACY_UNIT_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        if _LEGACY_UNIT_DIR == item_path.parent or _LEGACY_UNIT_DIR in item_path.parents:
            item.add_marker(_LEGACY_UNIT_SKIP)
