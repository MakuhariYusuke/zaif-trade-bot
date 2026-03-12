from pathlib import Path

from ztb.io.text_io import read_last_lines


def test_read_last_lines_returns_trailing_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.log"
    lines = [f"line-{i}\n" for i in range(1, 201)]
    path.write_text("".join(lines), encoding="utf-8")

    result = read_last_lines(path, count=3)

    assert result == ["line-198\n", "line-199\n", "line-200\n"]


def test_read_last_lines_handles_count_larger_than_file(tmp_path: Path) -> None:
    path = tmp_path / "small.log"
    path.write_text("a\nb\n", encoding="utf-8")

    result = read_last_lines(path, count=10)

    assert result == ["a\n", "b\n"]


def test_read_last_lines_returns_empty_when_count_is_non_positive(tmp_path: Path) -> None:
    path = tmp_path / "small.log"
    path.write_text("a\nb\n", encoding="utf-8")

    assert read_last_lines(path, count=0) == []
