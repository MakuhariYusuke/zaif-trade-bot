import pathlib
from tools.find_duplicates import analyze, find_similar


def test_analyze_small(tmp_path):
    # run a very small scan of the repo root with max_files to ensure it completes
    root = pathlib.Path(".").resolve()
    groups, normalized = analyze(root, (".py",), max_files=5, max_bytes=1000000, progress_every=1)
    assert isinstance(groups, dict)
    assert isinstance(normalized, dict)


def test_find_similar_limits():
    # small normalized dict with two similar entries
    normalized = {"h1": "def foo():\n    return 1", "h2": "def foo():\n    return 1"}
    similar = find_similar(normalized, threshold=0.9, max_comparisons=10)
    assert len(similar) >= 1


def test_min_lines_skips(tmp_path):
    # create two files, one with a 1-line function and one with a multi-line function
    f1 = tmp_path / "a.py"
    f1.write_text("def tiny(): pass\n")
    f2 = tmp_path / "b.py"
    f2.write_text("def longer():\n    x = 1\n    y = 2\n    return x + y\n")

    groups, normalized = analyze(tmp_path, (".py",), max_files=10, max_bytes=1000000, progress_every=1, min_lines=2)
    # tiny should be skipped, longer should be present
    names = [o.name for occs in groups.values() for o in occs]
    assert 'longer' in names
    assert 'tiny' not in names


def test_bucket_behavior_quick():
    # create contrived normalized codes with varying lengths to exercise bucketing
    normalized = {}
    for i in range(50):
        body = "\n".join([f"line{j}" for j in range(i + 1)])
        normalized[f"h{i}"] = "def f():\n" + body

    similar = find_similar(normalized, threshold=0.95, max_comparisons=100000)
    # sanity: should finish and return a list of pairs (may be >0 for adjacent lengths)
    assert isinstance(similar, list)
    assert len(similar) >= 0
