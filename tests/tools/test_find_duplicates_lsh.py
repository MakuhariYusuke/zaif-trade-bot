from tools.find_duplicates import find_similar


def test_lsh_detects_identical():
    normalized = {
        "h1": "def foo():\n    return 1",
        "h2": "def foo():\n    return 1",
        "h3": "def bar():\n    x = 1\n    return x",
    }
    res = find_similar(normalized, threshold=0.99, use_lsh=True, lsh_perm=16, lsh_bands=4, shingle_k=2)
    assert any(set([r[0], r[1]]) == {"h1", "h2"} for r in res)


def test_lsh_respects_max_comparisons():
    # generate many items to create many candidates, but set max_comparisons very low
    normalized = {f"h{i}": f"def f{i}():\n    line\n" * (i + 1) for i in range(120)}
    # allow LSH but limit comparisons to 1
    res = find_similar(normalized, threshold=0.9, use_lsh=True, lsh_perm=32, lsh_bands=8, shingle_k=3, max_comparisons=1)
    # function should complete and return list (possibly empty)
    assert isinstance(res, list)
