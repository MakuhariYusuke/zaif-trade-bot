"""
Tests for run_manifest module.
"""


from ztb.utils.run_manifest import (
    compare_manifests,
    compute_file_hash,
    generate_manifest,
    get_git_dirty_status,
    get_git_sha,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def test_get_git_sha():
    """Test getting git SHA."""
    sha = get_git_sha()

    # Should return either a 40-char hex string or "unknown"
    assert isinstance(sha, str)
    if sha != "unknown":
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


def test_get_git_dirty_status():
    """Test getting git dirty status."""
    is_dirty = get_git_dirty_status()

    # Should return a boolean
    assert isinstance(is_dirty, bool)


def test_compute_file_hash(tmp_path):
    """Test file hash computation."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")

    # Compute hash
    file_hash = compute_file_hash(test_file)

    # Should be a 64-character hex string (SHA256)
    assert len(file_hash) == 64
    assert all(c in "0123456789abcdef" for c in file_hash)

    # Same content should produce same hash
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_text("Hello, World!")
    file_hash2 = compute_file_hash(test_file2)

    assert file_hash == file_hash2

    # Same path with changed content should invalidate the stat-signature cache
    test_file.write_text("Hello, World! updated")
    file_hash_updated = compute_file_hash(test_file)

    assert file_hash_updated != file_hash

    # Different content should produce different hash
    test_file3 = tmp_path / "test3.txt"
    test_file3.write_text("Different content")
    file_hash3 = compute_file_hash(test_file3)

    assert file_hash != file_hash3


def test_generate_manifest_minimal(tmp_path):
    """Test generating manifest with minimal information."""
    config = {"learning_rate": 0.001, "batch_size": 32}
    feature_names = ["feature_a", "feature_b", "feature_c"]
    warmup = 220

    manifest = generate_manifest(
        model_dir=tmp_path,
        config=config,
        feature_names=feature_names,
        warmup=warmup,
    )

    # Check structure
    assert "version" in manifest
    assert "git" in manifest
    assert "hashes" in manifest
    assert "training" in manifest

    # Check git info
    assert "sha" in manifest["git"]
    assert "dirty" in manifest["git"]

    # Check hashes
    assert "schema" in manifest["hashes"]
    assert "scaler" in manifest["hashes"]
    assert "config_fingerprint" in manifest["hashes"]

    # Check training info
    assert manifest["training"]["config"] == config
    assert manifest["training"]["feature_names"] == feature_names
    assert manifest["training"]["warmup"] == warmup
    assert manifest["training"]["n_features"] == 3


def test_generate_manifest_with_files(tmp_path):
    """Test generating manifest with existing schema and scaler files."""
    # Create schema file
    schema_file = tmp_path / "feature_schema.json"
    schema_file.write_text('{"features": []}')

    # Create scaler file
    scaler_file = tmp_path / "normalization_stats.pkl"
    scaler_file.write_bytes(b"fake scaler data")

    config = {"learning_rate": 0.001}
    feature_names = ["feature_a"]
    warmup = 220

    manifest = generate_manifest(
        model_dir=tmp_path,
        config=config,
        feature_names=feature_names,
        warmup=warmup,
    )

    # Hashes should not be "not_available"
    assert manifest["hashes"]["schema"] != "not_available"
    assert manifest["hashes"]["scaler"] != "not_available"

    # Should be valid SHA256 hashes
    assert len(manifest["hashes"]["schema"]) == 64
    assert len(manifest["hashes"]["scaler"]) == 64


def test_generate_manifest_with_additional_metadata(tmp_path):
    """Test generating manifest with additional metadata."""
    config = {"learning_rate": 0.001}
    feature_names = ["feature_a"]
    warmup = 220
    additional = {"experiment_id": "exp_123", "notes": "Test run"}

    manifest = generate_manifest(
        model_dir=tmp_path,
        config=config,
        feature_names=feature_names,
        warmup=warmup,
        additional_metadata=additional,
    )

    # Additional metadata should be included
    assert "additional" in manifest
    assert manifest["additional"]["experiment_id"] == "exp_123"
    assert manifest["additional"]["notes"] == "Test run"


def test_generate_manifest_with_inference_config(tmp_path):
    """Test generating manifest with inference configuration."""
    config = {"learning_rate": 0.001}
    feature_names = ["feature_a"]
    warmup = 220
    inference_config = {
        "temperature": 0.7,
        "tiebreaker_tau": 0.05,
        "enable_tiebreaker": True,
        "enable_advantage_tiebreaker": True,
        "enable_cost_gate": True,
        "cost_gate_lambda": 1.2,
        "deterministic": True,
    }

    manifest = generate_manifest(
        model_dir=tmp_path,
        config=config,
        feature_names=feature_names,
        warmup=warmup,
        inference_config=inference_config,
    )

    # Inference config should be included
    assert "inference" in manifest
    assert manifest["inference"]["temperature"] == 0.7
    assert manifest["inference"]["tiebreaker_tau"] == 0.05
    assert manifest["inference"]["enable_advantage_tiebreaker"] is True
    assert manifest["inference"]["cost_gate_lambda"] == 1.2


def test_save_and_load_manifest(tmp_path):
    """Test saving and loading manifest."""
    config = {"learning_rate": 0.001}
    feature_names = ["feature_a"]
    warmup = 220

    # Generate manifest
    manifest = generate_manifest(
        model_dir=tmp_path,
        config=config,
        feature_names=feature_names,
        warmup=warmup,
    )

    # Save manifest
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    # Check file exists
    assert manifest_path.exists()

    # Load manifest
    loaded_manifest = load_manifest(manifest_path)

    # Should be identical
    assert loaded_manifest == manifest


def test_validate_manifest_valid():
    """Test validating a valid manifest."""
    manifest = {
        "version": "1.0",
        "git": {"sha": "abc123", "dirty": False},
        "hashes": {
            "schema": "def456",
            "scaler": "ghi789",
            "config_fingerprint": "jkl012",
        },
        "training": {
            "config": {},
            "feature_names": ["a", "b"],
            "warmup": 220,
            "n_features": 2,
        },
    }

    is_valid, errors = validate_manifest(manifest)

    assert is_valid is True
    assert len(errors) == 0


def test_validate_manifest_missing_fields():
    """Test validating manifest with missing fields."""
    manifest = {
        "version": "1.0",
        # Missing git
        "hashes": {"schema": "abc"},  # Missing scaler and config_fingerprint
        # Missing training
    }

    is_valid, errors = validate_manifest(manifest)

    assert is_valid is False
    assert len(errors) > 0
    assert any("git" in err for err in errors)
    assert any("scaler" in err for err in errors)
    assert any("training" in err for err in errors)


def test_compare_manifests_identical():
    """Test comparing identical manifests."""
    manifest1 = {
        "git": {"sha": "abc123", "dirty": False},
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {
            "feature_names": ["a", "b"],
            "warmup": 220,
        },
    }

    manifest2 = manifest1.copy()

    are_compatible, differences = compare_manifests(manifest1, manifest2)

    assert are_compatible is True
    assert len(differences) == 0


def test_compare_manifests_different_schema():
    """Test comparing manifests with different schema."""
    manifest1 = {
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 220},
    }

    manifest2 = {
        "hashes": {"schema": "schema2", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 220},
    }

    are_compatible, differences = compare_manifests(manifest1, manifest2)

    assert are_compatible is False
    assert any("Schema" in diff for diff in differences)


def test_compare_manifests_different_features():
    """Test comparing manifests with different features."""
    manifest1 = {
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a", "b"], "warmup": 220},
    }

    manifest2 = {
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a", "c"], "warmup": 220},
    }

    are_compatible, differences = compare_manifests(manifest1, manifest2)

    assert are_compatible is False
    assert any("Feature" in diff for diff in differences)


def test_compare_manifests_different_warmup():
    """Test comparing manifests with different warmup."""
    manifest1 = {
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 220},
    }

    manifest2 = {
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 300},
    }

    are_compatible, differences = compare_manifests(manifest1, manifest2)

    assert are_compatible is False
    assert any("Warmup" in diff for diff in differences)


def test_compare_manifests_ignore_git():
    """Test comparing manifests with different git SHA (ignored by default)."""
    manifest1 = {
        "git": {"sha": "abc123"},
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 220},
    }

    manifest2 = {
        "git": {"sha": "def456"},
        "hashes": {"schema": "schema1", "scaler": "scaler1"},
        "training": {"feature_names": ["a"], "warmup": 220},
    }

    # With ignore_git=True (default)
    are_compatible, differences = compare_manifests(
        manifest1, manifest2, ignore_git=True
    )
    assert are_compatible is True

    # With ignore_git=False
    are_compatible, differences = compare_manifests(
        manifest1, manifest2, ignore_git=False
    )
    assert are_compatible is False
    assert any("Git" in diff for diff in differences)
