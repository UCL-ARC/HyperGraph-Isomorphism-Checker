"""Module for testing expected signatures of proof graphs."""

import json
import pytest
from pathlib import Path
from proofChecker_python_serial.hypergraph import create_hypergraph


def get_json_files():
    """Return a list of JSON file paths from the inputs directory."""
    test_data_dir = Path(__file__).parent / "../inputs"
    return list(test_data_dir.glob("*.json"))


def get_files_with_expected_signature():
    """Return a list of JSON file paths that contain an expected_signature field."""

    files = []
    for file in get_json_files():
        with open(file, "r") as f:
            data = json.load(f)
            if "expected_signature" in data:
                files.append(file)

    return files


@pytest.mark.parametrize("filepath", get_json_files(), ids=lambda p: p.name)
def test_hypergraph_creation(filepath: Path):
    """Test a single JSON file for hypergraph validity."""
    hypergraph = create_hypergraph(filepath)

    assert hypergraph is not None
    assert len(hypergraph.nodes) > 0
    assert len(hypergraph.edges) > 0


@pytest.mark.parametrize(
    "filepath", get_files_with_expected_signature(), ids=lambda p: p.name
)
def test_expected_signature(filepath: Path):
    """Test that the hypergraph's signature matches the expected signature from the JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    expected_signature = data.get("expected_signature")
    if expected_signature is None:
        pytest.skip(f"No expected_signature field in {filepath.name}")

    hypergraph = create_hypergraph(filepath)
    actual_signature = hypergraph.signature

    assert actual_signature == expected_signature, (
        f"Signature mismatch in {filepath.name}: "
        f"expected '{expected_signature}', got '{actual_signature}'"
    )
