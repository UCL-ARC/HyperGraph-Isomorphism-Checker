"""Tests for random_graph_generator.py."""

import json
import pytest
from IsomorphismChecker_python_serial.random_graph_generator import (
    generate_random_hypergraph,
    graph_to_json_serializable,
)


def test_invalid_input_output_sum():
    """Test that ValueError is raised when num_inputs + num_outputs > num_nodes."""
    with pytest.raises(
        ValueError,
        match="The sum of num_inputs and num_outputs cannot exceed num_nodes.",
    ):
        generate_random_hypergraph(
            num_nodes=5, num_edges=10, num_inputs=3, num_outputs=3
        )


def test_valid_graph_generation():
    """Test that a valid graph is generated when inputs are correct."""
    graph = generate_random_hypergraph(
        num_nodes=6, num_edges=10, num_inputs=2, num_outputs=2, seed=42
    )
    assert graph is not None
    assert len(graph.nodes) == 16  # 6 wires + 10 boxes
    input_wires = [n for n, d in graph.nodes(data=True) if d.get("input")]
    output_wires = [n for n, d in graph.nodes(data=True) if d.get("output")]
    assert len(input_wires) == 2
    assert len(output_wires) == 2


def test_json_serialization():
    """Test that the generated graph can be serialized to JSON."""

    graph = generate_random_hypergraph(
        num_nodes=6, num_edges=10, num_inputs=2, num_outputs=2, seed=42
    )
    json_serializable_graph = graph_to_json_serializable(graph, file_name=None)

    assert isinstance(json_serializable_graph, dict)
    assert "nodes" in json_serializable_graph
    assert "hyperedges" in json_serializable_graph
    assert "Inputs" in json_serializable_graph
    assert "Outputs" in json_serializable_graph


def test_json_file_creation(tmp_path):
    """Test that the function creates a JSON file when file_name is provided."""

    graph = generate_random_hypergraph(
        num_nodes=6, num_edges=10, num_inputs=2, num_outputs=2, seed=42
    )

    # Use pytest's tmp_path fixture to create a temporary directory
    test_file_name = "test_hypergraph"
    json_serializable_graph = graph_to_json_serializable(
        graph, file_name=test_file_name, directory=str(tmp_path)
    )

    # Check that the file was created
    expected_file_path = tmp_path / f"{test_file_name}.json"
    assert expected_file_path.exists()

    # Check that the file contains valid JSON
    with open(expected_file_path, "r") as f:
        file_data = json.load(f)

    # Verify the file contents match the returned dictionary
    assert file_data == json_serializable_graph
    assert "nodes" in file_data
    assert "hyperedges" in file_data
    assert "Inputs" in file_data
    assert "Outputs" in file_data
