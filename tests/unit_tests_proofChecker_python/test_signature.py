"""Tests for Signature class in signature module."""

import pytest
from proofChecker_python_serial.signature import Signature


class TestSignatureCreation:
    """Test Signature class creation and validation."""

    def test_valid_signature_creation(self):
        """Test creation with valid signature strings."""
        valid_signatures = [
            "1F-2a-3b",
            "123G-456c_789d-1e_2f",
            "1A-2B-3C 4D-5E-6F",
            "42x-1y_2z-99w 88a-77b-66c",
        ]

        for sig_str in valid_signatures:
            signature = Signature(sig_str)
            assert signature.signature == sig_str

    def test_invalid_signature_creation(self):
        """Test that invalid signatures raise ValueError."""
        invalid_signatures = [
            "",  # empty string
            "1a-2b",  # incomplete word (missing third part)
            "abc-def-ghi",  # doesn't start with digits
            "123-456-789",  # only digits
            "1a-2b-3c  4d-5e-6f",  # double space
            "1a-2b-3c-4d-5e-6f",  # too many hyphens
            "1a_2b_3c",  # underscores instead of hyphens
            "1a-2b-3c 4d-5e",  # second word incomplete
        ]

        for sig_str in invalid_signatures:
            with pytest.raises(ValueError, match="Signature must follow the pattern"):
                Signature(sig_str)

    def test_non_string_signature(self):
        """Test that non-string signatures raise ValueError."""
        invalid_types = [123, None, [], {}, 45.67]

        for invalid_sig in invalid_types:
            with pytest.raises(ValueError, match="Signature must be a string"):
                Signature(invalid_sig)


class TestSignatureProperties:
    """Test Signature class properties."""

    @pytest.fixture
    def simple_signature(self):
        """Simple signature with one edge."""
        return Signature("1F-2a-3b")

    @pytest.fixture
    def complex_signature(self):
        """Complex signature with multiple edges and groups."""
        return Signature("1F-2a_3b-4c 5G-6d-7e_8f 9H-10i_11j_12k-13l")

    def test_signature_elements(self, simple_signature, complex_signature):
        """Test signature_elements property."""
        # Simple signature - no spaces, hyphens, or underscores to split on in this context
        # This property seems to split the signature by spaces, hyphens, or underscores
        simple_elements = simple_signature.signature_elements
        assert len(simple_elements) > 0

        # Complex signature should have more elements
        complex_elements = complex_signature.signature_elements
        assert len(complex_elements) > len(simple_elements)

    def test_signature_length(self, simple_signature, complex_signature):
        """Test signature_length property."""
        assert simple_signature.signature_length > 0
        assert complex_signature.signature_length > simple_signature.signature_length

    def test_edges_property(self, simple_signature, complex_signature):
        """Test edges property returns correct list of edges."""
        # Simple signature has one edge
        simple_edges = simple_signature.edges
        assert simple_edges == ["1F-2a-3b"]
        assert len(simple_edges) == 1

        # Complex signature has three edges
        complex_edges = complex_signature.edges
        expected_edges = ["1F-2a_3b-4c", "5G-6d-7e_8f", "9H-10i_11j_12k-13l"]
        assert complex_edges == expected_edges
        assert len(complex_edges) == 3

    def test_num_edges_property(self, simple_signature, complex_signature):
        """Test num_edges property."""
        assert simple_signature.num_edges == 1
        assert complex_signature.num_edges == 3

    def test_node_signatures_property(self, simple_signature):
        """Test node_signatures property extracts nodes correctly."""
        # For "1F-2a-3b", sources="2a", targets="3b"
        # So nodes should be ["2a", "3b"]
        nodes = simple_signature.node_signatures
        assert set(nodes) == {"2a", "3b"}
        assert len(nodes) == 2

    def test_node_signatures_complex(self):
        """Test node_signatures with complex groups."""
        # For "1F-2a_3b-4c_5d", sources="2a_3b", targets="4c_5d"
        # So nodes should be ["2a", "3b", "4c", "5d"]
        signature = Signature("1F-2a_3b-4c_5d")
        nodes = signature.node_signatures
        assert set(nodes) == {"2a", "3b", "4c", "5d"}
        assert len(nodes) == 4

    def test_node_signatures_multiple_edges(self):
        """Test node_signatures with multiple edges."""
        # "1F-2a-3b 4G-5c-6d" should have nodes ["2a", "3b", "5c", "6d"]
        signature = Signature("1F-2a-3b 4G-5c-6d")
        nodes = signature.node_signatures
        assert set(nodes) == {"2a", "3b", "5c", "6d"}
        assert len(nodes) == 4

    def test_node_signatures_with_overlapping_nodes(self):
        """Test node_signatures when same nodes appear in multiple edges."""
        # Both edges share the same target node "3b"
        signature = Signature("1F-2a-3b 4G-5c-3b")
        nodes = signature.node_signatures
        # Should deduplicate the shared node
        assert set(nodes) == {"2a", "3b", "5c"}
        assert len(nodes) == 3


class TestSignatureEdgeCases:
    """Test edge cases and special scenarios."""

    def test_signature_with_complex_groups(self):
        """Test signature with complex multi-unit groups."""
        signature = Signature("1F-2a_3b_4c-5d_6e")

        edges = signature.edges
        assert len(edges) == 1
        assert edges[0] == "1F-2a_3b_4c-5d_6e"

        nodes = signature.node_signatures
        expected_nodes = {"2a", "3b", "4c", "5d", "6e"}
        assert set(nodes) == expected_nodes

    def test_signature_properties_consistency(self):
        """Test that signature properties are consistent with each other."""
        signature = Signature("1F-2a-3b 4G-5c_6d-7e")

        # num_edges should equal length of edges list
        assert signature.num_edges == len(signature.edges)

        # signature_length should equal length of signature_elements
        assert signature.signature_length == len(signature.signature_elements)

        # All edges should be valid words according to our regex
        from proofChecker_python_serial.regex_check import matches_word_pattern

        for edge in signature.edges:
            assert matches_word_pattern(edge)

    def test_signature_immutability(self):
        """Test that signature properties don't modify the original signature."""
        original_sig = "1F-2a-3b 4G-5c-6d"
        signature = Signature(original_sig)

        # Access all properties
        _ = signature.signature_elements
        _ = signature.edges
        _ = signature.node_signatures

        # Original signature should be unchanged
        assert signature.signature == original_sig


class TestSignatureValidationMessages:
    """Test that validation error messages are helpful."""

    def test_pattern_validation_error_message(self):
        """Test that pattern validation provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            Signature("invalid-signature")

        error_msg = str(exc_info.value)
        assert "Signature must follow the pattern" in error_msg
        assert "word (space word)*" in error_msg
        assert "unit-group-group" in error_msg

    def test_type_validation_error_message(self):
        """Test that type validation provides helpful error message."""
        with pytest.raises(ValueError) as exc_info:
            Signature(123)

        error_msg = str(exc_info.value)
        assert "Signature must be a string" in error_msg


class TestSignatureRealWorldExamples:
    """Test signature with realistic examples."""

    def test_hypergraph_signature_example(self):
        """Test with a realistic hypergraph signature."""
        # Example from a proof checker hypergraph
        signature = Signature("0F-1a_2b-3c 1G-3c-4d 2H-4d_5e-6f")

        assert signature.num_edges == 3
        assert len(signature.node_signatures) == 6  # a,b,c,d,e,f

        expected_nodes = {"1a", "2b", "3c", "4d", "5e", "6f"}
        assert set(signature.node_signatures) == expected_nodes

    def test_complex_proof_signature(self):
        """Test with a more complex proof signature."""
        signature = Signature("0F-1x_2y_3z-4w 1G-4w-5a_6b 2H-5a_6b_7c-8d_9e")

        assert signature.num_edges == 3

        # Should extract all unique nodes
        expected_nodes = {"1x", "2y", "3z", "4w", "5a", "6b", "7c", "8d", "9e"}
        assert set(signature.node_signatures) == expected_nodes
