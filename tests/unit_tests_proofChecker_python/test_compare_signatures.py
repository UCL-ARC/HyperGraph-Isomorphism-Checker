"""Tests for CompareSignatures class in compare_signatures module."""

from proofChecker_python_serial.signature import Signature
from proofChecker_python_serial.compare_signatures import CompareSignatures


class TestCompareSignaturesCreation:
    """Test CompareSignatures class creation and basic functionality."""

    def test_identical_signatures(self):
        """Test comparison of identical signatures."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("1F-2a-3b")

        comparison = CompareSignatures(sig1, sig2)

        assert comparison.are_equal is True
        assert comparison.signature1 == sig1
        assert comparison.signature2 == sig2
        assert comparison.details == "Signatures are equal."

    def test_different_signatures(self):
        """Test comparison of different signatures."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("4G-5c-6d")

        comparison = CompareSignatures(sig1, sig2)

        assert comparison.are_equal is True  # Same structure, different labels
        assert comparison.signature1 == sig1
        assert comparison.signature2 == sig2


class TestEqualLengths:
    """Test are_equal_lengths method."""

    def test_equal_lengths(self):
        """Test signatures with equal lengths."""
        sig1 = Signature("1F-2a-3b")  # Length = 9
        sig2 = Signature("4G-5c-6d")  # Length = 9

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_lengths() is True

    def test_unequal_lengths(self):
        """Test signatures with unequal lengths."""
        sig1 = Signature("1F-2a-3b")  # Shorter signature
        sig2 = Signature("1F-2a_3b-4c_5d")  # Longer signature

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_lengths() is False

    def test_multi_edge_length_comparison(self):
        """Test length comparison with multi-edge signatures."""
        sig1 = Signature("1F-2a-3b 4G-5c-6d")
        sig2 = Signature("7H-8e-9f 10I-11g-12h")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_lengths() is True


class TestEqualNumEdges:
    """Test are_equal_num_edges method."""

    def test_equal_single_edges(self):
        """Test signatures with equal number of single edges."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("4G-5c-6d")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_edges() is True

    def test_equal_multiple_edges(self):
        """Test signatures with equal number of multiple edges."""
        sig1 = Signature("1F-2a-3b 4G-5c-6d")
        sig2 = Signature("7H-8e-9f 10I-11g-12h")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_edges() is True

    def test_unequal_edge_counts(self):
        """Test signatures with different number of edges."""
        sig1 = Signature("1F-2a-3b")  # 1 edge
        sig2 = Signature("4G-5c-6d 7H-8e-9f")  # 2 edges

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_edges() is False

    def test_complex_edge_count_comparison(self):
        """Test edge count comparison with complex signatures."""
        sig1 = Signature("1F-2a_3b-4c 5G-6d-7e_8f 9H-10i-11j")  # 3 edges
        sig2 = Signature("12I-13k-14l 15J-16m_17n-18o 19K-20p-21q")  # 3 edges

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_edges() is True


class TestEqualSourcesTargets:
    """Test are_equal_num_sources_targets method."""

    def test_equal_simple_sources_targets(self):
        """Test signatures with equal sources and targets structure."""
        sig1 = Signature("1F-2a-3b")  # 1 source, 1 target = 2 total
        sig2 = Signature("4G-5c-6d")  # 1 source, 1 target = 2 total

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_targets() is True

    def test_equal_complex_sources_targets(self):
        """Test signatures with equal complex sources and targets."""
        sig1 = Signature("1F-2a_3b-4c_5d")  # 2 sources, 2 targets = 4 total
        sig2 = Signature("6G-7e_8f-9g_10h")  # 2 sources, 2 targets = 4 total

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_targets() is True

    def test_equal_multi_edge_sources_targets(self):
        """Test multi-edge signatures with equal sources/targets distribution."""
        # Edge 1: 1+1=2, Edge 2: 2+1=3 → sorted: [2, 3]
        sig1 = Signature("1F-2a-3b 4G-5c_6d-7e")
        # Edge 1: 2+1=3, Edge 2: 1+1=2 → sorted: [2, 3]
        sig2 = Signature("8H-9f_10g-11h 12I-13i-14j")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_targets() is True

    def test_unequal_sources_targets(self):
        """Test signatures with different sources and targets structure."""
        sig1 = Signature("1F-2a-3b")  # 1+1 = 2 total
        sig2 = Signature("4G-5c_6d-7e")  # 2+1 = 3 total

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_targets() is False


class TestEqualSourcesAllEdges:
    """Test are_equal_num_sources_all_edges method."""

    def test_equal_sources_single_edge(self):
        """Test signatures with equal sources for single edges."""
        sig1 = Signature("1F-2a_3b-4c")  # 2 sources
        sig2 = Signature("5G-6d_7e-8f")  # 2 sources

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_all_edges() is True

    def test_equal_sources_multiple_edges(self):
        """Test signatures with equal sources distribution across edges."""
        # Sources: [1, 2] → sorted: [1, 2]
        sig1 = Signature("1F-2a-3b 4G-5c_6d-7e")
        # Sources: [2, 1] → sorted: [1, 2]
        sig2 = Signature("8H-9f_10g-11h 12I-13i-14j")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_all_edges() is True

    def test_unequal_sources_distribution(self):
        """Test signatures with different sources distribution."""
        # Sources: [1, 1] → sorted: [1, 1]
        sig1 = Signature("1F-2a-3b 4G-5c-6d")
        # Sources: [1, 2] → sorted: [1, 2]
        sig2 = Signature("7H-8e-9f 10I-11g_12h-13i")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_all_edges() is False

    def test_complex_sources_comparison(self):
        """Test complex sources comparison with varied edge structures."""
        # Sources: [3, 1, 2] → sorted: [1, 2, 3]
        sig1 = Signature("1F-2a_3b_4c-5d 6G-7e-8f 9H-10g_11h-12i")
        # Sources: [2, 3, 1] → sorted: [1, 2, 3]
        sig2 = Signature("13I-14j_15k-16l 17J-18m_19n_20o-21p 22K-23q-24r")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_sources_all_edges() is True


class TestEqualTargetsAllEdges:
    """Test are_equal_num_targets_all_edges method."""

    def test_equal_targets_single_edge(self):
        """Test signatures with equal targets for single edges."""
        sig1 = Signature("1F-2a-3b_4c")  # 2 targets
        sig2 = Signature("5G-6d-7e_8f")  # 2 targets

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_targets_all_edges() is True

    def test_equal_targets_multiple_edges(self):
        """Test signatures with equal targets distribution across edges."""
        # Targets: [1, 2] → sorted: [1, 2]
        sig1 = Signature("1F-2a-3b 4G-5c-6d_7e")
        # Targets: [2, 1] → sorted: [1, 2]
        sig2 = Signature("8H-9f-10g_11h 12I-13i-14j")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_targets_all_edges() is True

    def test_unequal_targets_distribution(self):
        """Test signatures with different targets distribution."""
        # Targets: [1, 1] → sorted: [1, 1]
        sig1 = Signature("1F-2a-3b 4G-5c-6d")
        # Targets: [1, 3] → sorted: [1, 3]
        sig2 = Signature("7H-8e-9f 10I-11g-12h_13i_14j")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_targets_all_edges() is False

    def test_complex_targets_comparison(self):
        """Test complex targets comparison with varied edge structures."""
        # Targets: [1, 3, 2] → sorted: [1, 2, 3]
        sig1 = Signature("1F-2a-3b 4G-5c-6d_7e_8f 9H-10g-11h_12i")
        # Targets: [2, 1, 3] → sorted: [1, 2, 3]
        sig2 = Signature("13I-14j-15k_16l 17J-18m-19n 20K-21o-22p_23q_24r")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal_num_targets_all_edges() is True


class TestPostInitComparison:
    """Test __post_init__ method and comparison logic."""

    def test_identical_signatures_post_init(self):
        """Test post_init with identical signatures."""
        sig1 = Signature("1F-2a-3b 4G-5c_6d-7e")
        sig2 = Signature("1F-2a-3b 4G-5c_6d-7e")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is True

    def test_different_lengths_post_init(self):
        """Test post_init fails on different lengths."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("1F-2a_3b-4c_5d")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different lengths" in comparison.details
        assert f"{sig1.signature_length}" in comparison.details
        assert f"{sig2.signature_length}" in comparison.details

    def test_different_edge_counts_post_init(self):
        """Test post_init fails on different edge counts."""
        sig1 = Signature("1F-2a-3b")  # 1 edge
        sig2 = Signature("1F-2a-3b 4G-5c-6d")  # 2 edges

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of edges" in comparison.details
        assert "1 vs 2" in comparison.details

    def test_different_sources_targets_post_init(self):
        """Test post_init fails on different sources/targets structure."""
        sig1 = Signature("1F-2a-3b")  # 1+1=2 total
        sig2 = Signature("1F-2a_3b-4c")  # 2+1=3 total

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of sources and targets" in comparison.details

    def test_different_sources_post_init(self):
        """Test post_init fails on different sources distribution."""
        sig1 = Signature("1F-2a-3b 4G-5c-6d")  # Sources: [1, 1]
        sig2 = Signature("1F-2a_3b-4c 5G-6d-7e")  # Sources: [2, 1]

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of sources for edges" in comparison.details

    def test_different_targets_post_init(self):
        """Test post_init fails on different targets distribution."""
        sig1 = Signature("1F-2a-3b 4G-5c-6d")  # Targets: [1, 1]
        sig2 = Signature("1F-2a-3b_4c 5G-6d-7e")  # Targets: [2, 1]

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of targets for edges" in comparison.details

    def test_structurally_equivalent_signatures(self):
        """Test signatures that are structurally equivalent but have different labels."""
        # Same structure: both have 2 edges with [1,2] sources and [1,1] targets
        sig1 = Signature("1F-2a-3b 4G-5c_6d-7e")
        sig2 = Signature("8H-9x-10y 11I-12z_13w-14v")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is True


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_complex_edge_comparison(self):
        """Test comparison of single complex edges."""
        sig1 = Signature("1F-2a_3b_4c-5d_6e_7f")  # 3 sources, 3 targets
        sig2 = Signature("8G-9x_10y_11z-12w_13v_14u")  # 3 sources, 3 targets

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is True

    def test_many_edges_comparison(self):
        """Test comparison with many edges."""
        # 4 edges with varied structures
        sig1 = Signature("1F-2a-3b 4G-5c_6d-7e 8H-9f-10g_11h 12I-13i_14j-15k")
        sig2 = Signature("16J-17l-18m 19K-20n_21o-22p 23L-24q-25r_26s 27M-28t_29u-30v")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is True

    def test_comparison_properties_consistency(self):
        """Test that comparison properties are consistent with each other."""
        sig1 = Signature("1F-2a_3b-4c_5d 6G-7e-8f")
        sig2 = Signature("9H-10g_11h-12i_13j 14I-15k-16l")

        comparison = CompareSignatures(sig1, sig2)

        # If overall comparison is true, individual comparisons should also be true
        if comparison.are_equal:
            assert comparison.are_equal_lengths()
            assert comparison.are_equal_num_edges()
            assert comparison.are_equal_num_sources_targets()
            assert comparison.are_equal_num_sources_all_edges()
            assert comparison.are_equal_num_targets_all_edges()


class TestComparisonErrorMessages:
    """Test that comparison error messages are informative."""

    def test_length_error_message_format(self):
        """Test format of length comparison error message."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("1F-2a_3b_4c-5d_6e")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "Signatures have different lengths:" in comparison.details
        assert str(sig1.signature_length) in comparison.details
        assert str(sig2.signature_length) in comparison.details

    def test_edge_count_error_message_format(self):
        """Test format of edge count error message."""
        sig1 = Signature("1F-2a-3b")
        sig2 = Signature("1F-2a-3b 4G-5c-6d")

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of edges:" in comparison.details
        assert f"{sig1.num_edges} vs {sig2.num_edges}" in comparison.details

    def test_sources_targets_error_message_includes_arrays(self):
        """Test that sources/targets error message includes the actual arrays."""
        sig1 = Signature("1F-2a-3b")  # [2]
        sig2 = Signature("1F-2a_3b-4c")  # [3]

        comparison = CompareSignatures(sig1, sig2)
        assert comparison.are_equal is False
        assert "different number of sources and targets:" in comparison.details
        # Should include the actual arrays being compared
        assert "[2]" in comparison.details
        assert "[3]" in comparison.details
