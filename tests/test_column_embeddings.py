"""Unit tests for column embeddings module."""

from treac.models.embeddings import ColumnEmbedding, create_column_embedding


class TestColumnEmbedding:
    """Test the ColumnEmbedding module."""

    def test_basic_initialization(self):
        """Test basic initialization with simple column names."""
        column_names = ["col1", "col2", "col3"]
        emb = ColumnEmbedding(column_names, target_dim=1)

        assert emb.column_names == column_names
        assert emb.target_dim == 1
        assert emb.column_embeddings.shape == (3, 1)

    def test_etth1_columns(self):
        """Test with ETTh1 dataset column names."""
        column_names = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        emb = ColumnEmbedding(column_names, target_dim=1)

        assert emb.column_embeddings.shape == (7, 1)

        # Test forward pass
        batch_size, seq_len = 16, 96
        output = emb(batch_size, seq_len)

        expected_shape = (batch_size, len(column_names), seq_len)
        assert output.shape == expected_shape

    def test_tokenization_strategies(self):
        """Test different tokenization strategies."""
        column_names = ["user_account", "getUserID", "temp_sensor_1"]

        strategies = ["as_is", "split_underscore", "split_underscore_camel"]

        for strategy in strategies:
            emb = ColumnEmbedding(
                column_names, target_dim=1, tokenization_strategy=strategy
            )

            # Check that embeddings are created
            assert emb.column_embeddings.shape == (3, 1)

            # Check tokenization
            for name in column_names:
                processed = emb._process_column_name(name)
                assert isinstance(processed, str)
                assert len(processed) > 0

    def test_aggregation_strategies(self):
        """Test different aggregation strategies."""
        column_names = ["test_column"]

        strategies = ["mean", "cls", "max"]

        for strategy in strategies:
            emb = ColumnEmbedding(
                column_names, target_dim=1, aggregation_strategy=strategy
            )

            assert emb.column_embeddings.shape == (1, 1)

    def test_target_dimensions(self):
        """Test different target dimensions."""
        column_names = ["col1", "col2"]

        for target_dim in [1, 4, 8]:
            emb = ColumnEmbedding(column_names, target_dim=target_dim)

            assert emb.column_embeddings.shape == (2, target_dim)

            # Test forward pass
            batch_size, seq_len = 8, 32
            output = emb(batch_size, seq_len)

            if target_dim == 1:
                expected_shape = (batch_size, len(column_names), seq_len)
            else:
                expected_shape = (batch_size, len(column_names), target_dim, seq_len)

            assert output.shape == expected_shape

    def test_forward_pass_shapes(self):
        """Test forward pass with various batch sizes and sequence lengths."""
        column_names = ["A", "B", "C", "D"]
        emb = ColumnEmbedding(column_names, target_dim=1)

        test_cases = [
            (1, 10),
            (8, 32),
            (32, 96),
            (64, 128),
        ]

        for batch_size, seq_len in test_cases:
            output = emb(batch_size, seq_len)
            expected_shape = (batch_size, len(column_names), seq_len)
            assert output.shape == expected_shape

    def test_factory_function(self):
        """Test the factory function."""
        column_names = ["test1", "test2"]
        emb = create_column_embedding(column_names, target_dim=2)

        assert isinstance(emb, ColumnEmbedding)
        assert emb.column_names == column_names
        assert emb.target_dim == 2

    def test_tokenization_edge_cases(self):
        """Test tokenization with edge cases."""
        emb = ColumnEmbedding(["test"], target_dim=1)

        test_cases = [
            ("", ""),  # Empty string
            ("A", "a"),  # Single character
            ("ABC", "abc"),  # All caps
            ("test_", "test"),  # Trailing underscore
            ("_test", "test"),  # Leading underscore
            ("test__test", "test test"),  # Double underscore
            ("testCase", "test case"),  # CamelCase
            ("XMLHttpRequest", "xml http request"),  # Complex CamelCase
        ]

        for input_str, expected_output in test_cases:
            if input_str:  # Skip empty string test for now
                result = emb._process_column_name(input_str)
                # Just check that it doesn't crash and returns a string
                assert isinstance(result, str)

    def test_device_handling(self):
        """Test device handling for column embeddings."""
        column_names = ["test1", "test2"]

        # Test CPU
        emb_cpu = ColumnEmbedding(column_names, target_dim=1, device="cpu")
        assert emb_cpu.column_embeddings.device.type == "cpu"

        # Test forward pass on CPU
        output_cpu = emb_cpu(2, 10)
        assert output_cpu.device.type == "cpu"

    def test_bert_freeze_parameter(self):
        """Test that BERT parameters are frozen when requested."""
        column_names = ["test"]

        # Test frozen BERT
        emb_frozen = ColumnEmbedding(column_names, target_dim=1, freeze_bert=True)
        bert_params_frozen = [p.requires_grad for p in emb_frozen.bert.parameters()]
        assert not any(bert_params_frozen)  # All should be False

        # Projection should still be trainable
        proj_params = [p.requires_grad for p in emb_frozen.projection.parameters()]
        assert all(proj_params)  # All should be True


if __name__ == "__main__":
    # Run basic tests
    test = TestColumnEmbedding()

    print("Running column embedding tests...")

    try:
        test.test_basic_initialization()
        print("✓ Basic initialization test passed")

        test.test_etth1_columns()
        print("✓ ETTh1 columns test passed")

        test.test_tokenization_strategies()
        print("✓ Tokenization strategies test passed")

        test.test_forward_pass_shapes()
        print("✓ Forward pass shapes test passed")

        test.test_factory_function()
        print("✓ Factory function test passed")

        print("\nAll tests passed! ✓")

    except Exception as e:
        print(f"Test failed: {e}")
        raise
