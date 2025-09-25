# test_utils.py

import pytest
import torch
import torch.nn.functional as F
import math
from src.utils import AttentionHead, MultiHeadAttention

@pytest.mark.order(1)
def test_attention_head_output_shape():
    """Test the output shape of AttentionHead's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = d_q = d_v = 8

    # Create random input tensors
    x_q = torch.randn(batch_size, seq_len, d_model)
    x_k = torch.randn(batch_size, seq_len, d_model)
    x_v = torch.randn(batch_size, seq_len, d_model)

    # Initialize AttentionHead
    attention_head = AttentionHead(d_model, d_k, d_q, d_v)

    # Forward pass
    output = attention_head(x_q, x_k, x_v)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_v), (
        f"Expected output shape {(batch_size, seq_len, d_v)}, got {output.shape}"
    )

@pytest.mark.order(2)
def test_attention_forward():
    """Test the scaled_dot_product_attention method of AttentionHead with and without mask."""
    import math

    # Create fixed input tensors
    q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]]])  # Shape: (1, 3, 4)
    k = q.clone()
    v = q.clone()  # Using q as v for simplicity

    # Initialize AttentionHead with identity projections
    attention_head = AttentionHead(d_model=4, d_k=4, d_q=4, d_v=4)
    attention_head.wq = torch.nn.Identity()
    attention_head.wk = torch.nn.Identity()
    attention_head.wv = torch.nn.Identity()

    # Without mask
    output_no_mask = attention_head(q, k, v)  # Using the forward method

    # Expected output without mask
    expected_output_no_mask = torch.tensor([[[0.4519, 0.2741, 0.2741, 0.0000],
                                            [0.2741, 0.4519, 0.2741, 0.0000],
                                            [0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_no_mask, expected_output_no_mask, atol=1e-4), "Output without mask does not match expected output"

    # Now test with mask
    # Create a causal mask that masks out future tokens
    # For position i, tokens at positions > i are masked
    # Mask shape: (1, 3, 3)
    mask = torch.tensor([[[1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 1]]])  # Causal mask

    # Use the scaled_dot_product_attention method with mask
    output_with_mask = attention_head(q, k, v, mask)

    # Expected output with mask
    expected_output_with_mask = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000],
                                                [0.3775, 0.6225, 0.0000, 0.0000],
                                                [0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_with_mask, expected_output_with_mask, atol=1e-4), "Output with mask does not match expected output"


@pytest.mark.order(3)
def test_multihead_attention_output_shape():
    """Test the output shape of MultiHeadAttention's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    # Create random input tensors
    x_q = torch.randn(batch_size, seq_len, d_model)
    x_k = torch.randn(batch_size, seq_len, d_model)
    x_v = torch.randn(batch_size, seq_len, d_model)

    # Initialize MultiHeadAttention
    multihead_attention = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    output = multihead_attention(x_q, x_k, x_v)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

@pytest.mark.order(4)
def test_multihead_attention_forward():
    """Test the forward method of MultiHeadAttention."""
    # Initialize parameters
    d_model = 4
    num_heads = 2

    # Create fixed input tensors
    x_q = torch.tensor([[[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]]])  # Shape: (1, 3, 4)
    x_k = x_q.clone()
    x_v = x_q.clone()

    # Initialize MultiHeadAttention with identity projections
    multihead_attention = MultiHeadAttention(d_model, num_heads)
    for head in multihead_attention.heads:
        head.wq = torch.nn.Identity()
        head.wk = torch.nn.Identity()
        head.wv = torch.nn.Identity()
    multihead_attention.output_linear = torch.nn.Identity()

    # Forward pass
    output = multihead_attention(x_q, x_k, x_v)

    # Expected outputs
    expected_output = torch.tensor([[[0.4519, 0.2741, 0.2741, 0.0000, 0.4519, 0.2741, 0.2741, 0.0000],
                                    [0.2741, 0.4519, 0.2741, 0.0000, 0.2741, 0.4519, 0.2741, 0.0000],
                                    [0.2741, 0.2741, 0.4519, 0.0000, 0.2741, 0.2741, 0.4519, 0.0000]]])

    # Since output_linear is identity, the final output is the concatenated outputs
    assert torch.allclose(output, expected_output, atol=1e4), "Output does not match expected output"

    # Create a causal mask that masks out future tokens
    mask = torch.tensor([[[1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 1]]])  # Shape: (1, 3, 3)

    # Forward pass with mask
    output_with_mask = multihead_attention(x_q, x_k, x_v, mask)

    # Expected outputs with mask
    expected_output_with_mask = torch.tensor([[[1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                                                [0.3775, 0.6225, 0.0000, 0.0000, 0.3775, 0.6225, 0.0000, 0.0000],
                                                [0.2741, 0.2741, 0.4519, 0.0000, 0.2741, 0.2741, 0.4519, 0.0000]]])

    # Check if the outputs match expected outputs
    assert torch.allclose(output_with_mask, expected_output_with_mask, atol=1e-4), "Output with mask does not match expected output"
