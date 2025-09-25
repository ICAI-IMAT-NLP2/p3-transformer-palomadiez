# test_encoder.py

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.encoder import TransformerEncoderLayer, TransformerEncoder

@pytest.mark.order(5)
def test_transformer_encoder_layer_output_shape():
    """Test the output shape of TransformerEncoderLayer's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    # Create random input tensor and mask
    x = torch.randn(batch_size, seq_len, d_model)
    mask = None

    # Initialize TransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, intermediate_size)

    # Forward pass
    output = encoder_layer(x, mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

@pytest.mark.order(6)
def test_transformer_encoder_output_shape():
    """Test the output shape of TransformerEncoder's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    max_position_embeddings = 20
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    # Create random input tensor and mask
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = None

    # Initialize TransformerEncoder
    encoder = TransformerEncoder(
        vocab_size,
        max_position_embeddings,
        d_model,
        num_heads,
        intermediate_size,
        num_layers
    )

    # Forward pass
    output = encoder(x, mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )