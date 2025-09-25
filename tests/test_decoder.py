# test_decoder.py

import pytest
import torch
import torch.nn as nn

from src.decoder import TransformerDecoderLayer, TransformerDecoder

@pytest.mark.order(7)
def test_transformer_decoder_layer_output_shape():
    """Test the output shape of TransformerDecoderLayer's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    intermediate_size = 16

    # Create random input tensors
    x = torch.randn(batch_size, seq_len, d_model)
    enc_output = torch.randn(batch_size, seq_len, d_model)
    tgt_mask = torch.ones(batch_size, seq_len, seq_len)

    # Initialize TransformerDecoderLayer
    decoder_layer = TransformerDecoderLayer(d_model, num_heads, intermediate_size)

    # Forward pass
    output = decoder_layer(x, enc_output, tgt_mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )

@pytest.mark.order(8)
def test_transformer_decoder_output_shape():
    """Test the output shape of TransformerDecoder's forward method."""
    # Initialize parameters
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    max_position_embeddings = 20
    d_model = 8
    num_heads = 2
    intermediate_size = 16
    num_layers = 2

    # Create random input tensors
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    enc_output = torch.randn(batch_size, seq_len, d_model)

    # Initialize TransformerDecoder
    decoder = TransformerDecoder(
        vocab_size,
        max_position_embeddings,
        d_model,
        num_heads,
        intermediate_size,
        num_layers
    )

    # Forward pass
    output = decoder(input_ids, enc_output)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    )
