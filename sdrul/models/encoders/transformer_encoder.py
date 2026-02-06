"""
Transformer-based feature extractor for RUL prediction.

This module provides a shared feature extractor used by both TCSD and DSA-MoE.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeatureExtractor(nn.Module):
    """
    Transformer-based feature extractor for sensor time series.

    Shared by TCSD and DSA-MoE for extracting temporal features from
    multi-sensor degradation data.

    Args:
        sensor_dim: Number of input sensor channels (default: 14)
        d_model: Transformer model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 4)
        dim_feedforward: Feedforward network dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
        max_len: Maximum sequence length (default: 500)
    """

    def __init__(
        self,
        sensor_dim: int = 14,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(sensor_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features from sensor time series.

        Args:
            x: Sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask [batch, seq_len], True for valid positions

        Returns:
            features: Extracted features [batch, seq_len, d_model]
        """
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask if provided
        # TransformerEncoder expects src_key_padding_mask where True = ignore
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask  # Invert: True -> ignore

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Output normalization
        x = self.output_norm(x)

        return x

    def get_sequence_embedding(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pooling: str = 'last',
    ) -> torch.Tensor:
        """
        Get a single embedding vector for the entire sequence.

        Args:
            x: Sensor data [batch, seq_len, sensor_dim]
            mask: Optional attention mask [batch, seq_len]
            pooling: Pooling method ('last', 'mean', 'max')

        Returns:
            embedding: Sequence embedding [batch, d_model]
        """
        features = self.forward(x, mask)  # [batch, seq_len, d_model]

        if pooling == 'last':
            if mask is not None:
                # Get last valid position for each sequence
                lengths = mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(features.size(0), device=features.device)
                embedding = features[batch_indices, lengths]
            else:
                embedding = features[:, -1, :]
        elif pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                embedding = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                embedding = features.mean(dim=1)
        elif pooling == 'max':
            if mask is not None:
                features = features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            embedding = features.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return embedding
