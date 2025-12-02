"""
Fusion Features Extractor for Multi-Modal Perception

This module implements a dual-stream network that processes both visual (camera)
and proprioceptive (physics telemetry) inputs into a unified latent representation
suitable for recurrent policy learning.

Architecture:
    Visual Stream: 3-layer CNN (NatureCNN-style) -> 256-dim features
    Physics Stream: Identity passthrough -> 12-dim telemetry
    Fusion: Concatenate + LayerNorm -> 268-dim output

The LayerNorm is crucial for stable LSTM training, as it normalizes the
heterogeneous feature scales between high-variance visual features and
bounded physics values.
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FusionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Dual-Stream Fusion Network with Layer Normalization.
    Designed for 128x128 RGB input images with 12-dimensional telemetry vectors.
    The output dimension is 268 (256 CNN features + 12 physics features).
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 268):  # 256 (CNN) + 12 (Physics)
        """
        Initialize the fusion extractor.

        Args:
            observation_space: Dict space with 'image' and 'vec' keys
            features_dim: Total output dimension (should match cnn_output + vec_dim)
        """
        # We need the vector dimension from the environment
        vec_dim = observation_space["vec"].shape[0]
        cnn_output_dim = 256
        total_dim = cnn_output_dim + vec_dim

        super().__init__(observation_space, features_dim=total_dim)

        # 1. Compressed Visual Stream (Custom NatureCNN-like)
        # Adapted for 128x128 resolution
        # Calculation:
        # Input: 128x128
        # Conv1 (8x8, s4): (128-8)/4 + 1 = 31
        # Conv2 (4x4, s2): (31-4)/2 + 1 = 14
        # Conv3 (3x3, s1): (14-3)/1 + 1 = 12
        # Flatten: 64 channels * 12 * 12 = 9216

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, cnn_output_dim),  # UPDATED: 3136 -> 9216 for 128x128
            nn.ReLU()
        )

        # 2. Physics Stream is passed through raw (Identity)

        # 3. Regularization: LayerNorm
        # Normalizes the combined [Visual (0-inf), Physics (-1 to 1)] vector.
        self.fusion_norm = nn.LayerNorm(total_dim)

    def forward(self, observations: dict) -> torch.Tensor:
        # Process images
        visual_feats = self.cnn(observations["image"])

        # Get physics vector
        physics_feats = observations["vec"]

        # Fuse
        fused = torch.cat([visual_feats, physics_feats], dim=1)

        # Normalize and Return
        return self.fusion_norm(fused)