"""
VAE V2.7: Latent Diffusion Model for text-to-synth parameter generation.
"""

from .vae_v2p7 import (
    VAE_V2P7,
    VAE_Text_to_Synth_Standard,
    LatentDiffusionModel,
    DiffusionScheduler,
    ResidualBlock,
    SinusoidalTimeEmbedding,
    FiLM_Modulate,
    FiLMLayer,
    SliceLayer,
    reconstruct_parameters_from_heads,
    GROUPED_PARAMETER_TYPES,
    FLAT_PARAMETER_TYPES,
    CATEGORICAL_NUM_CLASSES,
)

__all__ = [
    "VAE_V2P7",
    "VAE_Text_to_Synth_Standard",
    "LatentDiffusionModel",
    "DiffusionScheduler",
    "ResidualBlock",
    "SinusoidalTimeEmbedding",
    "FiLM_Modulate",
    "FiLMLayer",
    "SliceLayer",
    "reconstruct_parameters_from_heads",
    "GROUPED_PARAMETER_TYPES",
    "FLAT_PARAMETER_TYPES",
    "CATEGORICAL_NUM_CLASSES",
]
