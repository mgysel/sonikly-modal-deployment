"""
Text encoding functions for VAE models.

This module provides text embedding functionality using:
- CLAP (Contrastive Language-Audio Pretraining)
- SentenceTransformer
"""

import torch
import numpy as np

# Try to import CLAP
try:
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    print("Warning: transformers library not available for CLAP. Only SentenceTransformer will be available.")

# Global model storage
_clap_model = None
_clap_processor = None
_sentence_transformer_model = None


def clap_encode_text(texts, batch_size=128, normalize=True, max_length=None):
    """
    Encode text using CLAP model.
    
    Args:
        texts: List of text strings or single string
        batch_size: Batch size for processing
        normalize: Whether to normalize embeddings
        max_length: Maximum sequence length
    
    Returns:
        numpy array of embeddings [B, 512]
    """
    global _clap_model, _clap_processor
    
    if _clap_model is None or _clap_processor is None:
        raise RuntimeError("CLAP model not loaded. Call load_clap_model() first.")
    
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure model is on the correct device
    _clap_model = _clap_model.to(device)
    _clap_model.eval()
    
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i+batch_size]]
            inputs = _clap_processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            # Move all input tensors to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = _clap_model.get_text_features(**inputs)  # [B, 512]
            if normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            out.append(feats.cpu())
    
    return torch.cat(out, dim=0).numpy()


def get_sentence_transformer_model():
    """Get the SentenceTransformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("SentenceTransformer model loaded")
    return _sentence_transformer_model


def load_clap_model():
    """Load the CLAP model."""
    global _clap_model, _clap_processor
    
    if not CLAP_AVAILABLE:
        raise RuntimeError("CLAP model requested but transformers library not available")
    
    if _clap_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clap_model_name = "laion/clap-htsat-fused"
        print(f"Loading CLAP model: {clap_model_name} on {device}")
        _clap_model = ClapModel.from_pretrained(clap_model_name).to(device)
        _clap_processor = ClapProcessor.from_pretrained(clap_model_name)
        _clap_model.eval()
        print(f"CLAP model loaded successfully on {device}")
    
    return _clap_model, _clap_processor


def encode_text(texts, embedding_model_type="clap"):
    """
    Encode text using the specified embedding model.
    
    Args:
        texts: List of text strings or single string
        embedding_model_type: "clap" or "sentence_transformer"
    
    Returns:
        numpy array of embeddings
    """
    if embedding_model_type == "clap":
        if _clap_model is None:
            load_clap_model()
        return clap_encode_text(texts)
    else:
        model = get_sentence_transformer_model()
        if isinstance(texts, str):
            texts = [texts]
        return model.encode(texts)
