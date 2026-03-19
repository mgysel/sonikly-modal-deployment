"""
Modal deployment for VAE V2.7 Latent Diffusion Model
Features: Parallel Batching (5 outputs default) + Fast Cold Starts + V7 RAG Attention SOTA
"""

import modal
from pathlib import Path
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "vae-v2p7-v8-rag-attn-sota"
VOLUME_NAME = "vae-v2p7-models-v8"
MODEL_PATH = "/root/ldm_final.keras"  # Baked weights path
GPU_TYPE = "A100"

app = modal.App(APP_NAME)

# Path to local weights file (must exist on your machine)
local_weights_path = Path(__file__).parent.parent / "weights" / "ldm_final.keras"
local_rag_embeddings_path = Path(__file__).parent.parent / "weights" / "rag_db_embeddings.npy"
local_rag_params_path = Path(__file__).parent.parent / "weights" / "rag_db_params.npy"

# ============================================================================
# Image Definition
# ============================================================================

def download_models():
    """Download CLAP model during build time"""
    from transformers import ClapModel, ClapProcessor
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    print("🎨 Baking CLAP model into image...")
    model_name = "laion/clap-htsat-fused"
    ClapProcessor.from_pretrained(model_name)
    ClapModel.from_pretrained(model_name)
    print("✓ CLAP model cached in image!")

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    # STRICTER ENV VARS TO KILL XLA (Maintains fast startup)
    .env({
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "KERAS_BACKEND": "tensorflow"
    })
    .pip_install(
        "tensorflow==2.17.1",
        "keras==3.12.0",
        "numpy==1.26.4",
        "sentence-transformers==2.3.1",
        "torch==2.9.0",
        "transformers==4.57.1",
        "tf-keras",
        "fastapi[standard]",
    )
    .run_function(download_models)
    # BAKE WEIGHTS
    .add_local_file(local_path=local_weights_path, remote_path="/root/ldm_final.keras")
    .add_local_file(local_path=Path(__file__).parent.parent / "vae_v2p7.py", remote_path="/root/vae_v2p7.py")
    .add_local_file(local_path=Path(__file__).parent.parent / "encoders.py", remote_path="/root/encoders.py")
    # BAKE RAG DATABASE
    .add_local_file(local_path=local_rag_embeddings_path, remote_path="/root/rag_db_embeddings.npy")
    .add_local_file(local_path=local_rag_params_path, remote_path="/root/rag_db_params.npy")
)

# ============================================================================
# Inference Service
# ============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=2,
    enable_memory_snapshot=True,
)
class VAEv2p7Inference:
    
    @modal.enter(snap=True)
    def load_model(self):
        """
        Snapshot Phase: Load from baked weights (fast!)
        No warmup to avoid triggering XLA compilation.
        """
        import os
        import tensorflow as tf
        import keras
        
        # Double-tap XLA disable
        tf.config.optimizer.set_jit(False)
        
        # Enable Mixed Precision for A100 Speedup
        print("🚀 Enabling Mixed Precision (Float16)...")
        keras.mixed_precision.set_global_policy("mixed_float16")
        
        # Add model directory to Python path
        sys.path.insert(0, "/root")
        from vae_v2p7 import VAE_V2P7
        
        print(f"💾 Loading model from baked weights: {MODEL_PATH}")
        with tf.device("/cpu:0"):
            self.model = VAE_V2P7(
                model_path=MODEL_PATH,
                embedding_model_type="clap",
                default_diffusion_steps=50,
            )
            self.model._load_model()
            self.model._load_embedding_model()
            
            # Load RAG data if available
            self.model.load_rag_data(
                embeddings_path="/root/rag_db_embeddings.npy",
                params_path="/root/rag_db_params.npy"
            )
            
            # NO WARMUP - Avoids XLA compilation trap.
            # We accept the "unbuilt state" warnings in exchange for 40s faster boot.
        
        print("✅ Snapshot Ready with RAG database.")
    
    @modal.method()
    def generate(self, description: str, diffusion_steps: int = 50, num_outputs: int = 5, seed: int = None, 
                 guidance_scale: float = 7.5, use_rag: bool = False, rag_strength: float = 0.8, 
                 rag_top_k: int = 10, verbose: bool = False):
        """
        Generate synthesizer parameters with TRUE parallel batching and RAG support.
        
        PARALLEL EXECUTION:
        Now defaults to num_outputs=5. It sends a batch of 5 prompts to the GPU
        simultaneously. The A100 GPU will calculate all 5 in roughly the same time as 1.
        
        Performance:
            - Cold start: ~5-8s (snapshot restore)
            - Inference (1 output): ~200ms
            - Inference (5 outputs): ~250ms (only 25% slower for 5x results!)
        """
        try:
            # 1. Create the Batch
            # Instead of a loop, we create a list of N identical prompts.
            # TensorFlow handles the parallelization internally.
            prompts = [description] * num_outputs
            
            # 2. Run Parallel Inference
            # Pass the LIST directly - the model processes all prompts simultaneously on GPU
            
            # Fix for deterministic behavior due to memory snapshotting:
            if seed is None:
                import time
                # Use current time to ensure entropy
                seed = int(time.time() * 1000000) % (2**32)
                
            params = self.model.generate(
                text_description=prompts,  # Pass the LIST, not the string
                diffusion_steps=diffusion_steps,
                seed=seed,
                guidance_scale=guidance_scale,
                use_rag=use_rag,
                rag_strength=rag_strength,
                rag_top_k=rag_top_k,
                verbose=verbose,
            )
            
            if params is None:
                return {"success": False, "message": "Generation failed", "automatable_parameters": None}
            
            # 3. Return Results
            # Handle both single output (202,) and batch outputs (N, 202)
            import numpy as np
            params_array = np.asarray(params)
            
            # Ensure 2D shape for consistency
            if params_array.ndim == 1:
                # Single output: reshape (202,) -> (1, 202)
                params_array = params_array.reshape(1, -1)
            
            return {
                "success": True,
                "message": "Success",
                "automatable_parameters": params_array.tolist(),  # Returns list of N arrays
                "shape": params_array.shape,
                "count": params_array.shape[0],  # Number of outputs (first dimension)
            }

        except Exception as e:
            import traceback
            return {"success": False, "message": str(e), "traceback": traceback.format_exc()}

    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, 'model') and self.model is not None,
        }

# ============================================================================
# Web Endpoint
# ============================================================================

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_web(request: dict):
    """HTTP endpoint with parallel batching support"""
    from fastapi import HTTPException
    
    description = request.get("description")
    if not description:
        raise HTTPException(status_code=400, detail="description is required")
    
    inference = VAEv2p7Inference()
    return inference.generate.remote(
        description=description,
        diffusion_steps=request.get("diffusion_steps", 50),
        num_outputs=request.get("num_outputs", 5),  # Default to 5 parallel outputs
        seed=request.get("seed"),
        guidance_scale=request.get("guidance_scale", 7.5),
        use_rag=request.get("use_rag", False),
        rag_strength=request.get("rag_strength", 0.8),
        rag_top_k=request.get("rag_top_k", 10),
        verbose=request.get("verbose", False),
    )

# ============================================================================
# CLI Test
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the deployed model (run with: modal run modal_deploy.py)"""
    print("🧪 Testing parallel batching...")
    
    inference = VAEv2p7Inference()
    
    # Test with 5 parallel outputs
    import time
    start = time.time()
    result = inference.generate.remote(
        description="deep wobbling bass",
        diffusion_steps=50,
        num_outputs=5,
        seed=42
    )
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"✅ Generated {result['count']} variations in {elapsed:.2f}s")
        print(f"   Shape: {result['shape']}")
        print(f"   Time per output: {elapsed/result['count']:.3f}s")
    else:
        print(f"❌ Failed: {result['message']}")
