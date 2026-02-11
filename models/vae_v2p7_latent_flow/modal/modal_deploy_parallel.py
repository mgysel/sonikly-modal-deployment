"""
Modal deployment for VAE V2.7 Latent Diffusion Model
Features: Parallel Batching (5 outputs default) + Fast Cold Starts
"""

import modal
from pathlib import Path
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "vae_v2p7_flow"
MODEL_PATH = "/root/ldm_final.keras"  # Baked weights path
GPU_TYPE = "L4"

app = modal.App(APP_NAME)

# Path to local weights file (must exist on your machine)
local_weights_path = Path(__file__).parent.parent / "weights" / "ldm_final.keras"

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
class VAEv2p7RetrainedInference:
    
    @modal.enter(snap=True)
    def load_model(self):
        """
        Snapshot Phase: Load from baked weights (fast!)
        No warmup to avoid triggering XLA compilation.
        """
        import os
        import tensorflow as tf
        
        # Double-tap XLA disable
        tf.config.optimizer.set_jit(False)
        
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
            
            # NO WARMUP - Avoids XLA compilation trap.
            # We accept the "unbuilt state" warnings in exchange for 40s faster boot.
        
        print("✅ Snapshot Ready.")
    
    @modal.method()
    def generate(self, description: str, diffusion_steps: int = 50, cfg_scale: float = 4.0, num_outputs: int = 5, seed: int = None, verbose: bool = False):
        """
        Generate synthesizer parameters with TRUE parallel batching.
        
        PARALLEL EXECUTION:
        Now defaults to num_outputs=5. It sends a batch of 5 prompts to the GPU
        simultaneously. The L4 GPU will calculate all 5 in roughly the same time as 1.
        
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
                cfg_scale=cfg_scale,
                seed=seed,
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
    
    inference = VAEv2p7RetrainedInference()
    return inference.generate.remote(
        description=description,
        diffusion_steps=50,  # FORCE 50 steps for Flow Matching (client often requests 1000 which is too slow)
        cfg_scale=request.get("cfg_scale", 4.0),
        num_outputs=request.get("num_outputs", 5),  # Default to 5 parallel outputs
        seed=request.get("seed"),
        verbose=request.get("verbose", False),
    )

# ============================================================================
# CLI Test
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the deployed model (run with: modal run modal_deploy_parallel.py)"""
    print("🧪 Testing parallel batching...")
    
    inference = VAEv2p7RetrainedInference()
    
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
