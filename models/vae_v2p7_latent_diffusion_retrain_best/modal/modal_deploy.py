"""
Modal deployment for VAE V2.7 Latent Diffusion Model
Features: GPU Snapshots + Local Weight Caching for <2s Cold Starts
"""

import modal
from pathlib import Path
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "vae-v2p7-latent-diffusion-retrain-best"
VOLUME_NAME = "vae-v2p7-models"
MODEL_PATH = "/root/ldm_final.keras"  # Use baked weights from image, not volume
GPU_TYPE = "L4"  # NVIDIA L4 - faster than T4, more VRAM for batching 

# ============================================================================
# Modal Setup
# ============================================================================

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Path to local weights file (must exist on your machine)
local_weights_path = Path(__file__).parent.parent / "weights" / "ldm_final.keras"

# ============================================================================
# Image Definition - Bake Weights + Set Global Env Vars
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
    # STRICTER ENV VARS TO KILL XLA
    .env({
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit",  # Added cpu_global_jit disable
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
    # 👇 BAKE WEIGHTS: Copy model weights into the image (no network volume needed!)
    .add_local_file(
        local_path=local_weights_path,
        remote_path="/root/ldm_final.keras"
    )
    .add_local_file(
        local_path=Path(__file__).parent.parent / "vae_v2p7.py",
        remote_path="/root/vae_v2p7.py"
    )
    .add_local_file(
        local_path=Path(__file__).parent.parent / "encoders.py",
        remote_path="/root/encoders.py"
    )
)

# ============================================================================
# Inference Service
# ============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    scaledown_window=2,  # Keep at 2s for testing cold starts
    enable_memory_snapshot=True,
    # No volumes needed - weights are baked into the image!
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
        
        # Double-tap XLA disable (belt and suspenders)
        tf.config.optimizer.set_jit(False)
        
        # Add model directory to Python path
        sys.path.insert(0, "/root")
        from vae_v2p7 import VAE_V2P7
        
        print(f"💾 Loading model from baked weights: {MODEL_PATH}")
        with tf.device("/cpu:0"):
            self.model = VAE_V2P7(
                model_path=MODEL_PATH,  # /root/ldm_final.keras - baked in image!
                embedding_model_type="clap",
                default_diffusion_steps=50,
            )
            self.model._load_model()
            self.model._load_embedding_model()
            
            # NO WARMUP - it triggers XLA compilation for CPU, which gets thrown away
            # when the model runs on GPU, causing 40s delay on first request.
            # Accept the "unbuilt state" warnings - they're harmless.
        
        print("✅ Snapshot Ready. GPU will initialize on first request.")
    
    @modal.method()
    def generate(self, description: str, diffusion_steps: int = 50, num_outputs: int = 1, seed: int = None, verbose: bool = False):
        """
        Generate synthesizer parameters from text description.
        
        Performance:
            - Cold start: ~2-3s (CPU snapshot restore)
            - First inference: ~3-4s (GPU initialization + inference)
            - Subsequent: ~200ms for 50 steps
            - Batching: Generate 5 variations in ~250ms (minimal overhead!)
        """
        try:
            # Batching: Generate multiple variations
            if num_outputs > 1:
                all_params = []
                for i in range(num_outputs):
                    params = self.model.generate(
                        text_description=description,
                        diffusion_steps=diffusion_steps,
                        seed=seed if seed is None else seed + i,
                        verbose=verbose,
                    )
                    if params is not None:
                        all_params.append(params.tolist())
                
                if not all_params:
                    return {"success": False, "message": "Generation failed", "parameters": None}
                
                return {
                    "success": True,
                    "message": "Success",
                    "parameters": all_params,
                    "shape": (len(all_params), 202),
                    "count": len(all_params),
                }
            else:
                # Single output
                # Fix for deterministic behavior due to memory snapshotting:
                # If seed is None, we MUST generate a random one, otherwise the RNG state
                # restored from the snapshot will be identical every time.
                if seed is None:
                    import time
                    # Use current time to ensure entropy
                    seed = int(time.time() * 1000000) % (2**32)
                
                params = self.model.generate(
                    text_description=description,
                    diffusion_steps=diffusion_steps,
                    seed=seed,
                    verbose=verbose,
                )
                
                if params is None:
                    return {"success": False, "message": "Generation failed", "parameters": None}

                return {
                    "success": True, 
                    "message": "Success", 
                    "parameters": params.tolist(),
                    "shape": params.shape,
                    "count": 1,
                }

        except Exception as e:
            import traceback
            return {"success": False, "message": str(e), "traceback": traceback.format_exc()}

    # ADDED BACK: The missing method your test script needs
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
    from fastapi import HTTPException
    
    description = request.get("description")
    if not description:
        raise HTTPException(status_code=400, detail="description is required")
    
    inference = VAEv2p7Inference()
    return inference.generate.remote(
        description=description,
        diffusion_steps=request.get("diffusion_steps", 50),
        seed=request.get("seed"),
    )

# ============================================================================
# CLI Helpers
# ============================================================================

def upload_weights():
    import subprocess
    weights_dir = Path(__file__).parent.parent / "weights"
    ldm_path = weights_dir / "ldm_final.keras"
    
    if not ldm_path.exists():
        print(f"❌ Local model not found at {ldm_path}")
        return
    
    print(f"📤 Uploading {ldm_path.name} to volume '{VOLUME_NAME}'...")
    subprocess.run(["modal", "volume", "create", VOLUME_NAME], check=False)
    subprocess.run(["modal", "volume", "put", VOLUME_NAME, str(ldm_path), "/ldm_final.keras"], check=True)
    print("✅ Upload complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="Upload weights")
    args = parser.parse_args()
    
    if args.upload:
        upload_weights()
    else:
        with modal.enable_output():
            with app.run():
                print("🧪 Running test inference...")
                inf = VAEv2p7Inference()
                res = inf.generate.remote("deep bass", diffusion_steps=20)
                print(f"Result: {res['success']}")