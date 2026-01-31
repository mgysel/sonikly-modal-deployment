"""
Modal deployment for VAE V2.7 Latent Diffusion Model

Quick Start:
    1. Install Modal: pip install modal
    2. Authenticate: modal setup
    3. Upload weights: python modal_deploy.py --upload
    4. Deploy: modal deploy modal_deploy.py
    5. Test: modal run modal_deploy.py

Usage in your backend:
    import modal
    model = modal.Function.lookup("vae-v2p7-inference", "VAEv2p7Inference")
    result = model.generate.remote(description="deep bass", diffusion_steps=1000)
    params = result["parameters"]  # List of 202 floats
"""

import modal
from pathlib import Path
import sys

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "vae-v2p7-inference"
VOLUME_NAME = "vae-v2p7-models"
MODEL_PATH = "/models/ldm_final.keras"
GPU_TYPE = "T4"  # Options: "T4", "A10G", "A100"

# ============================================================================
# Modal Setup
# ============================================================================

app = modal.App(APP_NAME)

# ============================================================================
# Model Caching - Bake models into image for fast cold starts
# ============================================================================

def download_models():
    """Download CLAP model during build time to bake it into the image"""
    from transformers import ClapModel, ClapProcessor
    import os
    
    # Silence warnings during build
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    print("🎨 Baking CLAP model into image...")
    # This matches the model used in encoders.py
    model_name = "laion/clap-htsat-fused"
    print(f"Downloading {model_name}...")
    ClapProcessor.from_pretrained(model_name)
    ClapModel.from_pretrained(model_name)
    print("✓ CLAP model cached in image!")

# Container image with all dependencies
image = (
    # Use NVIDIA CUDA base image for GPU support (devel includes libdevice for JIT compilation)
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "tensorflow==2.17.1",
        "keras==3.12.0",
        "numpy==1.26.4",
        "sentence-transformers==2.3.1",
        "torch==2.9.0",
        "transformers==4.57.1",
        "tf-keras",
        "fastapi[standard]",  # Required for web endpoints
    )
    .run_function(download_models)  # <--- Bake CLAP model into image for fast cold starts
    .add_local_file(
        local_path=Path(__file__).parent.parent / "vae_v2p7.py",
        remote_path="/root/vae_v2p7.py"
    )
    .add_local_file(
        local_path=Path(__file__).parent.parent / "encoders.py",
        remote_path="/root/encoders.py"
    )
)

# Persistent volume for model weights
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ============================================================================
# Inference Service
# ============================================================================

@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=300,
    scaledown_window=120,
    volumes={"/models": volume},
    enable_memory_snapshot=True,  # Enable memory snapshotting for fast cold starts
)
class VAEv2p7Inference:
    """VAE V2.7 inference service"""
    
    @modal.enter(snap=True)
    def load_model(self):
        """Load model on container startup (v2 - fixed device placement)"""
        import os
        import tensorflow as tf
        
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ.pop("TF_USE_LEGACY_KERAS", None)
        
        # Disable XLA globally to avoid libdevice issues, but keep @tf.function optimization
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
        tf.config.optimizer.set_jit(False)
        
        # Configure TensorFlow GPU
        print("Checking GPU availability...")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s): {gpus}")
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
            print("✓ XLA JIT disabled (using @tf.function graph optimization instead)")
        else:
            print("⚠️  WARNING: No GPU detected! Running on CPU.")
        
        # Add model directory to Python path
        sys.path.insert(0, "/root")
        
        from vae_v2p7 import VAE_V2P7
        
        print(f"Loading model from {MODEL_PATH}")
        self.model = VAE_V2P7(
            model_path=MODEL_PATH,
            embedding_model_type="clap",
            default_diffusion_steps=1000,
        )
        self.model._load_model()
        self.model._load_embedding_model()
        print("Model loaded successfully!")
        
        # Warmup inference to trigger JIT compilation during startup
        print("🔥 Running warmup inference for snapshot...")
        try:
            _ = self.model.generate(
                text_description="warmup",
                diffusion_steps=10,  # Small warmup - compiled state will be snapshotted
                verbose=False
            )
            print("✓ Warmup complete - State is ready to snapshot!")
        except Exception as e:
            print(f"⚠️  Warmup inference failed (non-critical): {e}")
    
    @modal.method()
    def generate(
        self,
        description: str,
        diffusion_steps: int = 50,  # Reduced from 1000 for faster inference
        seed: int = None,
        verbose: bool = False,
    ) -> dict:
        """Generate synthesizer parameters from text description"""
        try:
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
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "parameters": None,
            }
    
    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
        }

# ============================================================================
# Web Endpoint (Optional)
# ============================================================================

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate_web(request: dict):
    """HTTP endpoint for generation"""
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
# Helper Scripts
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the deployed model (run with: modal run modal_deploy.py)"""
    print("Testing VAE V2.7 model...")
    
    inference = VAEv2p7Inference()
    result = inference.generate.remote(
        description="deep wobbling bass with heavy modulation",
        diffusion_steps=1000,
        seed=42,
        verbose=True,
    )
    
    print(f"\nSuccess: {result['success']}")
    print(f"Message: {result['message']}")
    if result['success']:
        print(f"Parameters shape: {result['shape']}")
        print(f"First 10 params: {result['parameters'][:10]}")


def upload_weights():
    """Upload model weights to Modal volume using the CLI"""
    import subprocess
    
    weights_dir = Path(__file__).parent.parent / "weights"
    ldm_path = weights_dir / "ldm_final.keras"
    
    if not ldm_path.exists():
        print(f"❌ Model not found at {ldm_path}")
        return False
    
    print(f"📤 Uploading {ldm_path.name} ({ldm_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    
    # Ensure the volume exists first (idempotent if it already exists)
    subprocess.run(["modal", "volume", "create", VOLUME_NAME], check=False)

    try:
        # Use the Modal CLI to handle the upload
        subprocess.run(
            ["modal", "volume", "put", VOLUME_NAME, str(ldm_path), "/ldm_final.keras"],
            check=True
        )
        print("✅ Upload complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VAE V2.7 Modal Deployment")
    parser.add_argument("--upload", action="store_true", help="Upload model weights to Modal")
    args = parser.parse_args()
    
    if args.upload:
        upload_weights()
    else:
        print("Usage:")
        print("  Upload weights: python modal_deploy.py --upload")
        print("  Deploy:         modal deploy modal_deploy.py")
        print("  Test:           modal run modal_deploy.py")
