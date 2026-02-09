"""
Modal deployment for VAE V2.7 Oscillator Separated Model
Features: Parallel Batching + Wavetable/Noise Matching + Fast Cold Starts
"""

import modal
from pathlib import Path
import sys
import os
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "vae-v2p7-osc-separated-inference"
MODEL_PATH = "/root/weights/ldm_final_tuned.keras"  # Baked weights path
WT_LIB_DIR = "/root/wavetables"
GPU_TYPE = "L4"

app = modal.App(APP_NAME)

# Local paths
local_model_dir = Path(__file__).parent.parent
local_weights_path = local_model_dir / "weights" / "ldm_final_tuned.keras"
local_wavetables_dir = local_model_dir / "wavetables"

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
        "transformers==4.38.0", # Updated transformers version
        "tf-keras",
        "fastapi[standard]",
        "scikit-learn" 
    )
    .run_function(download_models)
    # BAKE WEIGHTS
    .add_local_file(local_path=local_weights_path, remote_path=MODEL_PATH)
    # BAKE WAVETABLE DATA
    .add_local_dir(local_path=local_wavetables_dir, remote_path=WT_LIB_DIR)
    # ADD CODE
    .add_local_file(local_path=local_model_dir / "vae_v2p7_osc_separated.py", remote_path="/root/vae_v2p7_osc_separated.py")
    .add_local_file(local_path=local_model_dir / "serum_params.py", remote_path="/root/serum_params.py")
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
class VAEv2p7OscSeparatedInference:
    
    @modal.enter(snap=True)
    def load_model(self):
        """
        Snapshot Phase: Load from baked weights + Wavetable Libraries
        """
        import os
        import tensorflow as tf
        import numpy as np
        from transformers import ClapModel, ClapProcessor
        
        # Double-tap XLA disable
        tf.config.optimizer.set_jit(False)
        
        # 1. Load CLAP (Text Encoder)
        print("🎧 Loading CLAP...")
        model_name = "laion/clap-htsat-fused"
        self.clap_processor = ClapProcessor.from_pretrained(model_name)
        self.clap_model = ClapModel.from_pretrained(model_name)
        
        # 2. Setup Imports & Utils
        sys.path.insert(0, "/root")
        from serum_params import SERUM_PARAMETERS
        
        # Import the model classes directly so Keras can reconstruct the graph
        from vae_v2p7_osc_separated import (
            VAE_V2P7_OSC_SEPARATED,
            ParameterUtils, numpy_to_json
        )
        
        self.SERUM_PARAMETERS = SERUM_PARAMETERS
        self.numpy_to_json = numpy_to_json
        self.ParameterUtils = ParameterUtils
        self.param_info = ParameterUtils.get_indices_and_classes(SERUM_PARAMETERS)
        
        # 3. Load Keras Model via Wrapper
        print(f"💾 Loading model wrapper from: {MODEL_PATH}")
        self.model = VAE_V2P7_OSC_SEPARATED(MODEL_PATH)
        self.model.load()
            
        # 4. Load Wavetable Libraries
        # Using self.model here? No, using numpy loads directly
        print("📚 Loading Wavetable & Noise Libraries...")
        try:
            self.wt_names = np.load(os.path.join(WT_LIB_DIR, 'default_wavetable_names.npy'), allow_pickle=True)
            self.wt_embeds = np.load(os.path.join(WT_LIB_DIR, 'default_wavetable_embeddings.npy'), allow_pickle=True).astype('float32')
            self.noise_names = np.load(os.path.join(WT_LIB_DIR, 'default_noise_names.npy'), allow_pickle=True)
            self.noise_embeds = np.load(os.path.join(WT_LIB_DIR, 'default_noise_embeddings.npy'), allow_pickle=True).astype('float32')
            
             # Normalize library embeddings for cosine similarity
            self.wt_embeds = self.wt_embeds / (np.linalg.norm(self.wt_embeds, axis=1, keepdims=True) + 1e-9)
            self.noise_embeds = self.noise_embeds / (np.linalg.norm(self.noise_embeds, axis=1, keepdims=True) + 1e-9)
            
            print(f"✓ Loaded {len(self.wt_names)} wavetables and {len(self.noise_names)} noises.")
        except Exception as e:
            print(f"❌ Error loading libraries: {e}")
            # Non-critical failure, can still generate parameters
            self.wt_names = []
            self.wt_embeds = None
            
        print("✅ Snapshot Ready.")

    def find_nearest_matches(self, target_embedding, library_embeddings, library_names, top_k=3):
        if library_embeddings is None: return []
        
        # Normalize target
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
        
        # Cosine Similarity
        similarities = np.dot(library_embeddings, target_norm)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        matches = []
        for idx in top_indices:
            matches.append({
                "name": str(library_names[idx]),
                "score": float(similarities[idx]),
                "index": int(idx)
            })
        return matches

    def encode_text(self, text_list):
        """Helper to convert text strings to Embeddings using CLAP"""
        import torch
        inputs = self.clap_processor(text=text_list, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeds = self.clap_model.get_text_features(**inputs)
        return text_embeds.numpy() # Returns (Batch, 512)

    @modal.method()
    def generate(self, description: str, diffusion_steps: int = 50, num_outputs: int = 5, seed: int = None, verbose: bool = False):
        try:
            import tensorflow as tf
            
            # 1. Handle Seeding
            if seed is not None:
                tf.random.set_seed(seed)
                np.random.seed(seed)

            # 2. Encode Text (Critical Step)
            prompts = [description] * num_outputs
            text_embeds = self.encode_text(prompts) # Shape: (N, 512)
            
            # 3. Run Parallel Inference
            # Pass TENSORS (text_embeds), not strings
            raw_outputs = self.model.generate(
                text_embeds=text_embeds,
                steps=diffusion_steps
            )
            
            # 4. Process Batch
            all_auto_params = []
            all_non_auto_params = []
            
            # Convert all heads to numpy first
            numpy_heads = [t.numpy() for t in raw_outputs]
            batch_size = numpy_heads[0].shape[0]
            
            for i in range(batch_size):
                # Slice heads for this sample
                sample_heads = [h[i] for h in numpy_heads]
                
                # Reconstruct Parameters & Get Audio Vectors
                # Need to flatten SERUM_PARAMETERS map first for compatibility
                flat_params = {int(p['id']): p for group in self.SERUM_PARAMETERS.values() for p in group}
                
                rec_params, audio_vecs = self.ParameterUtils.reconstruct_parameters_from_heads(
                    sample_heads,
                    flat_params, 
                    self.param_info[3] # categorical_num_classes
                )

                # Match Wavetables
                match_a = self.find_nearest_matches(audio_vecs.get("osc_a"), self.wt_embeds, self.wt_names)
                match_b = self.find_nearest_matches(audio_vecs.get("osc_b"), self.wt_embeds, self.wt_names)
                match_n = self.find_nearest_matches(audio_vecs.get("noise"), self.noise_embeds, self.noise_names)
                
                # 1. Automatable Parameters (Flat Array)
                all_auto_params.append(rec_params.tolist())
                
                # 2. Non-Automatable Parameters (Wavetable Paths)
                all_non_auto_params.append({
                    "osc_a": [m['name'] for m in match_a],
                    "osc_b": [m['name'] for m in match_b],
                    "osc_n": [m['name'] for m in match_n]
                })

            return {
                "success": True,
                "message": "Success",
                "count": num_outputs,
                "automatable_parameters": all_auto_params,
                "non_automatable_parameters": all_non_auto_params
            }

        except Exception as e:
            import traceback
            return {"success": False, "message": str(e), "traceback": traceback.format_exc()}

    @modal.method()
    def health_check(self) -> dict:
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
    
    inference = VAEv2p7OscSeparatedInference()
    return inference.generate.remote(
        description=description,
        diffusion_steps=request.get("diffusion_steps", 50),
        num_outputs=request.get("num_outputs", 5),
        seed=request.get("seed"),
        verbose=request.get("verbose", False),
    )

# ============================================================================
# CLI Test
# ============================================================================

@app.local_entrypoint()
def test():
    """Test the deployed model"""
    print("🧪 Testing VAE V2.7 Oscillator Separated Inference...")
    
    inference = VAEv2p7OscSeparatedInference()
    
    import time
    start = time.time()
    result = inference.generate.remote(
        description="aggressive reese bass",
        diffusion_steps=50,
        num_outputs=2,
        seed=42
    )
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"✅ Generated {result['count']} outputs in {elapsed:.2f}s")
        
        auto_params_list = result['automatable_parameters']
        non_auto_list = result['non_automatable_parameters']
        
        for i in range(len(auto_params_list)):
            print(f"\nOutput {i+1}:")
            
            # automatable params (Flat list check)
            params = auto_params_list[i]
            print(f"  Params Length: {len(params)}")
            print(f"  Param 0: {params[0]:.4f}")
            
            # non-automatable
            non_auto = non_auto_list[i]
            print(f"  Osc A: {non_auto['osc_a']}")
            print(f"  Osc B: {non_auto['osc_b']}")
            print(f"  Noise: {non_auto['osc_n']}")
    else:
        print(f"❌ Failed: {result['message']}")