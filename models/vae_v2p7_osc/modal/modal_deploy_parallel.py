"""
Modal deployment for VAE V2.7 Oscillator Model
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

APP_NAME = "vae-v2p7-osc-inference"
MODEL_PATH = "/root/ldm_final_audio.keras"  # Baked weights path
WT_LIB_DIR = "/root/wavetables"
GPU_TYPE = "L4"

app = modal.App(APP_NAME)

# Local paths
local_model_dir = Path(__file__).parent.parent
local_weights_path = local_model_dir / "weights" / "ldm_final_audio.keras"
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
        "transformers==4.57.1",
        "tf-keras",
        "fastapi[standard]",
        "scikit-learn" # For cosine similarity if needed, though numpy is faster for simple cases
    )
    .run_function(download_models)
    # BAKE WEIGHTS
    .add_local_file(local_path=local_weights_path, remote_path=MODEL_PATH)
    # BAKE WAVETABLE DATA
    .add_local_dir(local_path=local_wavetables_dir, remote_path=WT_LIB_DIR)
    # ADD CODE
    .add_local_file(local_path=local_model_dir / "vae_v2p7_osc.py", remote_path="/root/vae_v2p7_osc.py")
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
class VAEv2p7OscInference:
    
    @modal.enter(snap=True)
    def load_model(self):
        """
        Snapshot Phase: Load from baked weights + Wavetable Libraries
        """
        import os
        import tensorflow as tf
        import numpy as np
        
        # Double-tap XLA disable
        tf.config.optimizer.set_jit(False)
        
        # Add model directory to Python path
        sys.path.insert(0, "/root")
        from vae_v2p7_osc import VAE_V2P7_OSC, ParameterUtils, numpy_to_json
        from serum_params import SERUM_PARAMETERS
        
        self.SERUM_PARAMETERS = SERUM_PARAMETERS
        self.numpy_to_json = numpy_to_json
        self.ParameterUtils = ParameterUtils
        
        # Pre-compute indices
        self.param_info = ParameterUtils.get_indices_and_classes(SERUM_PARAMETERS)
        
        print(f"💾 Loading model from baked weights: {MODEL_PATH}")
        with tf.device("/cpu:0"):
            self.model = VAE_V2P7_OSC(
                model_path=MODEL_PATH,
                timesteps=1000
            )
            self.model.load()
            
        print("📚 Loading Wavetable & Noise Libraries...")
        try:
            self.wt_names = np.load(os.path.join(WT_LIB_DIR, 'default_wavetable_names.npy'), allow_pickle=True)
            self.wt_embeds = np.load(os.path.join(WT_LIB_DIR, 'default_wavetable_embeddings.npy'), allow_pickle=True).astype('float32')
            self.noise_names = np.load(os.path.join(WT_LIB_DIR, 'default_noise_names.npy'), allow_pickle=True)
            self.noise_embeds = np.load(os.path.join(WT_LIB_DIR, 'default_noise_embeddings.npy'), allow_pickle=True).astype('float32')
            
             # Normalize library embeddings for cosine similarity (Dot product of normalized vectors)
            self.wt_embeds = self.wt_embeds / np.linalg.norm(self.wt_embeds, axis=1, keepdims=True)
            self.noise_embeds = self.noise_embeds / np.linalg.norm(self.noise_embeds, axis=1, keepdims=True)
            
            print(f"✓ Loaded {len(self.wt_names)} wavetables and {len(self.noise_names)} noises.")
        except Exception as e:
            print(f"❌ Error loading libraries: {e}")
            raise e
            
        print("✅ Snapshot Ready.")

    def find_nearest_matches(self, target_embedding, library_embeddings, library_names, top_k=3):
        """
        Find top_k nearest neighbors using Cosine Similarity.
        target_embedding: (512,)
        library_embeddings: (N, 512) - Already normalized
        """
        # Normalize target
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
        
        # Cosine Similarity: Dot product
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

    @modal.method()
    def generate(self, description: str, diffusion_steps: int = 50, num_outputs: int = 5, seed: int = None, verbose: bool = False):
        """
        Generate synthesizer parameters + Matched Wavetables
        """
        try:
            # 1. Create the Batch
            prompts = [description] * num_outputs
            
            # 2. Run Parallel Inference
            # Fix for deterministic behavior
            if seed is None:
                import time
                seed = int(time.time() * 1000000) % (2**32)
                
            # Calling the model wrapper
            # Returns list of numpy arrays. Last 3 are audio embeddings.
            # Decoder output order: 
            # [unipolar(202), bipolar_gate(32), bipolar_value(32), boolean(16), cat_0...cat_N, osc_a(512), osc_b(512), osc_n(512)]
            # WAIT. The decoder outputs a LIST of tensors.
            # We need to correctly parse this list.
            
            # Let's inspect the `vae_v2p7_osc.py` decoder structure again.
            # outs = [unipolar, bipolar_gate, bipolar_val, bool, cat_0...cat_N, osc_a, osc_b, osc_n]
            # Since `params` returned by model.generate() is the raw list of outputs from Keras model.
            
            raw_outputs = self.model.generate(
                prompts=prompts,
                steps=diffusion_steps,
                seed=seed,
            )
            
            # 3. Process Batch
            results = []
            
            # Convert all heads to numpy first
            numpy_heads = [t.numpy() for t in raw_outputs]
            batch_size = numpy_heads[0].shape[0]
            
            for i in range(batch_size):
                # Slice heads for this sample
                sample_heads = [h[i] for h in numpy_heads]
                
                # Reconstruct Parameters & Get Audio Vectors
                rec_params, audio_vecs = self.ParameterUtils.reconstruct_parameters_from_heads(
                    sample_heads, 
                    self.param_info
                )
                
                # Match Wavetables
                match_a = self.find_nearest_matches(audio_vecs["osc_a"], self.wt_embeds, self.wt_names)
                match_b = self.find_nearest_matches(audio_vecs["osc_b"], self.wt_embeds, self.wt_names)
                match_n = self.find_nearest_matches(audio_vecs["noise"], self.noise_embeds, self.noise_names)
                
                # Convert to Notebook-Style JSON
                json_params = self.numpy_to_json(rec_params, self.SERUM_PARAMETERS)
                
                results.append({
                    "automatable_parameters": json_params,
                    "non_automatable_parameters": {
                        "osc_a": [m['name'] for m in match_a], # Just strings as requested? User said "list of the 3 strings"
                        "osc_b": [m['name'] for m in match_b],
                        "osc_n": [m['name'] for m in match_n],
                        # Keeping full match objects potentially useful but user asked for "list of the 3 strings"
                        # Actually user said "osc_a, osc_b, and osc_n list of the 3 strings"
                        # But typically frontend likes objects. The notebook prints "Name (Score)". 
                        # I will return the objects in a cleaner key if needed, or just the strings.
                        # Let's stick to the list of strings for the "non_automatable_parameters" keys as requested.
                        # BUT I will also add a "matches_details" key just in case.
                    },
                    "matches_details": {
                        "osc_a": match_a,
                        "osc_b": match_b,
                        "osc_n": match_n
                    }
                })

            return {
                "success": True,
                "message": "Success",
                "count": num_outputs,
                "results": results 
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
    
    inference = VAEv2p7OscInference()
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
    print("🧪 Testing VAE V2.7 Oscillator Inference...")
    
    inference = VAEv2p7OscInference()
    
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
        for i, res in enumerate(result['results']):
            print(f"\nOutput {i+1}:")
            # automatable params
            params = res['automatable_parameters']
            # print first 2 params as check
            p0 = params.get('0', {})
            print(f"  Param 0 ({p0.get('name')}): {p0.get('value'):.4f}")
            
            # non-automatable
            non_auto = res['non_automatable_parameters']
            print(f"  Osc A: {non_auto['osc_a']}")
            print(f"  Osc B: {non_auto['osc_b']}")
            print(f"  Noise: {non_auto['osc_n']}")
    else:
        print(f"❌ Failed: {result['message']}")
