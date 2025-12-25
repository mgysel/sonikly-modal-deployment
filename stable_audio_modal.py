"""
Stable Audio Open deployment on Modal
Generates audio from text prompts using Stability AI's Stable Audio Open model
"""

import modal
import io
from pathlib import Path

# Define the Modal app
app = modal.App("stable-audio-open")

# Create a custom image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "einops",
        "numpy<2",
        "scipy",
        "soundfile",
        "huggingface_hub[hf_transfer]==0.27.1",
        "transformers",
        "accelerate",
        "safetensors",
    )
    .pip_install(
        "git+https://github.com/Stability-AI/stable-audio-tools.git"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Create a volume to cache the model weights
model_volume = modal.Volume.from_name("stable-audio-cache", create_if_missing=True)
MODEL_CACHE_DIR = "/cache"


@app.cls(
    image=image,
    gpu="A10G",  # A10G is sufficient for inference, can upgrade to A100 if needed
    timeout=600,
    volumes={MODEL_CACHE_DIR: model_volume},
    scaledown_window=2,  # Keep container alive for 5 seconds after last request
    secrets=[modal.Secret.from_name("huggingface")],  # Add HuggingFace authentication
)
class StableAudioModel:
    """
    Stable Audio Open model class for generating audio from text prompts
    """

    @modal.enter()
    def load_model(self):
        """Load the model when container starts"""
        import torch
        import os
        from stable_audio_tools import get_pretrained_model
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        
        print("Loading Stable Audio Open model...")
        
        # Set HuggingFace cache directory
        os.environ['HF_HOME'] = MODEL_CACHE_DIR
        os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
        
        # Load the model (cache_dir is handled via environment variables)
        self.model, self.model_config = get_pretrained_model(
            "stabilityai/stable-audio-open-1.0"
        )
        
        # Move model to GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Store the generation function
        self.generate_fn = generate_diffusion_cond
        
        print(f"Model loaded successfully on {self.device}")

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        duration: float = 10.0,
        steps: int = 100,
        cfg_scale: float = 7.0,
        seed: int = -1,
    ) -> bytes:
        """
        Generate audio from a text prompt
        
        Args:
            prompt: Text description of the audio to generate
            negative_prompt: Text description of what to avoid in the generation
            duration: Duration of audio in seconds (max 47 seconds for stable-audio-open-1.0)
            steps: Number of diffusion steps (higher = better quality but slower)
            cfg_scale: Classifier-free guidance scale (higher = more prompt adherence)
            seed: Random seed for reproducibility (-1 for random)
        
        Returns:
            Audio file as bytes (WAV format)
        """
        import torch
        import soundfile as sf
        
        # Set random seed if specified
        if seed != -1:
            torch.manual_seed(seed)
        
        # Prepare conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration
        }]
        
        print(f"Generating audio for prompt: '{prompt}'")
        print(f"Duration: {duration}s, Steps: {steps}, CFG Scale: {cfg_scale}")
        
        # Generate audio using stable-audio-tools generation function
        # generate_diffusion_cond handles both latent generation and decoding internally
        with torch.no_grad():
            output = self.generate_fn(
                self.model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_size=int(duration * self.model_config["sample_rate"]),
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                device=self.device
            )
        
        # Convert to numpy and get first batch item
        output = output.cpu().numpy()[0]
        
        # Normalize audio to prevent clipping
        max_val = max(abs(output.max()), abs(output.min()))
        if max_val > 0:
            output = output / max_val
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        sf.write(
            buffer,
            output.T,  # Transpose to (samples, channels)
            self.model_config["sample_rate"],
            format="WAV"
        )
        buffer.seek(0)
        
        print("Audio generation complete!")
        return buffer.read()


@app.local_entrypoint()
def main(
    prompt: str = "A serene piano melody with ambient background sounds",
    duration: float = 10.0,
    output_path: str = "output.wav",
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = -1,
):
    """
    Generate audio from command line
    
    Example usage:
        modal run stable_audio_modal.py --prompt "A peaceful guitar melody"
        modal run stable_audio_modal.py --prompt "Upbeat electronic music" --duration 15 --steps 150
    """
    print(f"Generating audio with prompt: '{prompt}'")
    
    # Generate audio
    model = StableAudioModel()
    audio_bytes = model.generate.remote(
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
    )
    
    # Save to file
    output_file = Path(output_path)
    output_file.write_bytes(audio_bytes)
    
    print(f"✓ Audio saved to: {output_path}")
    print(f"  Duration: {duration} seconds")
    print(f"  Steps: {steps}")
    print(f"  CFG Scale: {cfg_scale}")


@app.function(image=image)
def test_imports():
    """Test that all imports work correctly"""
    try:
        import torch
        import torchaudio  # noqa: F401
        import soundfile as sf  # noqa: F401
        from stable_audio_tools import get_pretrained_model  # noqa: F401
        print("✓ All imports successful!")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

