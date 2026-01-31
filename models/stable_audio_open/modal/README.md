# Stable Audio Open - Modal Deployment

Text-to-audio generation using Stability AI's Stable Audio Open model.

## Quick Start

### Deploy the Model

```bash
modal deploy models/stable_audio_open/modal/modal_deploy.py
```

### Generate Audio

```bash
# Basic usage
modal run models/stable_audio_open/modal/modal_deploy.py --prompt "A peaceful piano melody"

# Advanced usage
modal run models/stable_audio_open/modal/modal_deploy.py \
  --prompt "Upbeat electronic music with synthesizers" \
  --duration 15 \
  --steps 150 \
  --cfg-scale 7.5 \
  --seed 42 \
  --output-path output.wav
```

## Parameters

- `--prompt`: Text description of the audio to generate (required)
- `--duration`: Duration in seconds (default: 10.0, max: 47.0)
- `--steps`: Number of diffusion steps (default: 100, higher = better quality)
- `--cfg-scale`: Classifier-free guidance scale (default: 7.0, higher = more prompt adherence)
- `--seed`: Random seed for reproducibility (default: -1 for random)
- `--output-path`: Output file path (default: output.wav)

## Python API

```python
import modal

# Look up the deployed model
StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")

# Generate audio
model = StableAudioModel()
audio_bytes = model.generate.remote(
    prompt="A serene piano melody with ambient background",
    duration=10.0,
    steps=100,
    cfg_scale=7.0,
    seed=42
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

## Model Information

- **Model**: [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- **Sample Rate**: 44.1 kHz
- **Max Duration**: 47 seconds
- **License**: Stability AI Community License

## GPU Requirements

- **Default**: A10G GPU (sufficient for most use cases)
- **Alternative**: Can be upgraded to A100 for faster generation
- **Memory**: ~10GB VRAM required

## Cost Estimate

Based on Modal's GPU pricing:
- **A10G**: $0.000306/second
- Typical generation (10-second audio): ~$0.006 per generation

First run includes model download time (~30-60 seconds), subsequent runs are faster.

## Test the Setup

```bash
# Test imports
modal run models/stable_audio_open/modal/modal_deploy.py::test_imports
```
