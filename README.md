# Stable Audio Open - Modal Deployment

This repository contains a Modal deployment for [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0), Stability AI's open-source text-to-audio generation model.

## Features

- 🎵 Generate high-quality audio from text prompts
- ⚡ Serverless GPU inference with Modal
- 💾 Automatic model caching for fast cold starts
- 🎛️ Configurable generation parameters (steps, CFG scale, duration, seed)
- 🔧 Easy-to-use CLI and Python API

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Python 3.8+**: Ensure you have Python installed

## Installation

1. **Clone this repository**:
```bash
git clone <your-repo-url>
cd sonikly-modal-deployment
```

2. **Install Modal**:
```bash
pip install modal
```

3. **Authenticate with Modal**:
```bash
python3 -m modal setup
```
This will open a browser window for authentication.

## Usage

### Command Line Interface

Generate audio directly from the command line:

```bash
# Basic usage with default settings
modal run stable_audio_modal.py

# Custom prompt
modal run stable_audio_modal.py --prompt "A peaceful guitar melody with ocean waves"

# Advanced usage with all parameters
modal run stable_audio_modal.py \
  --prompt "Upbeat electronic dance music with synthesizers" \
  --duration 15 \
  --steps 150 \
  --cfg-scale 7.5 \
  --seed 42 \
  --output-path my_audio.wav
```

### Parameters

- `--prompt`: Text description of the audio to generate (required)
- `--duration`: Duration in seconds (default: 10.0, max: 47.0)
- `--steps`: Number of diffusion steps (default: 100, higher = better quality)
- `--cfg-scale`: Classifier-free guidance scale (default: 7.0, higher = more prompt adherence)
- `--seed`: Random seed for reproducibility (default: -1 for random)
- `--output-path`: Output file path (default: output.wav)

### Python API

Use the deployment programmatically:

```python
import modal

# Deploy the model
app = modal.App.lookup("stable-audio-open", create_if_missing=False)
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

See `example_client.py` for a complete example.

## Deployment

### Deploy as a Web Endpoint

To expose the model as a web API:

```bash
modal deploy stable_audio_modal.py
```

This will create a persistent deployment that you can call via HTTP.

### Test the Deployment

Test that all dependencies are correctly installed:

```bash
modal run stable_audio_modal.py::test_imports
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

## Cost Optimization

- Model weights are cached in a Modal Volume for fast cold starts
- Container idle timeout is set to 5 minutes to reduce costs
- Uses HF Transfer for faster model downloads

### Pricing Estimate

Based on Modal's GPU pricing:
- **A10G**: $0.000306/second
- **A100**: $0.000583/second

Typical generation (10-second audio, ~15-20 seconds compute):
- A10G: ~$0.006 per generation
- A100: ~$0.009 per generation

First run includes model download time (~30-60 seconds), subsequent runs are faster.

## Troubleshooting

### Import Errors

If you encounter import errors, run the test function:
```bash
modal run stable_audio_modal.py::test_imports
```

### Out of Memory

If you run out of GPU memory:
1. Reduce the duration
2. Reduce the number of steps
3. Upgrade to A100 GPU in `stable_audio_modal.py`

### Slow Generation

If generation is too slow:
1. Reduce the number of steps (try 50-75)
2. Use a more powerful GPU (A100)

## Examples

### Music Generation

```bash
# Classical music
modal run stable_audio_modal.py --prompt "A classical piano sonata in the style of Mozart"

# Electronic music
modal run stable_audio_modal.py --prompt "Energetic techno beat with heavy bass"

# Ambient soundscape
modal run stable_audio_modal.py --prompt "Peaceful forest ambience with birds chirping"
```

### Sound Effects

```bash
# Nature sounds
modal run stable_audio_modal.py --prompt "Ocean waves crashing on a beach" --duration 15

# Urban sounds
modal run stable_audio_modal.py --prompt "Busy city street with traffic and people talking"

# Sci-fi effects
modal run stable_audio_modal.py --prompt "Futuristic spaceship engine humming"
```

## License

This deployment code is provided as-is. The Stable Audio Open model is subject to the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE).

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Stable Audio Open Model Card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Stability AI](https://stability.ai/)

## Support

For issues with:
- **Modal deployment**: Check [Modal documentation](https://modal.com/docs) or [Slack community](https://modal.com/slack)
- **Model behavior**: See [Stable Audio Open repository](https://github.com/Stability-AI/stable-audio-tools)

