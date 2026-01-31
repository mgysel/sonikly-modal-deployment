# Sonikly - Modal Deployment

Deploy AI models for audio generation and synthesis on Modal.

## 🎯 Overview

This repository contains Modal deployments for various AI models:

- **Stable Audio Open**: Text-to-audio generation using Stability AI's model
- **VAE V2.7 Latent Diffusion**: Synthesizer parameter generation from text descriptions

## 🚀 Quick Start

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Python 3.8+**: Ensure you have Python installed

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd sonikly-modal-deployment
```

2. Install Modal:
```bash
pip install modal
```

3. Authenticate with Modal:
```bash
python3 -m modal setup
```

## 📦 Available Models

List all available models:
```bash
python deploy_model.py --list
```

### Stable Audio Open

Text-to-audio generation model that creates high-quality audio from text prompts.

**Deploy:**
```bash
python deploy_model.py --model stable_audio_open
```

**Generate audio:**
```bash
modal run models/stable_audio_open/modal/modal_deploy.py --prompt "A peaceful piano melody"
```

**Documentation:** See [models/stable_audio_open/modal/README.md](models/stable_audio_open/modal/README.md)

### VAE V2.7 Latent Diffusion

Synthesizer parameter generation model that creates Serum synthesizer parameters from text descriptions.

**Deploy:**
```bash
python deploy_model.py --model vae_v2p7_latent_diffusion
```

**Generate parameters:**
```bash
modal run models/vae_v2p7_latent_diffusion/modal/modal_deploy.py
```

**Documentation:** See [models/vae_v2p7_latent_diffusion/modal/README.md](models/vae_v2p7_latent_diffusion/modal/README.md)

## 🔧 Deployment Commands

### Deploy a specific model
```bash
python deploy_model.py --model <model_name>
```

### Deploy and test
```bash
python deploy_model.py --model <model_name> --test
```

### Test a deployed model
```bash
python deploy_model.py --model <model_name> --test-only
```

## 📁 Project Structure

```
sonikly-modal-deployment/
├── models/
│   ├── stable_audio_open/          # Stable Audio Open model
│   │   ├── __init__.py
│   │   └── modal/
│   │       ├── modal_deploy.py     # Modal deployment script
│   │       └── README.md           # Model-specific docs
│   └── vae_v2p7_latent_diffusion/  # VAE V2.7 model
│       ├── __init__.py
│       ├── vae_v2p7.py            # Model implementation
│       └── modal/
│           ├── modal_deploy.py     # Modal deployment script
│           └── README.md           # Model-specific docs
├── deploy_model.py                 # Unified deployment script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎨 Adding New Models

To add a new model to the deployment system:

1. **Create the model directory:**
```bash
mkdir -p models/your_model_name/modal
```

2. **Add your model implementation:**
```
models/your_model_name/
├── __init__.py              # Model package
├── your_model.py            # Model code
└── modal/
    ├── modal_deploy.py      # Modal deployment script
    └── README.md            # Documentation
```

3. **Register in deploy_model.py:**

Edit `deploy_model.py` and add your model to the `AVAILABLE_MODELS` dictionary:

```python
AVAILABLE_MODELS = {
    # ... existing models ...
    "your_model_name": {
        "name": "Your Model Name",
        "description": "Brief description of what your model does",
        "deploy_script": "models/your_model_name/modal/modal_deploy.py",
        "modal_app_name": "your-modal-app-name",
    },
}
```

4. **Deploy your model:**
```bash
python deploy_model.py --model your_model_name
```

## 💡 Usage Examples

### Python API

```python
import modal

# Stable Audio Open
StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")
model = StableAudioModel()
audio_bytes = model.generate.remote(
    prompt="A serene piano melody",
    duration=10.0
)

# VAE V2.7
VAEv2p7Inference = modal.Cls.lookup("vae-v2p7-inference", "VAEv2p7Inference")
inference = VAEv2p7Inference()
result = inference.generate.remote(
    description="deep bass with modulation"
)
```

## 💰 Cost Estimates

Based on Modal's GPU pricing:

### Stable Audio Open (A10G GPU)
- **Rate**: $0.000306/second
- **Typical generation** (10-second audio): ~$0.006 per generation

### VAE V2.7 (T4 GPU)
- **Rate**: ~$0.000150/second
- **Typical generation**: ~$0.003-0.005 per generation

First run includes model download time, subsequent runs are faster due to caching.

## 🛠️ Troubleshooting

### Modal Authentication Issues
```bash
python3 -m modal setup
```

### Import Errors
Test the model setup:
```bash
python deploy_model.py --model <model_name> --test-only
```

### Out of Memory
- Reduce batch size or duration
- Upgrade to a more powerful GPU in the deployment script

## 📚 Resources

- [Modal Documentation](https://modal.com/docs)
- [Stable Audio Open Model Card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Stability AI](https://stability.ai/)

## 📄 License

This deployment code is provided as-is. Individual models are subject to their respective licenses:
- Stable Audio Open: [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE)
- VAE V2.7: Check model-specific license

## 🤝 Support

For issues with:
- **Modal deployment**: Check [Modal documentation](https://modal.com/docs) or [Slack community](https://modal.com/slack)
- **Model-specific issues**: See individual model READMEs in the `models/` directory
