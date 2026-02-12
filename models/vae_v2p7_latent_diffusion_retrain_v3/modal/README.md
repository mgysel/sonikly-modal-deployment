# Modal Deployment for VAE V2.7

## Quick Start

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate
modal setup

# 3. Upload model weights
python modal_deploy.py --upload

# 4. Deploy
modal deploy modal_deploy.py

# 5. Test
modal run modal_deploy.py
```

## Usage

### From Python

```python
import modal

# Lookup deployed function
model = modal.Function.lookup("vae-v2p7-inference", "VAEv2p7Inference")

# Generate parameters
result = model.generate.remote(
    description="deep bass",
    diffusion_steps=1000,
    seed=42,
)

if result["success"]:
    params = result["parameters"]  # List of 202 floats
```

### From HTTP

```bash
curl -X POST https://your-app.modal.run/generate_web \
  -H "Content-Type: application/json" \
  -d '{"description": "deep bass", "diffusion_steps": 1000}'
```

## Configuration

Edit `modal_deploy.py`:
- `GPU_TYPE`: "T4" (cheap), "A10G" (balanced), "A100" (fast)
- `timeout`: Max generation time (default: 300s)
- `container_idle_timeout`: Keep warm time (default: 120s)

## Monitoring

```bash
modal app logs vae-v2p7-inference  # View logs
modal app list                      # List apps
```

## Costs

- T4 GPU: ~$0.60/hour when running
- Idle containers: Free
- Cold start: ~30-60 seconds
- Warm start: <1 second
