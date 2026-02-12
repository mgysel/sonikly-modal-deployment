# Testing the VAE V2.7 Deployment

## First, deploy the model

```bash
python3 deploy_model.py --model vae_v2p7_latent_diffusion
```

Wait for the deployment to complete successfully.

## Test via Modal Python API

The easiest way to test:

```bash
python3 models/vae_v2p7_latent_diffusion/test_client.py
```

Or test just the API:

```bash
python3 models/vae_v2p7_latent_diffusion/test_client.py --api-only
```

## Test via HTTP Endpoint

First, get your endpoint URL:

```bash
modal app list
```

Look for the `generate_web` function URL under the `vae-v2p7-inference` app.

Then test it:

```bash
python3 models/vae_v2p7_latent_diffusion/test_client.py --http https://your-endpoint-url.modal.run
```

## Example Response

A successful response looks like:

```json
{
  "success": true,
  "message": "Success",
  "parameters": [0.1234, 0.5678, ...],
  "shape": [202]
}
```

The `parameters` array contains 202 floats representing the synthesizer parameters.

## Quick Test via Modal Built-in

The deployment script also has a built-in test:

```bash
modal run models/vae_v2p7_latent_diffusion/modal/modal_deploy.py
```

## Test with curl

Once you have the HTTP endpoint URL:

```bash
curl -X POST https://your-endpoint-url.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "description": "deep wobbling bass with heavy modulation",
    "diffusion_steps": 1000,
    "seed": 42
  }'
```
