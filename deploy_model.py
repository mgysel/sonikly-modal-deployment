#!/usr/bin/env python3
"""
Unified deployment script for all models in the models/ directory

Usage:
    python deploy_model.py --list                           # List available models
    python deploy_model.py --model stable_audio_open        # Deploy Stable Audio Open
    python deploy_model.py --model vae_v2p7_latent_diffusion # Deploy VAE v2.7
    python deploy_model.py --model stable_audio_open --test # Deploy and test
"""

import sys
import subprocess
from pathlib import Path
import argparse


# Model registry - maps model names to their deployment scripts
AVAILABLE_MODELS = {
    "stable_audio_open": {
        "name": "Stable Audio Open",
        "description": "Text-to-audio generation using Stability AI's model",
        "deploy_script": "models/stable_audio_open/modal/modal_deploy.py",
        "modal_app_name": "stable-audio-open",
    },
    "vae_v2p7_latent_diffusion": {
        "name": "VAE V2.7 Latent Diffusion",
        "description": "Synthesizer parameter generation from text descriptions",
        "deploy_script": "models/vae_v2p7_latent_diffusion/modal/modal_deploy.py",
        "modal_app_name": "vae-v2p7-inference",
    },
}


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def list_models():
    """List all available models"""
    print_header("Available Models")
    print()
    for model_id, info in AVAILABLE_MODELS.items():
        print(f"📦 {model_id}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Deploy script: {info['deploy_script']}")
        print()


def check_model_exists(model_id):
    """Check if a model deployment script exists"""
    if model_id not in AVAILABLE_MODELS:
        print(f"❌ Error: Model '{model_id}' not found")
        print("\nAvailable models:")
        for mid in AVAILABLE_MODELS.keys():
            print(f"  - {mid}")
        return False
    
    model_info = AVAILABLE_MODELS[model_id]
    deploy_script = Path(model_info["deploy_script"])
    
    if not deploy_script.exists():
        print(f"❌ Error: Deploy script not found: {deploy_script}")
        return False
    
    return True


def deploy_model(model_id, test=False):
    """Deploy a model to Modal"""
    if not check_model_exists(model_id):
        return False
    
    model_info = AVAILABLE_MODELS[model_id]
    deploy_script = model_info["deploy_script"]
    
    print_header(f"Deploying {model_info['name']}")
    print(f"\n📋 Model: {model_id}")
    print(f"📄 Script: {deploy_script}")
    print(f"🚀 Modal App: {model_info['modal_app_name']}")
    print()
    
    try:
        # Deploy the model
        print("⏳ Deploying to Modal...")
        result = subprocess.run(
            ["modal", "deploy", deploy_script],
            check=True,
            capture_output=False,
            text=True
        )
        
        print("\n✅ Deployment successful!")
        
        # Run tests if requested
        if test:
            print("\n" + "-" * 70)
            print("🧪 Running tests...")
            print("-" * 70)
            test_result = subprocess.run(
                ["modal", "run", deploy_script],
                check=False,
                capture_output=False,
                text=True
            )
            if test_result.returncode == 0:
                print("\n✅ Tests passed!")
            else:
                print("\n⚠️  Tests failed (see output above)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print("\n❌ Error: Modal CLI not found. Please install it:")
        print("   pip install modal")
        print("   python3 -m modal setup")
        return False


def run_model_test(model_id):
    """Run tests for a deployed model"""
    if not check_model_exists(model_id):
        return False
    
    model_info = AVAILABLE_MODELS[model_id]
    deploy_script = model_info["deploy_script"]
    
    print_header(f"Testing {model_info['name']}")
    print()
    
    try:
        result = subprocess.run(
            ["modal", "run", deploy_script],
            check=True,
            capture_output=False,
            text=True
        )
        print("\n✅ Tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy AI models to Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python deploy_model.py --list
  
  # Deploy Stable Audio Open
  python deploy_model.py --model stable_audio_open
  
  # Deploy and test VAE v2.7
  python deploy_model.py --model vae_v2p7_latent_diffusion --test
  
  # Test a deployed model
  python deploy_model.py --model stable_audio_open --test-only
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model to deploy (e.g., stable_audio_open, vae_v2p7_latent_diffusion)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after deployment"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests (don't deploy)"
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_models()
        return 0
    
    # Require --model for other operations
    if not args.model:
        parser.print_help()
        print("\n❌ Error: Please specify a model with --model or use --list to see available models")
        return 1
    
    # Handle --test-only
    if args.test_only:
        success = run_model_test(args.model)
        return 0 if success else 1
    
    # Deploy the model
    success = deploy_model(args.model, test=args.test)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
