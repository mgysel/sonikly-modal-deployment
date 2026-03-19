#!/usr/bin/env python3
"""
Test client for VAE V2.7 Latent Diffusion model

This script tests the deployed model using both Modal's Python API and HTTP endpoint.
"""

import modal
import requests
import json


def test_via_modal_api():
    """Test using Modal's Python API"""
    print("=" * 70)
    print("Testing VAE V2.7 via Modal Python API")
    print("=" * 70)
    
    try:
        # Look up the deployed class
        print("\n1. Connecting to Modal deployment...")
        VAEv2p7Inference = modal.Cls.from_name("vae-v2p7-inference", "VAEv2p7Inference")
        
        # Create an instance
        print("2. Creating model instance...")
        inference = VAEv2p7Inference()
        
        # Test health check first
        print("\n3. Testing health check...")
        health = inference.health_check.remote()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        
        # Test generation
        print("\n4. Generating synthesizer parameters...")
        test_descriptions = [
            "deep wobbling bass with heavy modulation",
            "bright plucky synth lead",
            "warm analog pad with slow attack",
        ]
        
        for i, description in enumerate(test_descriptions, 1):
            print(f"\n   Test {i}: '{description}'")
            result = inference.generate.remote(
                description=description,
                diffusion_steps=50,  # Use 50 steps for fast, quality inference
                seed=42,
                verbose=False,
            )
            
            if result["success"]:
                print(f"   ✓ Success!")
                print(f"     - Parameters shape: {result['shape']}")
                print(f"     - First 5 params: {result['parameters'][:5]}")
                print(f"     - Message: {result['message']}")
            else:
                print(f"   ✗ Failed: {result['message']}")
                if 'traceback' in result:
                    print(f"     Traceback:\n{result['traceback']}")
        
        print("\n" + "=" * 70)
        print("✅ Modal API tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the model is deployed:")
        print("  python3 deploy_model.py --model vae_v2p7_latent_diffusion")


def test_via_http(endpoint_url=None):
    """Test using HTTP endpoint"""
    if not endpoint_url:
        print("\n" + "=" * 70)
        print("HTTP Endpoint Testing")
        print("=" * 70)
        print("\nTo test the HTTP endpoint, you need the endpoint URL.")
        print("\nGet it by running:")
        print("  modal app list")
        print("  # Look for the 'generate_web' function URL")
        print("\nThen run:")
        print("  python3 models/vae_v2p7_latent_diffusion/test_client.py --http <URL>")
        return
    
    print("=" * 70)
    print("Testing VAE V2.7 via HTTP Endpoint")
    print("=" * 70)
    print(f"\nEndpoint: {endpoint_url}")
    
    try:
        # Test generation
        print("\n1. Testing generation endpoint...")
        payload = {
            "description": "deep wobbling bass with heavy modulation",
            "diffusion_steps": 1000,
            "seed": 42,
        }
        
        print(f"   Request: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            endpoint_url,
            json=payload,
            timeout=300  # Generation can take time (5 minutes for cold start + inference)
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n   ✓ Success!")
            print(f"   Response: {json.dumps(result, indent=2)}")
        else:
            print(f"\n   ✗ Failed with status {response.status_code}")
            print(f"   Response: {response.text}")
        
        print("\n" + "=" * 70)
        print("✅ HTTP endpoint tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test VAE V2.7 deployment"
    )
    parser.add_argument(
        "--http",
        type=str,
        help="HTTP endpoint URL to test"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only test Modal API (skip HTTP)"
    )
    parser.add_argument(
        "--http-only",
        action="store_true",
        help="Only test HTTP endpoint (skip Modal API)"
    )
    
    args = parser.parse_args()
    
    if args.http_only:
        if not args.http:
            print("❌ Error: --http URL is required when using --http-only")
            return 1
        test_via_http(args.http)
    elif args.api_only:
        test_via_modal_api()
    else:
        # Test both
        test_via_modal_api()
        print("\n" * 2)
        test_via_http(args.http)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
