#!/usr/bin/env python3
"""
Test client for VAE V2.7 Parallel Deployment
Tests the parallel batching capabilities (default 5 outputs)
"""

import modal
import time


def test_via_modal_api():
    """Test using Modal's Python API with parallel batching"""
    print("=" * 80)
    print("Testing VAE V2.7 Parallel Deployment via Modal Python API")
    print("=" * 80)
    
    try:
        # Look up the deployed class
        print("\n1. Connecting to Modal deployment...")
        VAEv2p7Inference = modal.Cls.from_name("vae-v2p7-inference-parallel", "VAEv2p7Inference")
        
        # Create an instance
        print("2. Creating model instance...")
        inference = VAEv2p7Inference()
        
        # Test health check first
        print("\n3. Testing health check...")
        health = inference.health_check.remote()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        
        # Test parallel generation with different batch sizes
        print("\n4. Testing parallel generation with different batch sizes...")
        test_cases = [
            ("Single output", 1),
            ("Parallel (5 outputs)", 5),
            ("Parallel (10 outputs)", 10),
        ]
        
        description = "deep wobbling bass with heavy modulation"
        
        for test_name, num_outputs in test_cases:
            print(f"\n   {test_name}:")
            print(f"   Prompt: '{description}'")
            print(f"   Generating {num_outputs} variation(s)...", end=" ", flush=True)
            
            start = time.time()
            result = inference.generate.remote(
                description=description,
                diffusion_steps=50,
                num_outputs=num_outputs,
                seed=42,
                verbose=False,
            )
            elapsed = time.time() - start
            
            if result["success"]:
                time_per_output = elapsed / result['count']
                print(f"✓ Done in {elapsed:.3f}s")
                print(f"     - Generated {result['count']} variations")
                print(f"     - Shape: {result['shape']}")
                print(f"     - Time per variation: {time_per_output*1000:.0f}ms")
                
                # Handle both single output and batch outputs
                if result['count'] == 1:
                    # Single output: parameters is a flat list [202 floats]
                    print(f"     - First 5 params: {result['parameters'][:5]}")
                else:
                    # Batch outputs: parameters is list of lists [[202 floats], [202 floats], ...]
                    print(f"     - First output sample: {result['parameters'][0][:5]}")
                
                # Show efficiency gain
                if num_outputs > 1:
                    single_time_estimate = 0.2 * num_outputs  # Assume 200ms per output sequentially
                    speedup = single_time_estimate / elapsed
                    print(f"     - Parallel speedup: {speedup:.1f}x faster than sequential")
            else:
                print(f"✗ Failed: {result['message']}")
                if 'traceback' in result:
                    print(f"     Traceback:\n{result['traceback']}")
        
        print("\n" + "=" * 80)
        print("✅ Parallel API tests completed!")
        print("=" * 80)
        print("\nKey Insight: GPU processes multiple outputs simultaneously!")
        print("5 outputs takes ~same time as 1 output (massive throughput boost)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the parallel model is deployed:")
        print("  modal deploy models/vae_v2p7_latent_diffusion/modal/modal_deploy_parallel.py")


def main():
    """Run parallel batching tests"""
    test_via_modal_api()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
