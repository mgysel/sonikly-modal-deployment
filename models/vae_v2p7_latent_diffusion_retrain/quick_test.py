#!/usr/bin/env python3
"""Quick test with different diffusion step counts"""

import modal
import time

# Look up the deployed class
VAEv2p7Inference = modal.Cls.from_name("vae-v2p7-inference", "VAEv2p7Inference")
inference = VAEv2p7Inference()

test_cases = [
    ("50 steps", 50),
    ("100 steps", 100),
    ("200 steps", 200),
]

description = "deep wobbling bass"

print("Testing different diffusion step counts:")
print("=" * 70)

for name, steps in test_cases:
    print(f"\n{name}...")
    start = time.time()
    
    result = inference.generate.remote(
        description=description,
        diffusion_steps=steps,
        seed=42,
    )
    
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"  ✓ Success in {elapsed:.2f}s")
        print(f"    First 5 params: {result['parameters'][:5]}")
    else:
        print(f"  ✗ Failed: {result['message']}")

print("\n" + "=" * 70)
print("Done! Compare the quality vs speed trade-off.")
