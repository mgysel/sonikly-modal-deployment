#!/usr/bin/env python3
"""Benchmark different diffusion step counts with JIT optimization"""

import modal
import time

# Look up the deployed class
VAEv2p7Inference = modal.Cls.from_name("vae-v2p7-inference", "VAEv2p7Inference")
inference = VAEv2p7Inference()

test_cases = [
    ("50 steps", 50),
    ("100 steps", 100),
    ("200 steps", 200),
    ("500 steps", 500),
    ("1000 steps", 1000),
]

description = "deep wobbling bass with heavy modulation"

print("=" * 80)
print("BENCHMARKING VAE V2.7 WITH JIT COMPILATION")
print("=" * 80)
print(f"\nTest prompt: '{description}'")
print(f"Hardware: Tesla T4 GPU")
print("\n" + "-" * 80)

results = []

for name, steps in test_cases:
    print(f"\n{name}...", end=" ", flush=True)
    
    # Warmup call (first call might be slower due to graph tracing)
    if steps == 50:
        print("(includes warmup)", end=" ", flush=True)
    
    start = time.time()
    
    result = inference.generate.remote(
        description=description,
        diffusion_steps=steps,
        seed=42,
        verbose=False,
    )
    
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"✓ {elapsed:.3f}s ({elapsed*1000:.0f}ms)")
        results.append((name, steps, elapsed))
        print(f"  First 5 params: {[f'{x:.3f}' for x in result['parameters'][:5]]}")
    else:
        print(f"✗ Failed: {result['message']}")

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Steps':<10} {'Time':<15} {'ms/step':<15} {'Speedup vs 1000':<15}")
print("-" * 60)

baseline_time = None
for name, steps, elapsed in results:
    if steps == 1000:
        baseline_time = elapsed
    
    ms_per_step = (elapsed * 1000) / steps
    
    if baseline_time and steps != 1000:
        speedup = baseline_time / elapsed
        speedup_str = f"{speedup:.1f}x faster"
    else:
        speedup_str = "baseline"
    
    print(f"{steps:<10} {elapsed:.3f}s{'':<8} {ms_per_step:.2f}ms{'':<8} {speedup_str}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find the sweet spot (< 2 seconds, highest quality)
good_options = [(n, s, t) for n, s, t in results if t < 2.0]
if good_options:
    best = max(good_options, key=lambda x: x[1])  # Highest step count under 2s
    print(f"\nFor production: Use {best[1]} steps (~{best[2]:.2f}s)")
    print(f"  - Fast enough for interactive use")
    print(f"  - Good quality/speed trade-off")

if baseline_time:
    print(f"\nFor maximum quality: Use 1000 steps (~{baseline_time:.2f}s)")
    print(f"  - Best possible quality")
    print(f"  - Good for offline/batch generation")

print("\n" + "=" * 80)
