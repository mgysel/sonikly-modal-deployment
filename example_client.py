"""
Example client for using the Stable Audio Open Modal deployment

This script demonstrates how to use the deployed model programmatically.
"""

import modal
from pathlib import Path


def generate_audio_example():
    """
    Basic example of generating audio using the Modal deployment
    """
    print("=" * 60)
    print("Stable Audio Open - Example Client")
    print("=" * 60)
    
    # Look up the deployed app
    print("\n1. Connecting to Modal deployment...")
    StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")
    
    # Create an instance
    print("2. Creating model instance...")
    model = StableAudioModel()
    
    # Generate audio with default settings
    print("\n3. Generating audio (this may take a minute)...")
    prompt = "A serene piano melody with ambient background sounds"
    print(f"   Prompt: '{prompt}'")
    
    audio_bytes = model.generate.remote(
        prompt=prompt,
        duration=10.0,
        steps=100,
        cfg_scale=7.0,
        seed=42  # Use seed for reproducibility
    )
    
    # Save the audio
    output_path = Path("example_output.wav")
    output_path.write_bytes(audio_bytes)
    
    print(f"\n✓ Audio generated successfully!")
    print(f"  Saved to: {output_path}")
    print(f"  Size: {len(audio_bytes) / 1024:.1f} KB")


def generate_multiple_variations():
    """
    Generate multiple variations of the same prompt with different seeds
    """
    print("\n" + "=" * 60)
    print("Generating Multiple Variations")
    print("=" * 60)
    
    StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")
    model = StableAudioModel()
    
    prompt = "Upbeat electronic music with synthesizers"
    seeds = [42, 123, 456]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {len(seeds)} variations with different seeds...")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n  Variation {i} (seed={seed})...")
        
        audio_bytes = model.generate.remote(
            prompt=prompt,
            duration=8.0,
            steps=75,  # Fewer steps for faster generation
            cfg_scale=7.0,
            seed=seed
        )
        
        output_path = Path(f"variation_{seed}.wav")
        output_path.write_bytes(audio_bytes)
        print(f"    ✓ Saved to: {output_path}")


def generate_with_different_parameters():
    """
    Demonstrate the effect of different generation parameters
    """
    print("\n" + "=" * 60)
    print("Testing Different Parameters")
    print("=" * 60)
    
    StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")
    model = StableAudioModel()
    
    prompt = "A peaceful guitar melody"
    
    # Test different CFG scales
    cfg_scales = [3.0, 7.0, 12.0]
    
    print(f"\nPrompt: '{prompt}'")
    print("Testing different CFG scales (guidance strength)...")
    
    for cfg in cfg_scales:
        print(f"\n  CFG Scale: {cfg}...")
        
        audio_bytes = model.generate.remote(
            prompt=prompt,
            duration=6.0,
            steps=75,
            cfg_scale=cfg,
            seed=42  # Same seed for comparison
        )
        
        output_path = Path(f"cfg_{cfg}.wav")
        output_path.write_bytes(audio_bytes)
        print(f"    ✓ Saved to: {output_path}")
    
    print("\n  Note: Higher CFG scale = stronger adherence to prompt")


def batch_generation():
    """
    Generate multiple audio files in batch
    """
    print("\n" + "=" * 60)
    print("Batch Generation")
    print("=" * 60)
    
    StableAudioModel = modal.Cls.lookup("stable-audio-open", "StableAudioModel")
    model = StableAudioModel()
    
    prompts = [
        "A classical piano sonata",
        "Energetic rock guitar riff",
        "Peaceful forest ambience with birds",
        "Futuristic electronic soundscape",
    ]
    
    print(f"\nGenerating {len(prompts)} audio files...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  {i}/{len(prompts)}: '{prompt}'...")
        
        audio_bytes = model.generate.remote(
            prompt=prompt,
            duration=8.0,
            steps=75,
            cfg_scale=7.0,
        )
        
        # Create safe filename from prompt
        safe_name = prompt.lower().replace(" ", "_")[:30]
        output_path = Path(f"batch_{i}_{safe_name}.wav")
        output_path.write_bytes(audio_bytes)
        print(f"    ✓ Saved to: {output_path}")


def main():
    """
    Run all examples
    """
    try:
        # Example 1: Basic generation
        generate_audio_example()
        
        # Example 2: Multiple variations
        generate_multiple_variations()
        
        # Example 3: Different parameters
        generate_with_different_parameters()
        
        # Example 4: Batch generation
        batch_generation()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully! 🎉")
        print("=" * 60)
        print("\nGenerated files:")
        for wav_file in sorted(Path(".").glob("*.wav")):
            size_kb = wav_file.stat().st_size / 1024
            print(f"  - {wav_file.name} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Deployed the model: modal deploy stable_audio_modal.py")
        print("  2. Authenticated with Modal: python3 -m modal setup")


if __name__ == "__main__":
    main()

