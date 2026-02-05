
import os
import sys
import numpy as np

# Add local path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modal.modal_deploy_parallel import VAEv2p7OscInference

def test_matching_logic():
    print("🧪 Testing Wavetable Matching Logic...")
    
    # Mock data paths
    wt_dir = os.path.join(os.path.dirname(__file__), "wavetables")
    
    # Load actual .npy files
    print(f"Loading libraries from {wt_dir}...")
    wt_names = np.load(os.path.join(wt_dir, 'default_wavetable_names.npy'), allow_pickle=True)
    wt_embeds = np.load(os.path.join(wt_dir, 'default_wavetable_embeddings.npy'), allow_pickle=True).astype('float32')
    
    # Mock an Inference instance (just the needed attributes)
    inference = VAEv2p7OscInference()
    
    # Manually set attributes that load_model would set
    inference.wt_names = wt_names
    inference.wt_embeds = wt_embeds / np.linalg.norm(wt_embeds, axis=1, keepdims=True)
    
    # Create a dummy target vector (e.g., similar to the first wavetable)
    target = wt_embeds[0] + np.random.normal(0, 0.01, size=wt_embeds[0].shape)
    
    # Run matching
    matches = inference.find_nearest_matches(target.astype('float32'), inference.wt_embeds, inference.wt_names)
    
    print("\nTarget should match:", wt_names[0])
    print("Matches found:")
    for m in matches:
        print(f"  - {m['name']} (score: {m['score']:.4f})")
        
    # Verify top match is correct
    assert matches[0]['name'] == wt_names[0]
    print("\n✅ Matching logic verified!")

if __name__ == "__main__":
    try:
        test_matching_logic()
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
