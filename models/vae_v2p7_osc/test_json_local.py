
import sys
import os
import numpy as np
import json

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock tensorflow/keras modules BEFORE importing vae_v2p7_osc
from unittest.mock import MagicMock
sys.modules["tensorflow"] = MagicMock()
sys.modules["tensorflow.keras"] = MagicMock()
sys.modules["tensorflow.keras.backend"] = MagicMock()
sys.modules["tensorflow.keras.layers"] = MagicMock()
sys.modules["tensorflow.keras.models"] = MagicMock()
sys.modules["tensorflow.keras.utils"] = MagicMock()
sys.modules["keras.saving"] = MagicMock()

from vae_v2p7_osc import ParameterUtils, numpy_to_json
from serum_params import SERUM_PARAMETERS

def test_json_reconstruction():
    print("🧪 Testing JSON Reconstruction Logic...")
    
    # 1. Get Params Info
    param_info = ParameterUtils.get_indices_and_classes(SERUM_PARAMETERS)
    print(f"  - Total Params: {param_info['n_params']}")
    print(f"  - Unipolar: {len(param_info['unipolar_indices'])}")
    print(f"  - Bipolar: {len(param_info['bipolar_indices'])}")
    print(f"  - Boolean: {len(param_info['bool_indices'])}")
    print(f"  - Categorical: {len(param_info['cat_indices'])}")
    
    # 2. Mock Heads
    # We need to simulate the list of outputs the model would produce
    mock_heads = []
    
    # Unipolar Head
    n_uni = len(param_info['unipolar_indices'])
    mock_heads.append(np.random.rand(n_uni).astype(np.float32))
    
    # Bipolar Heads (Gate + Value)
    n_bi = len(param_info['bipolar_indices'])
    mock_heads.append(np.random.rand(n_bi).astype(np.float32)) # Gate
    mock_heads.append(np.random.rand(n_bi).astype(np.float32)) # Value
    
    # Boolean Head
    n_bool = len(param_info['bool_indices'])
    mock_heads.append(np.random.rand(n_bool).astype(np.float32))
    
    # Categorical Heads
    for idx in param_info['cat_indices']:
        n_classes = param_info['categorical_num_classes'][idx]
        mock_heads.append(np.random.rand(n_classes).astype(np.float32))
        
    # Audio Heads (Last 3)
    mock_heads.append(np.random.rand(512).astype(np.float32)) # Osc A
    mock_heads.append(np.random.rand(512).astype(np.float32)) # Osc B
    mock_heads.append(np.random.rand(512).astype(np.float32)) # Noise
    
    print(f"  - Generated {len(mock_heads)} mock heads")
    
    # 3. Run Reconstruction
    rec_params, audio_vecs = ParameterUtils.reconstruct_parameters_from_heads(
        mock_heads, 
        param_info
    )
    
    print(f"  - Reconstructed params shape: {rec_params.shape}")
    assert rec_params.shape[0] >= 200, "Reconstructed params too small"
    
    print(f"  - Audio keys: {list(audio_vecs.keys())}")
    assert "osc_a" in audio_vecs
    assert "osc_b" in audio_vecs
    assert "noise" in audio_vecs
    
    # 4. JSON Conversion (Mocking the deployment logic, not numpy_to_json)
    # The deployment now returns flat arrays, so we just check the reconstructed array directly
    # and the structure we intend to build in the deployment script.
    
    print(f"  - Reconstructed params: {rec_params.tolist()[:5]}...")
    assert isinstance(rec_params.tolist(), list)
    assert len(rec_params.tolist()) == 205
    
    # We no longer use numpy_to_json for the main output
    # but we can check if audio_vecs matches expectations
    print(f"  - Audio Match Keys: {list(audio_vecs.keys())}")
    
    print("✅ Logic Verified (Top-level aggregation happens in modal_deploy_parallel.py)")

if __name__ == "__main__":
    try:
        test_json_reconstruction()
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
