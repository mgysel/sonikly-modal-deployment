
import sys
import os
import numpy as np

# Add current directory to sys.path
sys.path.append(os.getcwd())

from serum_params import SERUM_PARAMETERS
from vae_v2p7_osc import ParameterUtils

def verify():
    print("Verifying Parameter Counts...")
    
    # 1. Check SERUM_PARAMETERS count
    total_params = sum(len(group) for group in SERUM_PARAMETERS.values())
    print(f"Total entries in SERUM_PARAMETERS: {total_params}")
    
    # Check for forbidden IDs
    flat_ids = [int(p['id']) for g in SERUM_PARAMETERS.values() for p in g]
    if 4 in flat_ids or 24 in flat_ids or 44 in flat_ids:
        print("❌ FAILED: Found forbidden IDs (4, 24, or 44) in parameters!")
    else:
        print("✅ No forbidden IDs found.")
        
    if total_params != 202:
         print(f"❌ FAILED: Expected 202 parameters, found {total_params}")
    else:
         print("✅ Count is correct (202).")

    # 2. Check ParameterUtils
    print("\nVerifying ParameterUtils...")
    info = ParameterUtils.get_indices_and_classes(SERUM_PARAMETERS)
    n_params = info['n_params']
    print(f"ParameterUtils n_params: {n_params}")
    
    # Check max ID
    max_id = max(flat_ids)
    print(f"Max ID in data: {max_id}")
    
    if max_id != 201:
        print(f"❌ FAILED: Expected Max ID 201, found {max_id}")
    else:
        print("✅ Max ID is correct (201).")

    # 3. Simulate Reconstruction
    print("\nSimulating Reconstruction...")
    # Mock heads
    # Unipolar: 202 items roughly
    # We just need to ensure no index out of bounds error
    
    # Create dummy heads matching the structure expected by reconstruct_parameters_from_heads
    # It expects a list of heads.
    # We can just pass random data and see if it crashes.
    
    # Structure from modal_deploy_parallel.py comments:
    # [unipolar, bipolar_gate, bipolar_value, boolean, cat_0...cat_N, osc_a, osc_b, osc_n]
    
    num_heads = 0
    if info['unipolar_indices']: num_heads += 1
    if info['bipolar_indices']: num_heads += 2
    if info['bool_indices']: num_heads += 1
    num_heads += len(info['cat_indices']) # One head per categorical
    num_heads += 3 # Audio heads
    
    print(f"Constructing {num_heads} dummy heads...")
    
    dummy_heads = []
    for _ in range(num_heads):
        # Shape doesn't matter much for the crash test, but let's make it look right
        dummy_heads.append(np.random.rand(1, 10)) # Batch size 1, 10 dims
        
    try:
        rec_params, audio = ParameterUtils.reconstruct_parameters_from_heads(dummy_heads, info)
        print(f"Reconstructed shape: {rec_params.shape}")
        
        if rec_params.shape[0] != 202:
             print(f"❌ FAILED: Reconstructed array length is {rec_params.shape[0]}, expected 202")
        else:
             print("✅ Reconstruction successful with shape (202,).")
             
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
