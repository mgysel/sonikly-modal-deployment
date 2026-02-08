
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

from serum_params import SERUM_PARAMETERS

def verify():
    print("Verifying Parameter Counts in SERUM_PARAMETERS...")
    
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
         
    # Check Max ID
    max_id = max(flat_ids)
    print(f"Max ID: {max_id}")
    if max_id == 201:
        print("✅ Max ID is correct (201).")
    else:
        print(f"❌ FAILED: Max ID is {max_id}, expected 201")

if __name__ == "__main__":
    verify()
