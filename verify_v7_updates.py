
import sys
import os

def check_syntax(filepath):
    print(f"Checking syntax of {filepath}...")
    try:
        with open(filepath, "r") as f:
            compile(f.read(), filepath, "exec")
        print(f"✅ {filepath} syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False

files_to_check = [
    "models/vae_v2p7_latent_diffusion_v7_rag_attention_sota/vae_v2p7.py",
    "models/vae_v2p7_latent_diffusion_v7_rag_attention_sota/modal/modal_deploy.py"
]

all_valid = True
for f in files_to_check:
    if not check_syntax(f):
        all_valid = False

if not all_valid:
    sys.exit(1)
