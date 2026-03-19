
import json
import re

def extract_params(content, var_name):
    # This is a simple regex extraction, might need adjustment if the format is complex
    match = re.search(f'{var_name}\s*=\s*({{.*?}}\n)', content, re.DOTALL)
    if not match:
        # Try to find the block more robustly
        start = content.find(f'{var_name} = {{')
        if start == -1: return None
        
        # Simple brace matching
        count = 0
        end = -1
        for i, char in enumerate(content[start:]):
            if char == '{': count += 1
            if char == '}': 
                count -= 1
                if count == 0:
                    end = start + i + 1
                    break
        if end != -1:
            return eval(content[start + len(f'{var_name} = '):end])
    return None

def load_file(path):
    with open(path, 'r') as f:
        return f.read()

notebook_content = load_file('/Users/mic43145/Documents/new_projects/sonikly/sonikly-modal-deployment/models/vae_v2p7_latent_diffusion_retrain_v2/notebook/vae_v2p7_notebook.py')
model_content = load_file('/Users/mic43145/Documents/new_projects/sonikly/sonikly-modal-deployment/models/vae_v2p7_latent_diffusion_retrain_v2/vae_v2p7.py')

params_notebook = extract_params(notebook_content, 'SERUM_PARAMETERS')
params_model = extract_params(model_content, 'GROUPED_PARAMETER_TYPES')

if params_notebook is None:
    print("Could not extract SERUM_PARAMETERS from notebook")
if params_model is None:
    print("Could not extract GROUPED_PARAMETER_TYPES from model file")

if params_notebook and params_model:
    # Compare
    diffs = []
    
    # Check keys
    keys_nb = set(params_notebook.keys())
    keys_md = set(params_model.keys())
    
    if keys_nb != keys_md:
        diffs.append(f"Different groups: Notebook has {keys_nb - keys_md}, Model has {keys_md - keys_nb}")
        
    all_keys = keys_nb.union(keys_md)
    for k in all_keys:
        if k in params_notebook and k in params_model:
            list_nb = params_notebook[k]
            list_md = params_model[k]
            
            if len(list_nb) != len(list_md):
                diffs.append(f"Group {k} length mismatch: NB={len(list_nb)}, Model={len(list_md)}")
                continue
                
            for i, (p1, p2) in enumerate(zip(list_nb, list_md)):
                # Compare critical fields
                if p1['id'] != p2['id']: diffs.append(f"Group {k} item {i} id mismatch: {p1['id']} vs {p2['id']}")
                if p1['serum_id'] != p2['serum_id']: diffs.append(f"Group {k} item {i} serum_id mismatch: {p1['serum_id']} vs {p2['serum_id']}")
                if p1['type'] != p2['type']: diffs.append(f"Group {k} item {i} type mismatch: {p1['type']} vs {p2['type']}")
                if p1.get('num_categories') != p2.get('num_categories'): diffs.append(f"Group {k} item {i} num_categories mismatch")

    if not diffs:
        print("Parameters MATCH perfectly.")
    else:
        print("Parameters DIFFER:")
        for d in diffs:
            print(d)
