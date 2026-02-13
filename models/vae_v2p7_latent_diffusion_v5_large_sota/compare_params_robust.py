
import re
import ast

def extract_block(content, start_marker):
    start_index = content.find(start_marker)
    if start_index == -1:
        return None
    
    # Find the opening brace
    brace_start = content.find('{', start_index)
    if brace_start == -1:
        return None
        
    count = 0
    for i, char in enumerate(content[brace_start:]):
        if char == '{':
            count += 1
        elif char == '}':
            count -= 1
            if count == 0:
                return content[brace_start:brace_start+i+1]
    return None

def extract_class_def(content, class_name):
    # Find "class ClassName"
    pattern = re.compile(rf'^class\s+{class_name}.*?:', re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None
    
    start_index = match.start()
    # Find the end of the class. We look for the next line that starts with a character (no indent)
    # This is heuristic but usually works for top-level classes
    
    lines = content[start_index:].splitlines()
    if not lines: return None
    
    class_lines = [lines[0]]
    for line in lines[1:]:
        if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('#'):
            # Found a top-level line (that is not a comment)
            break
        class_lines.append(line)
        
    return '\n'.join(class_lines)

def normalize_code(code):
    # Remove comments and empty lines for comparison
    lines = code.splitlines()
    params = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('@'):
             # Remove decorators for comparison as they might differ (e.g., register_keras_serializable)
            params.append(stripped)
    return '\n'.join(params)

def verify_params_and_model():
    with open('notebook/vae_v2p7_notebook.py', 'r') as f:
        nb_content = f.read()
    with open('vae_v2p7.py', 'r') as f:
        py_content = f.read()
        
    # 1. Compare Parameters
    nb_params_str = extract_block(nb_content, 'SERUM_PARAMETERS =')
    py_params_str = extract_block(py_content, 'GROUPED_PARAMETER_TYPES =')
    
    if nb_params_str and py_params_str:
        try:
            nb_params = ast.literal_eval(nb_params_str)
            py_params = ast.literal_eval(py_params_str)
            
            # Helper: Check keys and lengths
            diffs = []
            all_keys = set(nb_params.keys()) | set(py_params.keys())
            for k in all_keys:
                if k not in nb_params: diffs.append(f"Missing in NB: {k}"); continue
                if k not in py_params: diffs.append(f"Missing in PY: {k}"); continue
                if len(nb_params[k]) != len(py_params[k]): diffs.append(f"Len mismatch {k}: {len(nb_params[k])} vs {len(py_params[k])}")
                
            if not diffs:
                print("✅ Parameters MATCH.")
            else:
                print("❌ Parameters DIFFER:")
                for d in diffs: print(d)
        except Exception as e:
            print(f"Error parsing params: {e}")
    else:
        print("Could not extract parameter blocks.")

    # 2. Compare Class Definition (Heuristic)
    nb_class = extract_class_def(nb_content, 'VAE_Text_to_Synth_Standard')
    py_class = extract_class_def(py_content, 'VAE_Text_to_Synth_Standard')
    
    if nb_class and py_class:
        norm_nb = normalize_code(nb_class)
        norm_py = normalize_code(py_class)
        
        # Simple difflib
        import difflib
        ratio = difflib.SequenceMatcher(None, norm_nb, norm_py).ratio()
        if ratio > 0.99:
             print("✅ VAE_Text_to_Synth_Standard MATCHES (high similarity).")
        else:
             print(f"⚠️ VAE_Text_to_Synth_Standard DIFFERS (similarity: {ratio:.4f}).")
             # Print first few differ lines
             # for line in difflib.unified_diff(norm_nb.splitlines(), norm_py.splitlines(), n=0):
             #   if line.startswith(('+', '-')): print(line)
    else:
        print("Could not extract VAE class definition.")

if __name__ == "__main__":
    verify_params_and_model()
