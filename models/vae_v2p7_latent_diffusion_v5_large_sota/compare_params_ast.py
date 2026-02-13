
import ast
import re

def get_ast(path):
    with open(path, 'r') as f:
        content = f.read()
    
    # Preprocess to remove magic commands
    lines = content.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('!') or stripped.startswith('%'):
            cleaned_lines.append(f"# {line}") # Comment out magic commands
        else:
            cleaned_lines.append(line)
            
    return ast.parse('\n'.join(cleaned_lines))

def find_assignment(tree, var_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return node.value
    return None

def ast_to_obj(node):
    if isinstance(node, ast.Dict):
        return {ast_to_obj(k): ast_to_obj(v) for k, v in zip(node.keys, node.values)}
    elif isinstance(node, ast.List):
        return [ast_to_obj(x) for x in node.elts]
    elif isinstance(node, ast.Str):
        return node.s
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.NameConstant):
        return node.value
    else:
        return None

def compare_params():
    nb_path = 'notebook/vae_v2p7_notebook.py'
    py_path = 'vae_v2p7.py'
    
    nb_tree = get_ast(nb_path)
    py_tree = get_ast(py_path)
    
    nb_params_node = find_assignment(nb_tree, 'SERUM_PARAMETERS')
    py_params_node = find_assignment(py_tree, 'GROUPED_PARAMETER_TYPES')
    
    if not nb_params_node:
        print(f"Could not find SERUM_PARAMETERS in {nb_path}")
        return
    if not py_params_node:
        print(f"Could not find GROUPED_PARAMETER_TYPES in {py_path}")
        return
        
    nb_params = ast_to_obj(nb_params_node)
    py_params = ast_to_obj(py_params_node)
    
    if nb_params is None or py_params is None:
        print("Failed to parse parameters.")
        return

    diffs = []
    all_keys = set(nb_params.keys()) | set(py_params.keys())
    
    for k in all_keys:
        if k not in nb_params:
            diffs.append(f"Missing group in notebook: {k}")
            continue
        if k not in py_params:
            diffs.append(f"Missing group in python file: {k}")
            continue
            
        list_nb = nb_params[k]
        list_py = py_params[k]
        
        if len(list_nb) != len(list_py):
            diffs.append(f"Group {k} length mismatch: NB={len(list_nb)}, PY={len(list_py)}")
            continue
            
        for i, (p1, p2) in enumerate(zip(list_nb, list_py)):
            p1_id = int(p1.get('id', -1))
            p2_id = int(p2.get('id', -1))
            if p1_id != p2_id: diffs.append(f"Group {k} item {i} ID mismatch: {p1_id} vs {p2_id}")
            
            p1_sid = int(p1.get('serum_id', -1))
            p2_sid = int(p2.get('serum_id', -1))
            if p1_sid != p2_sid: diffs.append(f"Group {k} item {i} serum_id mismatch: {p1_sid} vs {p2_sid}")

            if p1.get('type') != p2.get('type'): diffs.append(f"Group {k} item {i} type mismatch: {p1.get('type')} vs {p2.get('type')}")
            
            if p1.get('num_categories') != p2.get('num_categories'):
                diffs.append(f"Group {k} item {i} num_categories mismatch: {p1.get('num_categories')} vs {p2.get('num_categories')}")

    if not diffs:
        print("Parameters MATCH.")
    else:
        print("Parameters DIFFER:")
        for d in diffs:
            print(d)

if __name__ == "__main__":
    compare_params()
