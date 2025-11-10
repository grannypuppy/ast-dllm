import os
import numpy as np
from transformers import AutoTokenizer
from py2cfg import CFGBuilder
import ast

# --- Configuration ---

def get_cfg_block_multiplier(block):
    """
    Calculates a weight multiplier based on a CFG block's structural importance.
    
    Args:
        block: A basic block from the py2cfg CFG.

    Returns:
        A float multiplier indicating the block's importance.
    """
    # A simple linear block usually has 1 predecessor and 1 exit (connectivity = 2).
    # Blocks that act as branches (like 'if') or merges have higher connectivity.
    connectivity = len(block.predecessors) + len(block.exits)
    
    # We assign a higher multiplier to blocks that are control flow hubs.
    if connectivity <= 2:
        return 1.0  # No extra weight for simple, linear blocks
    else:
        # Assign a bonus for being a control flow hub.
        # This formula can be tuned as a hyperparameter for the research.
        # Here, each extra connection point adds a 0.2 bonus.
        return 1.0 + 0.2 * (connectivity - 2)

# --- Core Implementation ---

def get_offsets_for_slow_tokenizer(code_string: str, tokenizer):
    """
    Manually creates an offset mapping for a slow tokenizer. This is a direct copy
    from ast_analysis.py for compatibility with non-fast tokenizers.
    """
    tokens = tokenizer.tokenize(code_string, add_special_tokens=False)
    offset_mapping = []
    current_pos = 0
    for token in tokens:
        findable_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
        try:
            start = code_string.index(findable_token, current_pos)
            end = start + len(findable_token)
            offset_mapping.append((start, end))
            current_pos = end
        except ValueError:
            offset_mapping.append((current_pos, current_pos))
    return tokens, offset_mapping

def apply_weights_from_cfg(cfg, char_weights_array, code_string):
    """
    Traverse the CFG, calculate a multiplier for each block based on its
    connectivity, and apply it to the character-level weight array.
    """
    # Pre-calculate the starting character index of each line for quick lookup.
    line_offsets = [0]
    for line in code_string.splitlines(True): # Keep endings for accurate offsets
        line_offsets.append(line_offsets[-1] + len(line))

    # Iterate over each block in the Control Flow Graph
    for block in cfg:

        multiplier = get_cfg_block_multiplier(block)

        print(f"\n[DEBUG] Found important CFG Block ID: {block.id} (Multiplier: {multiplier:.2f})")

        for stmt in block.statements:
            if not all(hasattr(stmt, attr) for attr in ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']):
                print(f"Warning: Statement {stmt} does not have the required attributes for line numbers and offsets.")

            # Line numbers are 1-based, list indices are 0-based
            start_line_idx = stmt.lineno - 1
            end_line_idx = stmt.end_lineno - 1
            start_col = stmt.col_offset
            end_col = stmt.end_col_offset
            # For control flow statements, we only apply the weight to their "header"
            # to avoid overlapping weights on nested bodies.
            if isinstance(stmt, (ast.For, ast.If, ast.While, ast.Try, ast.FunctionDef, ast.ClassDef)) and stmt.body:
                print("Special Stmt like for if def")
                print(stmt.body)
                end_line_idx = stmt.body[0].lineno - 1
                end_col = stmt.body[0].col_offset

            # Check if line numbers are valid
            if start_line_idx >= len(line_offsets) or end_line_idx >= len(line_offsets):
                raise ValueError(f"Line numbers are out of bounds: start_line_idx={start_line_idx}, end_line_idx={end_line_idx}, line_offsets={line_offsets}")
                
            # Calculate character start and end indices
            start_char_index = line_offsets[start_line_idx] + start_col
            end_char_index = line_offsets[end_line_idx] + end_col

            # Ensure indices are within bounds of the array
            end_char_index = min(end_char_index, len(char_weights_array))
            start_char_index = min(start_char_index, end_char_index)

            # --- Logging the details ---
            stmt_code = code_string[start_char_index:end_char_index]
            print(f"{stmt_code}\n(Type: {type(stmt).__name__})")
            print(f"     - Location: (L{stmt.lineno}:{stmt.col_offset}) to (L{end_line_idx + 1}:{end_col})")
            print(f"     - Char span: [{start_char_index}:{end_char_index}]")
            # ---------------------------

            char_weights_array[start_char_index:end_char_index] *= multiplier

def generate_token_weights_from_cfg(code_string: str, tokenizer):
    """
    Generates order-weights for each token based on CFG structural analysis.
    """
    # 1. Create a character-level weight array, initialized to 1.0
    char_weights = np.full(len(code_string), 1.0, dtype=float)
    
    # 2. Build the CFG and apply weights based on block connectivity
    try:
        cfg = CFGBuilder().build_from_src("code", code_string)
        apply_weights_from_cfg(cfg, char_weights, code_string)
    except Exception as e:
        print(f"Warning: Could not build CFG. Using default weights of 1.0. Error: {e}")

    # 3. Tokenize the code and get the offset mapping
    if tokenizer.is_fast:
        encoding = tokenizer(code_string, return_offsets_mapping=True, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        print("Warning: Using slow tokenizer. Manually creating offset mapping.")
        token_ids = tokenizer.encode(code_string, add_special_tokens=False)
        tokens, offset_mapping = get_offsets_for_slow_tokenizer(code_string, tokenizer)

    # 4. Aggregate character weights into token weights using the 'mean' strategy
    token_weights = []
    for start, end in offset_mapping:
        if start == end:
            token_weights.append(1.0)
            continue
        
        relevant_char_weights = char_weights[start:end]
        if len(relevant_char_weights) == 0:
            token_weights.append(1.0)
        else:
            # Use the mean weight of all characters covered by the token
            token_weight = np.mean(relevant_char_weights)
            token_weights.append(token_weight)
            
    return tokens, token_ids, token_weights, offset_mapping

# --- Demonstration ---

if __name__ == "__main__":
    model_path = "./local_models/diffucoder"

    print(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"Error loading a tokenizer: {e}")
        tokenizer = None

    if tokenizer:
        sample_code = """
def find_even_numbers(nums):
    # This is a comment
    results = []
    for num in nums:
        if num % 2 == 0:
            results.append(num)
    return results
"""
        print("\n--- Analyzing Sample Code with CFG ---")
        print(sample_code)
        print("-" * 40)
        
        tokens, token_ids, weights, offsets = generate_token_weights_from_cfg(
            sample_code, tokenizer
        )
        
        print(f"\n{'Token':<20} {'ID':<8} {'Offset':<12} {'Weight'}")
        print("=" * 55)
        for token, token_id, weight, offset in zip(tokens, token_ids, weights, offsets):
            print(f"{token:<20} {token_id:<8} {str(offset):<12} {weight:.2f}")

        print("\nAnalysis complete. Weights > 1.0 indicate tokens in important control flow blocks.")
