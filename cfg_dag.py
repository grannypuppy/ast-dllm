import os
import numpy as np
from transformers import AutoTokenizer
from py2cfg import CFGBuilder
import ast
from collections import deque
import logging
import graphviz


# --- Configuration ---
sample_code = """
import sys

import numpy as np

from functools import lru_cache

from collections import deque



#input = sys.stdin.readline

#sys.setrecursionlimit(10 ** 5)



H,W = list(map(int,input().split()))

grid = [list(eval(input())) for i in range(H)]

dp = [[0] * (W + 1) for i in range(H + 1)]

dp[0][0] = 1

mod = (10 ** 9) + 7



for i in range(H):

  for j in range(W):

    if i > 0 or j > 0:

      if grid[i][j] == "#":

        dp[i][j] = 0

      elif grid[i][j] == ".":
        dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % mod
      else:
        dp[i][j] = 0


print((dp[H - 1][W - 1]))

#print(dp)
"""

# --- Core Implementation ---

def get_source_segment(code_string: str, stmt: ast.AST, line_offsets: list) -> str:
    start_line_idx = stmt.lineno - 1
    end_line_idx = stmt.end_lineno - 1
    start_col = stmt.col_offset
    end_col = stmt.end_col_offset

    if isinstance(stmt, (ast.For, ast.If, ast.While, ast.Try, ast.FunctionDef, ast.ClassDef)):
        logging.debug("Special Stmt like for if def")
        if hasattr(stmt, 'orelse') and stmt.orelse:
            end_line_idx = stmt.orelse[0].lineno - 1
            end_col = stmt.orelse[0].col_offset
        elif hasattr(stmt, 'body') and stmt.body:
            end_line_idx = stmt.body[0].lineno - 1
            end_col = stmt.body[0].col_offset

    if start_line_idx >= len(line_offsets) or end_line_idx >= len(line_offsets):
        raise ValueError(f"Line numbers are out of bounds: start_line_idx={start_line_idx}, end_line_idx={end_line_idx}, line_offsets={line_offsets}")
        
    start_char_index = line_offsets[start_line_idx] + start_col
    end_char_index = line_offsets[end_line_idx] + end_col

    end_char_index = min(end_char_index, len(code_string))
    start_char_index = min(start_char_index, end_char_index)
    logging.debug(code_string[start_char_index:end_char_index])
    return start_char_index, end_char_index

def visualize_cfg_dag(cfg, back_edges, block_depths, code_string, output_filename="cfg_dag"):
    """
    Generates a visualization of the CFG, highlighting back edges and showing code in nodes.
    """
    dot = graphviz.Digraph('CFG-DAG', comment='Control Flow Graph as a DAG')
    # dot.attr(rankdir='TB', splines='ortho')
    dot.attr('node', shape='box', fontname='monospace', fontsize='10')
    # Pre-calculate line offsets for code extraction
    line_offsets = [0]
    for line in code_string.splitlines(True):
        line_offsets.append(line_offsets[-1] + len(line))

    for block in cfg:
        block_content_parts = []
        for stmt in block.statements:
            stmt_source_start, stmt_source_end = get_source_segment(code_string, stmt, line_offsets)
            stmt_source = code_string[stmt_source_start:stmt_source_end]
            block_content_parts.append(stmt_source)
        
        block_content = "\n".join(block_content_parts).strip()
        
        depth = block_depths.get(block)
        
        # Create an HTML-like label for the node
        label = f'Block {block.id}\nDepth: {depth}\n{block_content}'

        dot.node(str(block.id), label=label)

    # Add edges
    for block in cfg:
        for exit_link in block.exits:
            label_text = ""
            if exit_link.exitcase:
                # The exitcase is an AST node (e.g., ast.Name, ast.Constant).
                # We need to get its source text to use as a label.
                node = exit_link.exitcase
                # Ensure the node has position attributes before trying to extract source
                if all(hasattr(node, attr) for attr in ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']):
                    start_idx, end_idx = get_source_segment(code_string, node, line_offsets)
                    #label_text = f'{code_string[start_idx:end_idx]}\n({start_idx},{end_idx})'
                    label_text = f'{node.__class__.__name__}\n{code_string[start_idx:end_idx]}\n({start_idx},{end_idx})'
                else:
                    #label_text = "..."
                    label_text = f'{node.__class__.__name__}' # Fallback for nodes without position

            is_back_edge = exit_link in back_edges
            dot.edge(
                str(block.id), 
                str(exit_link.target.id), 
                label=label_text,
                style='dashed' if is_back_edge else 'solid',
                color='red' if is_back_edge else 'black'
            )
            
    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dot.render(output_filename, format='png', view=False, cleanup=True)
        logging.info(f"CFG-DAG visualization saved to '{output_filename}.png'")
    except graphviz.backend.ExecutableNotFound:
        logging.warning("Graphviz not found. Skipping CFG visualization.")


def get_analysis_entrypoint(cfg):
    """
    Finds the correct entry block for analysis based on the user's heuristic.
    For code with functions, the py2cfg's top-level entry is just the
    function definition, and the actual control flow starts from the second block
    in the iteration order.
    """
    all_blocks = list(cfg)
    if len(all_blocks) > 1:
        # Heuristic: The actual logic starts at the second block in the CFG's iteration order.
        return all_blocks[1]
    
    # Fallback for simple CFGs without function definitions.
    return cfg.entryblock

def find_back_edges(cfg):
    """
    Finds back edges in a CFG using Depth First Search.
    A back edge is an edge from a node to one of its ancestors in the DFS tree.
    """
    back_edges = set()
    visited = set()
    recursion_stack = set()

    def dfs(block):
        visited.add(block)
        recursion_stack.add(block)
        for exit_link in block.exits:
            target_block = exit_link.target
            if target_block in recursion_stack:
                back_edges.add(exit_link)
                logging.debug(f"Found back edge from {block.id} to {target_block.id}")
            elif target_block not in visited:
                dfs(target_block)
        
        recursion_stack.remove(block)

    start_block = get_analysis_entrypoint(cfg)
    if start_block:
        dfs(start_block)
    
    return back_edges

def calculate_block_depths(cfg, back_edges):
    """
    Calculates the depth of each block in the CFG, treating it as a DAG by ignoring back edges.
    The depth is the length of the longest path from the entry block.
    """
    nodes = list(cfg)
    node_map = {node.id: node for node in nodes}
    
    in_degree = {node.id: 0 for node in nodes}
    adj = {node.id: [] for node in nodes}

    for node in nodes:
        for exit_link in node.exits:
            if exit_link not in back_edges:
                adj[node.id].append(exit_link.target.id)
                in_degree[exit_link.target.id] += 1

    queue = deque([node.id for node in nodes if in_degree[node.id] == 0])
    depths = {node.id: 0 for node in nodes}
    
    topo_order = []
    while queue:
        u_id = queue.popleft()
        topo_order.append(u_id)
        
        for v_id in adj[u_id]:
            depths[v_id] = max(depths[v_id], depths[u_id] + 1)
            in_degree[v_id] -= 1
            if in_degree[v_id] == 0:
                queue.append(v_id)
                
    max_depth = 0
    if depths:
        max_depth = max(depths.values())

    return {node_map[nid]: d for nid, d in depths.items()}, max_depth


def apply_weights_from_cfg_dag(char_weights_array, code_string, block_depths, penalty_weight):
    """
    Traverses the CFG and applies depth-based weights to the character array.
    """
    line_offsets = [0]
    for line in code_string.splitlines(True):
        line_offsets.append(line_offsets[-1] + len(line))

    for block, depth in block_depths.items():
        weight = depth
        for stmt in block.statements:
            if not all(hasattr(stmt, attr) for attr in ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']):
                logging.warning(f"Statement {stmt} does not have the required attributes for line numbers and offsets.")

            start_char_index, end_char_index = get_source_segment(code_string, stmt, line_offsets)
            
            # We only update if the new weight is lower (more important)
            char_weights_array[start_char_index:end_char_index] = weight

    # Apply maximum depth weight to all whitespace characters (spaces and newlines)
    # Apply maximum depth weight to consecutive whitespace characters (3 or more)
    i = 0
    while i < len(code_string):
        if code_string[i] in (' ', '\n', '\t', '\r'):
            # Found a whitespace character, check if there are consecutive ones
            start = i
            while i < len(code_string) and code_string[i] in (' ', '\n', '\t', '\r'):
                i += 1
            # If we have 3 or more consecutive whitespace characters
            if i - start >= 3:
                char_weights_array[start:i] = penalty_weight
        else:
            i += 1

def get_offsets_for_slow_tokenizer(code_string: str, tokenizer):
    """
    Manually creates an offset mapping for a slow tokenizer.
    """
    tokens = tokenizer.tokenize(code_string)
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
            current_pos += 1
    return tokens, offset_mapping

def generate_token_weights_from_cfg_dag(code_string: str, tokenizer, visualize_path=None):
    """
    Generates an order-weight for each token based on the code's CFG DAG depth.
    """
    # 1. Build CFG
    cfg = CFGBuilder().build_from_src("code", code_string)
    
    # 2. Find back edges to convert CFG to DAG
    back_edges = find_back_edges(cfg)
    
    # 3. Calculate depth of each block in the DAG
    block_depths, max_depth = calculate_block_depths(cfg, back_edges)
    
    # 4. Create a char-level weight array, initialized with a penalty weight.
    penalty_weight = max_depth + 1
    char_weights = np.full(len(code_string), penalty_weight, dtype=float)
    
    # 5. Apply weights from the CFG DAG
    apply_weights_from_cfg_dag(char_weights, code_string, block_depths, penalty_weight)

    # (Optional) Visualize the CFG DAG
    if visualize_path:
        visualize_cfg_dag(cfg, back_edges, block_depths, code_string, visualize_path)

    # 6. Tokenize the code and get the offset mapping.
    if tokenizer.is_fast:
        encoding = tokenizer(
            code_string,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        token_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        # logging.warning("Using slow tokenizer. Manually creating offset mapping.")
        token_ids = tokenizer.encode(code_string, add_special_tokens=False)
        tokens, offset_mapping = get_offsets_for_slow_tokenizer(code_string, tokenizer)

    # 7. Aggregate character weights into token weights using the 'min' strategy.
    token_weights = []
    for start, end in offset_mapping:
        if start == end:
            token_weights.append(penalty_weight)
            continue
        
        relevant_char_weights = char_weights[start:end]
        token_weight = np.min(relevant_char_weights)
        token_weights.append(token_weight)
        
    return tokens, token_ids, token_weights, offset_mapping

# --- Demonstration ---

if __name__ == "__main__":
    model_path = "./local_models/diffucoder"

    logging.basicConfig(level=logging.DEBUG)
    logging.info(f"Loading tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    except Exception as e:
        logging.error(f"Error loading a tokenizer: {e}")
        tokenizer = None

    if tokenizer:
        logging.info("\n--- Analyzing Sample Code ---")
        logging.info(sample_code)
        logging.info("-" * 30)
        
        # Define where to save the visualization
        viz_path = "cfg_visualizations/cfg_with_tokens"

        tokens, token_ids, weights, offsets = generate_token_weights_from_cfg_dag(
            sample_code, tokenizer, visualize_path=viz_path
        )
        
        logging.info(f"\n{'Token':<20} {'ID':<8} {'Offset':<12} {'Weight (CFG-DAG Depth)'}")
        logging.info("=" * 65)
        for token, token_id, weight, offset in zip(tokens, token_ids, weights, offsets):
            logging.info(f"{token:<20} {token_id:<8} {str(offset):<12} {weight:.2f}")

        logging.info("\nAnalysis complete. Lower weight means structurally earlier in the CFG-DAG.")
