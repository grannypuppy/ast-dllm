import os
import subprocess
# The 'graphviz' library is excellent for creating and rendering, but not for parsing .dot files.
# To load a .dot file into a Python object, we'll use the 'pydot' library.
# Please install it if you haven't: pip install pydot
import pydot
import logging
import html
# --- Configuration ---
SOURCE_CODE_FILE = "cpg_target.py"
EXPORT_DIR = "cpg_exports"
VISUALIZATION_DIR = "cpg_visualizations"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_cpg_with_joern():
    """
    Uses Joern to parse the source code and export the CPG into .dot files.
    """
    logging.info(f"Starting CPG generation for {SOURCE_CODE_FILE}...")
    
    # 1. Clean previous exports
    if os.path.exists(EXPORT_DIR):
        logging.info(f"Removing existing export directory: {EXPORT_DIR}")
        subprocess.run(["rm", "-rf", EXPORT_DIR])
        
    # 2. Run joern-parse
    parse_command = ["joern-parse", SOURCE_CODE_FILE]
    logging.info(f"Running command: {' '.join(parse_command)}")
    result = subprocess.run(parse_command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("joern-parse failed!")
        logging.error(f"Stderr: {result.stderr}")
        return False
    
    # 3. Run joern-export
    export_command = ["joern-export", "--out", EXPORT_DIR, "--repr", "all", "--format", "dot"]
    logging.info(f"Running command: {' '.join(export_command)}")
    result = subprocess.run(export_command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("joern-export failed!")
        logging.error(f"Stderr: {result.stderr}")
        return False
        
    logging.info(f"CPG data successfully exported to {EXPORT_DIR}")
    return True

def visualize_and_analyze_cpg():
    """
    Finds all .dot files, parses them, applies styling and custom labels for
    better visualization, and renders them as PNGs.
    """
    if not os.path.exists(EXPORT_DIR):
        logging.error(f"Export directory '{EXPORT_DIR}' not found. Did Joern run correctly?")
        return

    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    dot_files = [f for f in os.listdir(EXPORT_DIR) if f.endswith('.dot')]
    if not dot_files:
        logging.warning(f"No .dot files found in {EXPORT_DIR} to process.")
        return
        
    logging.info(f"Found {len(dot_files)} .dot files to parse and visualize in {EXPORT_DIR}")

    for dot_file in dot_files:
        source_path = os.path.join(EXPORT_DIR, dot_file)
        
        try:
            logging.info(f"Parsing {source_path}...")
            graphs = pydot.graph_from_dot_file(source_path)
            if not graphs:
                logging.warning(f"pydot could not parse {dot_file}. Skipping.")
                continue
            graph = graphs[0]
            
            logging.info("num_nodes: " + str(len(graph.get_nodes())))
            logging.info("num_edges: " + str(len(graph.get_edges())))

            logging.info(f"Applying custom styles and labels to the graph from {dot_file}...")

            # 1. Set graph-level attributes for a cleaner layout
            graph.set('rankdir', 'TB')
            # graph.set('splines', 'curved') # Using curved splines for a smoother look
            # graph.set('nodesep', '0.8')
            # graph.set('ranksep', '1.2')

            # 2. Style nodes and create rich HTML-like labels
            for node in graph.get_nodes():
                attrs = node.get_attributes()
                node_type = attrs.get('label', 'N/A').strip('"')
                
                # Extract location information
                code = attrs.get('CODE', '').strip('"')
                line = attrs.get('LINE_NUMBER', '-').strip('"')
                col = attrs.get('COLUMN_NUMBER', '-').strip('"')
                line_end = attrs.get('LINE_END_NUMBER', '-').strip('"')
                col_end = attrs.get('COLUMN_END_NUMBER', '-').strip('"')
                # Create a new, more informative label
                label_text = f'{node_type}\n{code}\nL:{line} C:{col} - L:{line_end} C:{col_end}'
                node.set('label', label_text)
                # Set all nodes to be boxes with a neutral style
                node.set('shape', 'box')
                node.set('style', 'filled')
                node.set('fillcolor', 'whitesmoke')

            # 3. Style edges based on their type
            for edge in graph.get_edges():
                label = edge.get('label')
                if not label: continue
                
                label = label.strip('"')

                if label == "AST":
                    edge.set('color', 'deepskyblue')
                elif label == "CFG":
                    edge.set('color', 'firebrick1')
                elif label.startswith("REACHING_DEF") or label == "CDG":
                    edge.set('color', 'orange')
            
            # --- ANALYSIS (Example) ---
            logging.info(f"Graph '{dot_file}' successfully styled.")
            
            # --- VISUALIZATION ---
            output_png_path = os.path.join(VISUALIZATION_DIR, os.path.splitext(dot_file)[0] + '_rich_styled.png')
            logging.info(f"Rendering richly styled graph to {output_png_path}")
            graph.write_png(output_png_path)

        except Exception as e:
            logging.error(f"Failed to process {dot_file}: {e}", exc_info=True)

if __name__ == "__main__":
    # Step 1: Generate CPG data using Joern
    if generate_cpg_with_joern():
        # Step 2: Parse, analyze, and visualize the generated .dot files
        visualize_and_analyze_cpg()
        logging.info("Processing complete.")
    else:
        logging.error("Halting due to failure in CPG generation.")
