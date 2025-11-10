import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def generate_html_visualization(data: dict, output_path: str):
    """
    Generates an HTML file to visualize token weights.
    
    Each token is wrapped in a <span> with a background color corresponding
    to its weight. A color bar legend is included.

    This script requires `matplotlib`.
    
    Args:
        data (dict): A dictionary containing 'input_tokens', 'input_token_weights'.
                     It can also contain 'input' for a title.
        output_path (str): The path to save the generated HTML file.
    """
    tokens = data.get('input_tokens')
    weights = data.get('input_token_weights')
    
    if not tokens or not weights:
        print("Error: 'input_tokens' or 'input_token_weights' not found in data.")
        return

    # Normalize weights for color mapping
    min_weight = min(weights)
    max_weight = max(weights)
    
    color_min_val = 0.3

    if max_weight == min_weight:
        norm_weights = [0.5] * len(weights)
    else:
        # Original normalization to [0, 1]
        raw_norm = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        
        # Remap to a lighter range, e.g., [0.2, 1.0] instead of [0.0, 1.0].
        # This avoids the very dark end of the 'viridis' colormap.
        
        norm_weights = [((1.0 - color_min_val) * r) + color_min_val for r in raw_norm]

    # Use a matplotlib colormap. 'viridis' maps 0->purple, 1->yellow.
    # We've adjusted the range so the darkest color is not used.
    cmap = plt.get_cmap('viridis')
    
    # Generate HTML content
    html_content = "<html><head><title>Token Weight Visualization</title>"
    html_content += """
    <style>
        body { font-family: monospace; white-space: pre-wrap; background-color: #f5f5f5; padding: 20px; }
        .code-container { background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2, h3, h4 { font-family: sans-serif; }
        span { padding: 2px 1px; border-radius: 3px; }
        .legend { margin: 20px 0; font-family: sans-serif; }
        .legend-bar { display: flex; height: 25px; border: 1px solid #ccc; border-radius: 4px; overflow: hidden; }
        .legend-color { flex-grow: 1; }
        .legend-labels { display: flex; justify-content: space-between; margin-top: 5px; }
    </style>
    </head><body>
    """
    
    original_code = data.get('input', 'Code Visualization')
    html_content += f"<h2>Code Visualization</h2><div class='code-container'><pre><code>{original_code}</code></pre></div><hr>"
    
    html_content += "<h3>Token Weight Visualization</h3><div class='code-container'>"
    
    for i, token in enumerate(tokens):
        # Handle cases where token list might be longer than weights list
        if i >= len(norm_weights): break
        
        norm_w = norm_weights[i]
        color = cmap(norm_w)
        hex_color = mcolors.to_hex(color)
        
        # Clean up token for display and escape HTML
        display_token = token.replace('Ġ', ' ').replace('Ċ', '\n')
        display_token = display_token.replace('<', '&lt;').replace('>', '&gt;')

        raw_weight = weights[i]
        
        html_content += f'<span style="background-color: {hex_color};" title="Weight: {raw_weight:.2f}">{display_token}</span>'

    html_content += "</div>"
    
    # Add a color bar legend
    html_content += '<div class="legend"><h4>Weight Legend</h4>'
    html_content += '<div class="legend-bar">'
    for i in np.linspace(0, 1, 100):
        # We need to apply the same remapping to the legend bar
        color_val = ((1.0 - color_min_val) * i) + color_min_val
        hex_color = mcolors.to_hex(cmap(color_val))
        html_content += f'<div class="legend-color" style="background-color: {hex_color};"></div>'
    html_content += '</div>'
    html_content += f'<div class="legend-labels"><span>Low Weight (Detailed): {min_weight:.2f}</span><span>High Weight (Structural): {max_weight:.2f}</span></div>'
    html_content += '</div>'
    
    html_content += "</body></html>"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Visualization saved to: file://{os.path.abspath(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Visualize token weights from a .jsonl file.")
    parser.add_argument(
        'input_file',
        type=str,
        help="Path to the .jsonl file (e.g., 'train_with_weights.jsonl')."
    )
    parser.add_argument(
        '--index',
        type=int,
        default=None,
        help="The 0-based line index of the test case to visualize. If not provided, all lines will be processed."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations',
        help="Directory to save the HTML output."
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.index is not None:
                for i, line in enumerate(f):
                    if i == args.index:
                        try:
                            data = json.loads(line)
                            input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
                            output_path = os.path.join(args.output_dir, f'{input_basename}_line_{i}.html')
                            generate_html_visualization(data, output_path)
                        except json.JSONDecodeError:
                            print(f"Error: Could not decode JSON on line {args.index} of {args.input_file}.")
                        break
                else:
                    if 'i' not in locals() or i < args.index:
                        print(f"Error: Line index {args.index} is out of bounds for file {args.input_file}.")
            else:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
                        output_path = os.path.join(args.output_dir, f'{input_basename}_line_{i}.html')
                        generate_html_visualization(data, output_path)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON on line {i} of {args.input_file}. Skipping.")
                
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
