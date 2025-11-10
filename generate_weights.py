import json
import os
import argparse
import numpy as np
from transformers import AutoTokenizer
from ast_dag import generate_token_weights_from_dag
from tqdm import tqdm

def process_dataset(input_path: str, tokenizer):
    """
    Processes a .jsonl dataset file to add token weights to each entry.

    For each line in the input file, it parses the JSON, retrieves the 'input' code,
    generates token weights using the provided tokenizer and `ast_dag` logic,
    and writes the augmented JSON object to a new output file.

    Args:
        input_path (str): Path to the input .jsonl file.
        tokenizer: An initialized Hugging Face tokenizer.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    output_path = input_path.replace('.jsonl', '_with_weights.jsonl')
    print(f"Processing {input_path} -> {output_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc=f"Processing {os.path.basename(input_path)}"):
                try:
                    data = json.loads(line)
                    code = data.get('input')

                    if code and isinstance(code, str):
                        # 1. Get raw depth values from ast_dag.py
                        tokens, _, depths, _ = generate_token_weights_from_dag(code, tokenizer)
                        
                        # 2. Perform normalization
                        if depths:
                            depths_array = np.array(depths, dtype=float)
                            
                            max_depth = np.max(depths_array)

                            # Invert: Higher importance (lower depth) gets a higher score.
                            inverted_depths = (1 + max_depth - depths_array) / (1 + max_depth)
                            
                            weights = np.log(inverted_depths+np.e-1).tolist()
                        else:
                            weights = []
                        
                        # 3. Add new fields for tokens and their normalized weights
                        data['input_tokens'] = tokens
                        data['input_token_weights'] = weights
                    
                    # Write the updated data to the output file
                    outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Skipping line due to JSON decode error: {line.strip()}")
                except Exception as e:
                    print(f"Warning: Skipping line due to an error: {e}. Line: {line.strip()}")

        print(f"Successfully processed {input_path}. Output saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not open file {input_path} or create {output_path}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate token weights for code datasets.")
    parser.add_argument(
        '--files', 
        nargs='+', 
        default=['python_splits/train.jsonl', 'python_splits/test.jsonl'],
        help="List of .jsonl files to process."
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="./local_models/diffucoder",
        help="Path to the Hugging Face tokenizer model."
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.model_path}")
    try:
        # Initialize the tokenizer, consistent with ast_dag.py
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"Fatal: Error loading tokenizer from {args.model_path}: {e}")
        exit(1)

    for file_path in args.files:
        process_dataset(file_path, tokenizer)

    print("\nAll processing complete.")
