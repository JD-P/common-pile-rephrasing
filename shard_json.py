import json
import gzip
import math
from pathlib import Path

def shard_dataset(input_file, output_dir, num_shards=4):
    """
    Shard a JSON dataset into multiple gzipped JSON lines files.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory where shards will be saved
        num_shards (int): Number of shards to create
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract the training examples
    train_examples = data["train"]
    total_examples = len(train_examples)
    examples_per_shard = math.ceil(total_examples / num_shards)
    
    print(f"Found {total_examples} examples, splitting into {num_shards} shards")
    
    # Create each shard
    for shard_idx in range(num_shards):
        # Calculate start and end indices for this shard
        start_idx = shard_idx * examples_per_shard
        end_idx = min((shard_idx + 1) * examples_per_shard, total_examples)
        
        # Format the filename with zero-padding
        filename = f"train-{shard_idx:05d}-of-{num_shards:05d}.jsonl.gz"
        filepath = Path(output_dir) / filename
        
        print(f"Creating shard {shard_idx+1}/{num_shards}: {filename}")
        
        # Write the shard as gzipped JSON lines
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            for i in range(start_idx, end_idx):
                # Write each example as a JSON line
                json_line = json.dumps(train_examples[i])
                f.write(json_line + '\n')
    
    print(f"Finished creating {num_shards} shards in {output_dir}")

if __name__ == "__main__":
    # Configuration - update these paths as needed
    input_json_file = "1B_sample/train.json"  # Update this path
    output_directory = "1B_sample/sharded_dataset"           # Update this if needed
    
    # Shard the dataset into 4 parts
    shard_dataset(input_json_file, output_directory, num_shards=4)
