import os
import json
import gzip
import random
import math
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from argparse import ArgumentParser

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

json._default_encoder = DateTimeEncoder()

def normalize_dataset(dataset, dataset_name):
    """
    Normalize a dataset to ensure it has source, metadata, and text fields.
    Raises ValueError if metadata or text fields are missing.
    """
    base_name = dataset_name.split('/')[-1].replace('_filtered', '')
    
    def process_example(example):
        # Ensure source field exists
        if 'source' not in example:
            example['source'] = base_name
        
        # Check for required fields and raise error if missing
        if 'metadata' not in example:
            raise ValueError(f"Dataset {dataset_name} is missing required 'metadata' field")
        
        if 'text' not in example:
            raise ValueError(f"Dataset {dataset_name} is missing required 'text' field")
        
        # Extract source_specific_keys (all other fields)
        standard_fields = {'source', 'metadata', 'text'}
        source_specific = {}
        
        for key, value in example.items():
            if key not in standard_fields:
                source_specific[key] = value
        
        # Convert source_specific to JSON string
        example['source_specific_keys'] = json.dumps(source_specific)
        
        # Keep only standardized fields
        normalized_example = {
            'source': example['source'],
            'metadata': example['metadata'],
            'text': example['text'],
            'source_specific_keys': example['source_specific_keys']
        }
        
        return normalized_example
    
    return dataset.map(process_example)

def create_combined_index(datasets):
    """Create a combined index of (dataset_idx, example_idx) for all datasets"""
    indices = []
    for dataset_idx, dataset in enumerate(datasets):
        for example_idx in range(len(dataset)):
            indices.append((dataset_idx, example_idx))
    return indices

def estimate_total_tokens(datasets, tokenizer, sample_size=1000):
    """Estimate total tokens across all datasets using sampling"""
    total_tokens = 0
    total_examples = sum(len(dataset) for dataset in datasets)
    
    print("Estimating total tokens...")
    for dataset_idx, dataset in enumerate(datasets):
        dataset_tokens = 0
        # Sample up to sample_size examples from this dataset
        sample_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        
        for idx in tqdm(sample_indices, desc=f"Sampling dataset {dataset_idx}", leave=False):
            text = dataset[idx]['text']
            tokens = tokenizer(text)['input_ids']
            dataset_tokens += len(tokens)
        
        # Extrapolate to full dataset
        avg_tokens_per_example = dataset_tokens / len(sample_indices)
        dataset_total_tokens = avg_tokens_per_example * len(dataset)
        total_tokens += dataset_total_tokens
        
        print(f"Dataset {dataset_idx}: ~{dataset_total_tokens:,.0f} tokens ({avg_tokens_per_example:.1f} tokens/example)")
    
    return int(total_tokens)

def main():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default="combined_shards")
    parser.add_argument("--shard-size-tokens", type=int, default=250 * (10 ** 6))  # 250M tokens
    parser.add_argument("--resume-file", type=Path, default="combine_resume.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-interval", type=int, default=100, 
                       help="Check shard size every N examples")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of examples to sample for token estimation")
    args = parser.parse_args()
    
    # Set random seed for reproducible shuffling
    random.seed(args.seed)
    
    # Load tokenizer for token counting
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-1t")
    
    # List of datasets to combine
    dataset_names = [
        "common-pile/youtube_filtered",
        "common-pile/wikiteam_filtered", 
        "common-pile/wikimedia_filtered",
        "common-pile/uspto_filtered",
        "common-pile/usgpo_filtered",
        "common-pile/uk_hansard_filtered",
        "common-pile/ubuntu_irc_filtered",
        "common-pile/stackexchange_filtered",
        "common-pile/regulations_filtered",
        "common-pile/python_enhancement_proposals_filtered",
        "common-pile/pubmed_filtered",
        "common-pile/public_domain_review_filtered",
        "common-pile/project_gutenberg_filtered",
        "common-pile/pressbooks_filtered",
        "common-pile/pre_1929_books_filtered",
        "common-pile/peS2o_filtered",
        "common-pile/oercommons_filtered",
        "common-pile/news_filtered",
        "common-pile/libretexts_filtered",
        "common-pile/library_of_congress_filtered",
        "common-pile/foodista_filtered",
        "common-pile/doab_filtered",
        "common-pile/data_provenance_initiative_filtered",
        "common-pile/cccc_filtered",
        "common-pile/caselaw_access_project_filtered",
        "common-pile/biodiversity_heritage_library_filtered",
        "common-pile/arxiv_papers_filtered",
        "common-pile/arxiv_abstracts_filtered"
    ]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and normalize all datasets
    normalized_datasets = []
    total_examples = 0
    
    print("Loading and normalizing datasets...")
    for name in tqdm(dataset_names, desc="Datasets"):
        try:
            dataset = load_dataset(name, split='train')
            normalized_dataset = normalize_dataset(dataset, name)
            normalized_datasets.append(normalized_dataset)
            total_examples += len(normalized_dataset)
            print(f"✓ {name}: {len(normalized_dataset):,} examples")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            continue
    
    if not normalized_datasets:
        raise ValueError("No datasets were successfully loaded!")
    
    print(f"Total examples across all datasets: {total_examples:,}")
    
    # Estimate total tokens and calculate number of shards
    total_tokens = estimate_total_tokens(normalized_datasets, tokenizer, args.sample_size)
    num_shards = math.ceil(total_tokens / args.shard_size_tokens)
    
    print(f"Estimated total tokens: {total_tokens:,}")
    print(f"Creating {num_shards} shards with {args.shard_size_tokens:,} tokens each")
    
    # Create combined index and shuffle
    print("Creating combined index...")
    indices = create_combined_index(normalized_datasets)
    random.shuffle(indices)
    
    # Resume functionality
    used_indices = set()
    shard_index = 0
    current_shard_tokens = 0
    
    if args.resume_file.exists():
        print("Resuming from previous state...")
        with open(args.resume_file, 'r') as f:
            resume_data = json.load(f)
            used_indices = set(tuple(idx) for idx in resume_data['used_indices'])
            shard_index = resume_data['shard_index']
            current_shard_tokens = resume_data.get('current_shard_tokens', 0)
        
        # Delete the incomplete shard we're resuming from
        incomplete_shard = args.output_dir / f"train-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz"
        if incomplete_shard.exists():
            print(f"Deleting incomplete shard: {incomplete_shard}")
            incomplete_shard.unlink()
    
    # Process indices with streaming writes
    current_shard_file = None
    current_shard_path = None
    examples_in_current_shard = 0
    save_counter = 0
    
    progress = tqdm(total=len(indices), desc="Processing examples")
    
    try:
        for i, (dataset_idx, example_idx) in enumerate(indices):
            # Skip if already processed
            if (dataset_idx, example_idx) in used_indices:
                progress.update(1)
                continue
            
            # Get the example
            example = normalized_datasets[dataset_idx][example_idx]
            
            # Count tokens for this example
            text = example['text']
            token_count = len(tokenizer(text)['input_ids'])
            
            # Create output dictionary
            output_dict = {
                'source': example['source'],
                'metadata': example['metadata'],
                'text': example['text'],
                'source_specific_keys': example['source_specific_keys']
            }
            
            json_line = json.dumps(output_dict) + '\n'
            
            # Open new shard if needed
            if current_shard_file is None:
                current_shard_path = args.output_dir / f"train-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz"
                current_shard_file = gzip.open(current_shard_path, 'wt', encoding='utf-8')
                examples_in_current_shard = 0
                current_shard_tokens = 0
                print(f"Started new shard: {current_shard_path}")
            
            # Write to current shard
            current_shard_file.write(json_line)
            examples_in_current_shard += 1
            current_shard_tokens += token_count
            used_indices.add((dataset_idx, example_idx))
            
            # Check if we need to rotate shard (based on token count)
            if examples_in_current_shard % args.check_interval == 0:
                current_shard_file.flush()
                
                if current_shard_tokens >= args.shard_size_tokens:
                    # Close current shard and rotate
                    current_shard_file.close()
                    current_shard_file = None
                    actual_size = current_shard_path.stat().st_size
                    print(f"Completed shard {shard_index} with {examples_in_current_shard} examples "
                          f"({current_shard_tokens:,} tokens, {actual_size / (1024**2):.1f} MB compressed)")
                    shard_index += 1
                    examples_in_current_shard = 0
                    current_shard_tokens = 0
            
            # Save resume state periodically
            save_counter += 1
            if save_counter >= 50000:
                save_resume_state(args.resume_file, used_indices, shard_index, current_shard_tokens)
                save_counter = 0
            
            progress.update(1)
    
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        # Ensure resume state is saved on error
        if current_shard_file is not None:
            current_shard_file.close()
        save_resume_state(args.resume_file, used_indices, shard_index, current_shard_tokens)
        raise
    
    finally:
        # Close current shard if open
        if current_shard_file is not None:
            current_shard_file.close()
            actual_size = current_shard_path.stat().st_size
            print(f"Completed final shard {shard_index} with {examples_in_current_shard} examples "
                  f"({current_shard_tokens:,} tokens, {actual_size / (1024**2):.1f} MB compressed)")
        
        # Save final resume state
        save_resume_state(args.resume_file, used_indices, shard_index, current_shard_tokens)
    
    # Cleanup resume file if completed
    if len(used_indices) == len(indices):
        if args.resume_file.exists():
            args.resume_file.unlink()
        print("Processing completed successfully!")
    else:
        print(f"Processing paused. {len(used_indices)}/{len(indices)} examples processed.")
    
    progress.close()
    print(f"Shards saved to {args.output_dir}")

def save_resume_state(resume_file, used_indices, shard_index, current_shard_tokens=0):
    """Save resume state to file"""
    # Convert set of tuples to list of lists for JSON serialization
    serializable_used = [list(idx) for idx in used_indices]
    with open(resume_file, 'w') as f:
        json.dump({
            'used_indices': serializable_used,
            'shard_index': shard_index,
            'current_shard_tokens': current_shard_tokens
        }, f)

if __name__ == "__main__":
    main()
