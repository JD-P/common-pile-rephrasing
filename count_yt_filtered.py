from datasets import load_dataset

def analyze_dataset_memory_efficient():
    """
    Memory-efficient version that streams the dataset and prints progress every 50k documents
    """
    try:
        print("Loading dataset 'common-pile/youtube_filtered' with streaming...")
        dataset = load_dataset("common-pile/youtube_filtered", streaming=True)
        
        data = dataset['train'] if 'train' in dataset else dataset
        
        num_rows = 0
        total_length = 0
        non_empty_count = 0
        
        print("Processing dataset...")
        print("\n" + "="*60)
        print(f"{'Progress':<12} {'Rows':<12} {'Total Length':<15} {'Avg Length':<15}")
        print("="*60)
        
        for sample in data:
            num_rows += 1
            current_length = 0
            
            if sample.get('text'):
                current_length = len(sample['text'])
                total_length += current_length
                non_empty_count += 1
            
            # Print progress every 50,000 documents
            if num_rows % 50000 == 0:
                avg_length = total_length / non_empty_count if non_empty_count > 0 else 0
                print(f"{num_rows:<12} {num_rows:<12} {total_length:<15} {avg_length:,.2f}")
        
        # Final results after processing all documents
        avg_length = total_length / non_empty_count if non_empty_count > 0 else 0
        
        print("="*60)
        print(f"\n--- Final Results ---")
        print(f"Total number of rows: {num_rows:,}")
        print(f"Total length of all documents: {total_length:,} characters")
        print(f"Average document length: {avg_length:,.2f} characters")
        print(f"Number of non-empty documents: {non_empty_count:,}")
        
        return {
            'num_rows': num_rows,
            'total_length': total_length,
            'avg_length': avg_length,
            'non_empty_count': non_empty_count
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run the analysis
if __name__ == "__main__":
    results = analyze_dataset_memory_efficient()
