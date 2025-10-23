import re
import json
import random
import asyncio
import aiohttp
import os
import gzip
import glob
import math
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
import time
from pathlib import Path

# API configuration
API_KEY = os.environ.get("ASI_API_KEY")
BASE_URL = "https://inference.asicloud.cudos.org/v1"
MODEL = "mistralai/mistral-nemo"

# Common assistant response prefix
ASSISTANT_PREFIX = "Sure I can do that.\n\n<rephrase>\n"

# Rephrasing templates
TEMPLATES = [
    {
        "name": "C2 CEFR Obfuscation",
        "user_message": """Obfuscate the following passage so it can only be understood by someone with a C2 CEFR understanding of English. Use long words while preserving as much of the meaning as you can. Put the rephrased passage in rephrase tags <rephrase>like this</rephrase> so that it can be automatically extracted by an inference script.

<passage>
{passage}
</passage>

Try to preserve as much of the original meaning and information as possible during your rephrase while making it C2 CEFR reading level. Include all the information available in the original passage and go sentence by sentence.""",
        "prefix": "hard"
    },
    {
        "name": "Encyclopaedic Style", 
        "user_message": """Rewrite the following passage in an encyclopediac style as found on Wikipedia. Your rewrite should try to preserve as much of the original information as possible while rephrasing and transforming the style. If the passage is of a literary or abstract nature then do your best to interpret it in a factual and objective way.

<passage>
{passage}
</passage>

Be sure to put your answer in rephrase tags similar to the passage tags above.""",
        "prefix": "wiki"
    },
    {
        "name": "Q&A Format",
        "user_message": """Rewrite the following passage to be in a Q&A format. The question should start with the word "Question:" and be a question which the *information* in the following passage might be an answer to. The passage should then be rephrased so that it makes sense as an answer to the question. You should start the rephrased answer with the word "Answer:" and then put the rephrased passage after it. Try to preserve the core information from the original passage while still rephrasing it to not just be the same words.

<passage>
{passage}
</passage>

You should put both the question and its answer between rephrase tags.""",
        "prefix": "qa"
    },
    {
        "name": "B1 Intermediate Simplification",
        "user_message": """Rephrase the following passage so that it could be understood by someone at B1 intermediate understanding of English. Avoid words longer than 5-7 letters while preserving as much of the meaning as you can. Put the rephrased passage in <rephrase> tags <rephrase>like this</rephrase> so that it can be automatically extracted by an inference script.

<passage>
{passage}
</passage>

Try to preserve as much of the original meaning and information as possible during your rephrase while making it B1 intermediate reading level. Include all the information available in the original passage and go sentence by sentence.""",
        "prefix": "easy"
    }
]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-1t")

def compute_md5(text: str) -> str:
    """Compute MD5 hex digest of text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

class ShardedResultWriter:
    def __init__(self, output_dir: str, shard_size: int = 250 * (10 ** 6)):
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.current_shard_tokens = 0
        self.shard_index = 0
        self.current_file = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Resume from checkpoint if exists
        self.resume_file = self.output_dir / "rephrasing_resume.json"
        if self.resume_file.exists():
            with open(self.resume_file, 'r') as f:
                data = json.load(f)
                self.current_shard_tokens = data.get("current_shard_tokens", 0)
                self.shard_index = data.get("shard_index", 0)
                print(f"Resumed from checkpoint: shard {self.shard_index}, {self.current_shard_tokens} tokens")
        
        self._open_current_shard()
    
    def _open_current_shard(self):
        """Open the current shard file for writing"""
        if self.current_file:
            self.current_file.close()
        
        filename = f"rephrased-{self.shard_index:05d}.jsonl.gz"
        filepath = self.output_dir / filename
        self.current_file = gzip.open(filepath, 'wt', encoding='utf-8')
    
    def _save_checkpoint(self):
        """Save current progress to resume file"""
        checkpoint_data = {
            "current_shard_tokens": self.current_shard_tokens,
            "shard_index": self.shard_index,
            "timestamp": time.time()
        }
        with open(self.resume_file, 'w') as f:
            json.dump(checkpoint_data, f)
    
    async def add_result(self, result: Dict[str, Any]):
        """Add a result to the current shard, creating new shard if needed"""
        # Estimate tokens for this result
        result_tokens = result.get('estimated_tokens', 0)
        
        # Check if we need to start a new shard
        if self.current_shard_tokens + result_tokens > self.shard_size and self.current_shard_tokens > 0:
            self.shard_index += 1
            self.current_shard_tokens = 0
            self._open_current_shard()
            print(f"Started new shard: {self.shard_index}")
        
        # Write the result
        self.current_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.current_shard_tokens += result_tokens
        
        # Save checkpoint periodically (every ~10M tokens)
        if self.current_shard_tokens % (10 * 10 ** 6) < result_tokens:
            self._save_checkpoint()
    
    async def flush(self):
        """Flush any buffered data"""
        if self.current_file:
            self.current_file.flush()
        self._save_checkpoint()
    
    def close(self):
        """Close the current file and save final checkpoint"""
        if self.current_file:
            self.current_file.close()
        self._save_checkpoint()

def load_few_shot_examples(template_prefix):
    """Load three few-shot examples from the few_shot_rephrase_pool directory, ensuring the first matches the template type"""
    pool_dir = "few_shot_rephrase_pool"
    
    if not os.path.exists(pool_dir):
        print(f"Warning: Few-shot pool directory '{pool_dir}' not found. Continuing without few-shot examples.")
        return []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(pool_dir) if f.endswith('.json')]
    
    if len(json_files) < 3:
        print(f"Warning: Only {len(json_files)} JSON files found in '{pool_dir}'. Need at least 3 for few-shot learning.")
        if json_files:
            print("Using available files.")
        else:
            print("Continuing without few-shot examples.")
            return []
    
    # Separate files by prefix
    matching_files = [f for f in json_files if f.startswith(template_prefix)]
    other_files = [f for f in json_files if not f.startswith(template_prefix)]
    
    # Select files: first one matching the template, then two random from others
    selected_files = []
    
    # Always include at least one matching example if available
    if matching_files:
        selected_files.append(random.choice(matching_files))
        # Remove the selected file from other_files if it was there
        if selected_files[0] in other_files:
            other_files.remove(selected_files[0])
    else:
        print(f"Warning: No {template_prefix}* files found for template matching")
    
    # Fill remaining slots with random files
    remaining_slots = 3 - len(selected_files)
    if remaining_slots > 0 and other_files:
        if len(other_files) >= remaining_slots:
            selected_files.extend(random.sample(other_files, remaining_slots))
        else:
            selected_files.extend(other_files)
    
    all_messages = []
    
    for filename in selected_files:
        filepath = os.path.join(pool_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'messages' in data:
                    all_messages.extend(data['messages'])
                else:
                    print(f"Warning: File {filename} does not contain 'messages' key")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_messages

def extract_rephrased_text(response_text):
    """Extract content between <rephrase> tags using regex with multiple patterns"""
    patterns = [
        re.compile(r'<rephrase>\n?(.*?)\n?</rephrase>', re.DOTALL),
        re.compile(r'<rephrase>([\s\S]+)</rephrase>'),
        re.compile(r'<rephrase>([\s\S]+)<rephrase>'),
        re.compile(r'<rephrase>([\s\S]+)</passage>'),
        re.compile(r'<rephrase>([\s\S]+)<passage>'),
        re.compile(r'<rephrase>([\s\S]+)</r')
    ]
    
    for pattern in patterns:
        match = pattern.search(response_text)
        if match:
            return match.group(1).strip()
    
    # If no tags found, return ellipsis to indicate missing content
    return "…"

def estimate_tokens_per_character(sample_texts):
    """Estimate tokens per character using the comma tokenizer"""
    ratios = []
    for text in sample_texts[:1000]:  # Use first 1000 samples for estimation
        if not text:
            continue
        char_len = len(text)
        token_len = len(tokenizer(text)["input_ids"])
        try:
            ratios.append(token_len / char_len)
        except:
            ratios.append(0)
    return sum(ratios) / len(ratios) if ratios else 0.25

def split_into_passages(
    text: str,
    max_tokens: int = 350,
    tokens_per_char: float = 0.25,
    min_tokens: int = 50
) -> List[str]:
    """
    Split text into passages following the preprocessing rules in Pieler et al.
    """
    def estimate_tokens(chunk: str) -> int:
        return int(len(chunk) * tokens_per_char)
    
    def split_long_chunk(chunk: str, max_tokens: int) -> List[str]:
        # First try to split on sentence boundaries
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, chunk)
        
        # If sentences are still too long, split on whitespace
        result = []
        for sentence in sentences:
            if estimate_tokens(sentence) <= max_tokens:
                result.append(sentence)
            else:
                # Fallback: split on whitespace
                words = sentence.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    word_tokens = estimate_tokens(word)
                    
                    if current_length + word_tokens > max_tokens and current_chunk:
                        result.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = word_tokens
                    else:
                        current_chunk.append(word)
                        current_length += word_tokens
                
                if current_chunk:
                    result.append(' '.join(current_chunk))
        
        return result
    
    # Step 1: Split on line breaks
    chunks = text.split('\n')
    
    # Step 2: Remove empty passages
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Step 3: Split chunks exceeding max tokens
    new_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens:
            split_chunks = split_long_chunk(chunk, max_tokens)
            new_chunks.extend(split_chunks)
        else:
            new_chunks.append(chunk)
    
    chunks = new_chunks
    
    # Step 4: Merge consecutive passages
    passages = []
    current_passage = []
    current_length = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk)
        
        if chunk_tokens >= max_tokens:
            if current_passage:
                passages.append(' '.join(current_passage))
                current_passage = []
                current_length = 0
            passages.append(chunk)
            continue
            
        if current_length + chunk_tokens > max_tokens:
            if current_passage:
                passages.append(' '.join(current_passage))
                current_passage = []
                current_length = 0
                
        current_passage.append(chunk)
        current_length += chunk_tokens
    
    if current_passage:
        final_passage = ' '.join(current_passage)
        if estimate_tokens(final_passage) >= min_tokens:
            passages.append(final_passage)
    
    return passages

async def send_rephrase_request(session, template_name, user_message, few_shot_messages, max_retries=3):
    """Send a single rephrase request to the API with few-shot examples and retries"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build proper ChatML messages: few-shot examples + current request
    messages = []
    messages.extend(few_shot_messages)
    messages.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ASSISTANT_PREFIX}
    ])
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(f"{BASE_URL}/chat/completions", 
                                  json=payload, headers=headers) as response:
                
                if response.status == 200:
                    data = await response.json()
                    full_response = data["choices"][0]["message"]["content"]
                    rephrased_text = extract_rephrased_text(full_response)
                    
                    # If we got empty content but it's not the ellipsis, retry
                    if not rephrased_text and attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                        
                    return template_name, rephrased_text, True
                else:
                    error_text = await response.text()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return template_name, f"Error: HTTP {response.status} - {error_text}", False
                    
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return template_name, f"Error: {str(e)}", False
    
    return template_name, "…", False

async def rephrase_chunk(passage: str, template: Dict, semaphore: asyncio.Semaphore, tokens_per_char: float) -> Dict[str, Any]:
    """Rephrase a single passage using the specified template"""
    async with semaphore:
        # Load few-shot examples for this template
        few_shot_messages = load_few_shot_examples(template["prefix"])
        
        formatted_user_message = template["user_message"].format(passage=passage)
        
        async with aiohttp.ClientSession() as session:
            template_name, rephrased_text, success = await send_rephrase_request(
                session, template["name"], formatted_user_message, few_shot_messages
            )
            
            return {
                "template": template_name,
                "original_passage": passage,
                "original_passage_md5": compute_md5(passage),
                "rephrased_text": rephrased_text,
                "rephrased_text_md5": compute_md5(rephrased_text),
                "success": success,
                "estimated_tokens": int(len(rephrased_text) * tokens_per_char)
            }

async def process_document(doc: Dict[str, Any], tokens_per_char: float, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    """Process a single document through all templates"""
    original_text = doc.get("text", "")
    if not original_text or len(original_text.strip()) < 50:
        return []
    
    # Compute MD5 of original document text
    original_text_md5 = compute_md5(original_text)
    
    # Split document into passages
    passages = split_into_passages(original_text, tokens_per_char=tokens_per_char)
    if not passages:
        return []
    
    # Create tasks for all passages and templates
    tasks = []
    for passage in passages:
        for template in TEMPLATES:
            task = rephrase_chunk(passage, template, semaphore, tokens_per_char)
            tasks.append(task)
    
    # Process all tasks with timeout
    results = []
    for task in asyncio.as_completed(tasks, timeout=1200):  # 20 minute timeout
        try:
            result = await task
            results.append(result)
        except asyncio.TimeoutError:
            print("Task timed out")
        except Exception as e:
            print(f"Task failed with error: {e}")
    
    # Group results by template and combine
    template_results = {}
    for template in TEMPLATES:
        template_name = template["name"]
        template_passages = [r for r in results if r["template"] == template_name]
        template_passages.sort(key=lambda x: x.get("original_passage", ""))
        
        combined_text = " ".join([r["rephrased_text"] for r in template_passages])
        total_tokens = sum([r.get("estimated_tokens", 0) for r in template_passages])
        
        # Create output document with original metadata
        output_doc = {
            "source": doc.get("source", ""),
            "metadata": doc.get("metadata", {}),
            "text": combined_text,
            "template": template_name,
            "original_length": len(original_text),
            "rephrased_length": len(combined_text),
            "estimated_tokens": total_tokens,
            "source_specific_keys": doc.get("source_specific_keys", ""),
            "processing_timestamp": time.time(),
            # Add MD5 digests
            "original_text_md5": original_text_md5,
            "rephrased_text_md5": compute_md5(combined_text)
        }
        
        template_results[template_name] = output_doc
    
    return list(template_results.values())

def read_gzipped_jsonl_files(data_dir: str):
    """Read all gzipped JSONL files from the data directory"""
    pattern = os.path.join(data_dir, "*.jsonl.gz")
    files = glob.glob(pattern)
    
    if not files:
        raise ValueError(f"No .jsonl.gz files found in {data_dir}")
    
    for file_path in files:
        print(f"Reading from {file_path}")
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

async def main():
    if not API_KEY:
        print("Error: ASI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Data directory
    data_dir = "common_pile_low_code"
    output_dir = "nemo_rephrased_shards"
    
    # Estimate tokens per character using sample data
    print("Estimating tokens per character...")
    sample_texts = []
    sample_count = 0
    for doc in read_gzipped_jsonl_files(data_dir):
        if sample_count >= 1000:
            break
        if doc.get("text"):
            sample_texts.append(doc["text"])
            sample_count += 1
    
    tokens_per_char = estimate_tokens_per_character(sample_texts)
    print(f"Estimated tokens per character: {tokens_per_char:.4f}")
    
    # Initialize sharded result writer
    writer = ShardedResultWriter(output_dir, shard_size=250 * (10 ** 6))
    
    # Create semaphore to limit concurrent requests
    max_concurrent_requests = 20  # Adjust based on API limits
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    # Process documents with progress tracking
    total_docs = 0
    for _ in read_gzipped_jsonl_files(data_dir):
        total_docs += 1
    
    print(f"Processing {total_docs} documents...")
    
    # Reset generator
    document_generator = read_gzipped_jsonl_files(data_dir)
    
    # Global progress bar for documents
    doc_progress = tqdm(total=total_docs, desc="Documents")
    
    # Process documents in batches
    batch_size = 10
    processed_count = 0
    
    while processed_count < total_docs:
        batch_docs = []
        for _ in range(batch_size):
            try:
                doc = next(document_generator)
                batch_docs.append(doc)
            except StopIteration:
                break
        
        if not batch_docs:
            break
        
        # Create tasks for each document in batch
        tasks = []
        for doc in batch_docs:
            task = process_document(doc, tokens_per_char, semaphore)
            tasks.append(task)
        
        # Process batch
        for task in asyncio.as_completed(tasks):
            try:
                results = await task
                # Write results
                for result in results:
                    await writer.add_result(result)
                
                doc_progress.update(1)
                processed_count += 1
                
                # Update progress description
                doc_progress.set_description(f"Documents ({processed_count}/{total_docs}) - Shard {writer.shard_index}")
                
            except Exception as e:
                print(f"Error processing document: {e}")
                doc_progress.update(1)
                processed_count += 1
        
        # Small delay between batches to avoid overwhelming the API
        await asyncio.sleep(1)
    
    # Flush any remaining results and close writer
    await writer.flush()
    writer.close()
    doc_progress.close()
    
    print(f"Rephrasing completed! Output saved to {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
