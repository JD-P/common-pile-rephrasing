import re
import json
import random
import asyncio
import aiohttp
import os
import sys
import gzip
import glob
import math
import hashlib
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from queue import Queue
import time
from pathlib import Path
from argparse import ArgumentParser
from text_processing import TextProcessor

# Common assistant response prefix
ASSISTANT_PREFIX = "/no_think Sure I can do that.\n\n<rephrase>\n"

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
                    print(full_response)
                    rephrased_text = TextProcessor.extract_rephrased_text(ASSISTANT_PREFIX + full_response)
                    
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

async def rephrase_chunk(md5: str, index: int, passage: str, template: Dict, tokens_per_char: float) -> Dict[str, Any]:
    """Rephrase a single passage using the specified template"""
    # Load few-shot examples for this template
    few_shot_messages = load_few_shot_examples(template["prefix"])

    formatted_user_message = template["user_message"].format(passage=passage)

    async with aiohttp.ClientSession() as session:
        template_name, rephrased_text, success = await send_rephrase_request(
            session, template["name"], formatted_user_message, few_shot_messages
        )

        return {
            "md5": md5,
            "index": index,
            "template": template["prefix"],
            "rephrased_text": rephrased_text,
            "success": success,
            "estimated_tokens": int(len(rephrased_text) * tokens_per_char)
        }

async def process_document(doc: Dict[str, Any], tokens_per_char: float, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    """Process a single document through all templates"""
    original_text = doc.get("text", "")
    if not original_text or len(original_text.strip()) < 50:
        return []
    
    # Compute MD5 of original document text
    original_text_md5 = TextProcessor.compute_md5(original_text)
    
    # Split document into passages
    passages = TextProcessor.split_into_passages(original_text, tokens_per_char=tokens_per_char)
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
            "rephrased_text_md5": TextProcessor.compute_md5(combined_text)
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

class CommonPileLowCodeLoader:
    def __init__(self, data_dir):
        self.shard_paths = [os.path.join(data_dir, path)
                            for path in os.listdir(data_dir)
                            if path.endswith(".jsonl.gz")]
        self.documents = []
        self.until_refill = 0

    async def setup(self):
        await self._add_documents_from_shard()
        await self._add_documents_from_shard()
        
    async def _add_documents_from_shard(self):
        dcount = 0
        path = self.shard_paths.pop()
        with gzip.open(path, 'rt', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    try:
                        self.documents.append(json.loads(line))
                        dcount += 1
                    except json.JSONDecodeError as e:
                        print(e)
                        continue
        self.until_refill += (dcount // 2)
        return dcount

    async def fill_queue(self, q):
        putn = q.maxsize - q.qsize()
        if putn:
            for i in range(putn):
                try:
                    q.put(self.documents.pop())
                except IndexError:
                    await self._add_documents_from_shard()
                    q.put(self.documents.pop())
                self.until_refill -= 1
            if self.until_refill <= 0:
                self.until_refill = 0
                await self._add_documents_from_shard()

class RephraseAccumulator:
    def __init__(self):
        self.wip_rephrasings = {}

    def register_doc(self, doc):
        self.wip_rephrasings[(doc["md5"], doc["template"])] = {
            "chunks":[],
            "expected_chunks":doc["chunk_count"],
            "original_text": doc["text"],
            "template": doc["template"],
            "source": doc["source"],
            "metadata": doc["metadata"],
            "source_specific_keys": doc["source_specific_keys"],
        }
            
    def add_chunk(self, chunk):
        wip = self.wip_rephrasings[(chunk["md5"], chunk["template"])]
        wip["chunks"].append(chunk)
        if len(wip["chunks"]) >= wip["expected_chunks"]:
            wip["chunks"].sort(key=lambda chunk: chunk["index"])
            combined_text = " ".join([c["rephrased_text"] for c in wip["chunks"]])
            total_tokens = sum([r.get("estimated_tokens", 0)
                                for r in wip["chunks"]])
            doc = wip
            original_text = doc["original_text"]
            output_doc = {
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
                "text": combined_text,
                "template": doc.get("template", ""),
                "original_length": len(original_text),
                "rephrased_length": len(combined_text),
                "estimated_tokens": total_tokens,
                "source_specific_keys": doc.get("source_specific_keys", ""),
                "processing_timestamp": time.time(),
                # Add MD5 digests
                "original_text_md5": chunk["md5"],
                "rephrased_text_md5": TextProcessor.compute_md5(combined_text)
            }
        else:
            output_doc = None
        if output_doc:
            del self.wip_rephrasings[(chunk["md5"], chunk["template"])]
        return output_doc
                
async def main():
    parser = ArgumentParser()
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--model", default="qwen--qwen3-4b")
    parser.add_argument("--data-dir", default="../common_pile_low_code")
    parser.add_argument("--output-dir", default="rephrased_shards")
    args = parser.parse_args()
    # API configuration
    global API_KEY
    API_KEY = args.api_key
    global BASE_URL
    BASE_URL = args.base_url
    global MODEL
    MODEL = args.model
    
    if not API_KEY:
        print("Error: ASI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Data directory
    data_dir = args.data_dir
    output_dir = args.output_dir
    
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

    tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-1t")
    tp = TextProcessor(tokenizer)
    tokens_per_char = tp.estimate_tokens_per_character(sample_texts)
    print(f"Estimated tokens per character: {tokens_per_char:.4f}")
    
    # Initialize sharded result writer
    writer = ShardedResultWriter(output_dir, shard_size=250 * (10 ** 6))
        
    # Process documents with progress tracking
    total_docs = 0
    total_tokens = 0
    for doc in read_gzipped_jsonl_files(data_dir):
        total_docs += 1
        total_tokens += tokens_per_char * len(doc["text"]) * 4
    # Round for aesthetic purposes
    total_tokens = int(total_tokens)
    
    print(f"Processing {total_docs} documents with {total_tokens} tokens...")

    dataloader = CommonPileLowCodeLoader(data_dir)
    await dataloader.setup()
    
    # Global progress bar for documents
    progress = tqdm(total=total_tokens, desc="Tokens")
    
    processed_count = 0

    rp = RephraseAccumulator()
    # docs2rephrase exists so we never wait on I/O to read docs
    docs2rephrase = Queue(maxsize=16)
    chunks2rephrase = Queue()
    rephrase_jobs = Queue()
    while dataloader.shard_paths:
        if chunks2rephrase.qsize() < 1024:
            await dataloader.fill_queue(docs2rephrase)
            doc = docs2rephrase.get()
            if not doc["text"] or len(doc["text"].strip()) < 50:
                continue
            passages = TextProcessor.split_into_passages(
                doc["text"],
                tokens_per_char=tokens_per_char
            )
            if not passages:
                continue
            doc["md5"] = TextProcessor.compute_md5(doc["text"])
            doc["chunk_count"] = len(passages)
            for template in TEMPLATES:
                rp.register_doc(doc | {"template":template["prefix"]})
            for i, passage in enumerate(passages):
                # TODO: Remove this
                assert type(passage) == str
                chunk = {"index": i,
                         "md5": doc["md5"],
                         "text": passage}
                chunks2rephrase.put(chunk)
        if chunks2rephrase.qsize() >= 128 and rephrase_jobs.qsize() <= 128:
            for i in range(32):
                chunk = chunks2rephrase.get()
                for template in TEMPLATES:
                    rephrase_jobs.put(
                        rephrase_chunk(chunk["md5"],
                                       chunk["index"],
                                       chunk["text"],
                                       template,
                                       tokens_per_char)
                    )
        if not rephrase_jobs.empty():
            print("Making attempt...")
            current_job = rephrase_jobs.get()
            rephrased_chunk = await current_job
            completed = rp.add_chunk(rephrased_chunk)
            if completed:
                rephrased_doc = completed
                await writer.add_result(rephrased_doc)
                processed_count += 1
                progress.set_description(f"Documents ({processed_count}/{total_docs}) - Shard {writer.shard_index}")
            progress.update(rephrased_chunk["estimated_tokens"])
            
            print(rephrased_chunk)

        # Update progress description
        # 
 
    # Flush any remaining results and close writer
    await writer.flush()
    writer.close()
    progress.close()
    
    print(f"Rephrasing completed! Output saved to {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
