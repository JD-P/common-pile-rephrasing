import os
import json
import gzip
import math
import random
import requests
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--output-dir", type=Path, default="shards")
parser.add_argument("--tokens", default=10**9, type=int,
                    help="The number of tokens to subset.")
parser.add_argument("--shard-size", type=int, default=(250 * (10 ** 6)))
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-1t")
dataset = load_dataset("common-pile/comma_v0.1_training_dataset")

if not os.path.exists("shards"):
    os.mkdir("shards")

used = set()
token_count = 0
shard_index = 0
if os.path.exists("subset_resume.json"):
    with open("subset_resume.json") as infile:
        data = json.load(infile)
        spans = set(data["used"])
        token_count = data["token_count"]
        shard_index = data["shard_index"]

num_shards = math.ceil(args.tokens / args.shard_size)
milestone = args.shard_size
progress = tqdm(total=args.tokens)
while token_count < args.tokens:
    progress.set_description(f"Tokens Processed (Shard {shard_index})")
    filename = f"train-{shard_index:05d}-of-{num_shards:05d}.jsonl.gz"
    filepath = Path(args.output_dir) / filename
    with gzip.open(filepath, 'wt', encoding='utf-8') as outfile:
        while token_count < milestone:
            choices = set()
            for i in range(64):
                choice = random.randrange(dataset["train"].num_rows)
                while choice in used:
                    choice = random.randrange(dataset["train"].num_rows)
                used.add(choice)
                choices.add(choice)
            assert len(choices) == 64
            items = []
            for choice in choices:
                items.append(dataset["train"][choice])
            texts = [item["text"] for item in items]
            new_tokens = sum([len(i) for i in tokenizer(texts)["input_ids"]])
            token_count += new_tokens
            progress.update(new_tokens)
            for item in items:
                json_line = json.dumps(item)
                outfile.write(json_line + "\n")
            if token_count > milestone:
                with open("subset_resume.json", "w") as outfile:
                    serial_used = list(used)
                    json.dump({"used":serial_used, "token_count":token_count, "shard_index":shard_index}, outfile)
        milestone += args.shard_size
    shard_index += 1
