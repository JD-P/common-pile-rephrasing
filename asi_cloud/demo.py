#!/usr/bin/env python3
import argparse
import gzip
import json
import random
import hashlib
import os
from typing import Dict, Any, List, Set

def compute_md5(text: str) -> str:
    """Compute MD5 hex digest of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_rephrased_docs(file_path: str, num_examples: int = 10) -> List[Dict[str, Any]]:
    """Load a random sample of rephrased documents from a gzipped JSON lines file, handling truncated files."""
    docs = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
            except EOFError:
                break  # Stop if file is truncated
    return random.sample(docs, min(num_examples, len(docs)))

def find_originals(original_dir: str, target_md5s: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Scan a directory of original gzip files for documents matching the target MD5s."""
    found = {}
    for entry in os.scandir(original_dir):
        if entry.is_file() and entry.name.endswith(".jsonl.gz"):
            with gzip.open(entry.path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        md5 = compute_md5(doc["text"])
                        if md5 in target_md5s and md5 not in found:
                            found[md5] = doc
                            if len(found) == len(target_md5s):
                                return found
                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip malformed lines or missing keys
    return found

def main():
    parser = argparse.ArgumentParser(description="Compare rephrased and original documents.")
    parser.add_argument("rephrased_gz", help="Path to the rephrased JSON lines gzip file.")
    parser.add_argument("original_dir", help="Path to the directory containing original JSON lines gzip files.")
    args = parser.parse_args()

    # Step 1: Load 10 random rephrased documents and collect their MD5s
    rephrased_docs = load_rephrased_docs(args.rephrased_gz, 10)
    target_md5s = {doc["original_text_md5"] for doc in rephrased_docs}

    # Step 2: Scan original directory for matching MD5s
    original_docs = find_originals(args.original_dir, target_md5s)

    # Step 3: Generate HTML
    document = """
    <html>
    <head>
    <meta charset="utf-8">
    <style>
    body {
      margin-left: auto;
      margin-right: auto;
      width: 1200px;
    }
    .rephrase-set {
      margin-bottom: 2em;
      border-bottom: 1px solid #ccc;
      padding-bottom: 2em;
    }
    .rephrase {
      display: flex;
    }
    .rephrase div {
      padding: 1em;
      width: 600px;
    }
    </style>
    </head>
    <body>
    """

    for rephrased in rephrased_docs:
        original_md5 = rephrased.get("original_text_md5")
        if original_md5 not in original_docs:
            continue  # Skip if original not found

        original = original_docs[original_md5]
        original_text = original["text"].replace("<", "&lt;").replace(">", "&gt;")
        rephrased_text = rephrased["text"].replace("<", "&lt;").replace(">", "&gt;")

        entry = f"""
        <div class="rephrase-set">
          <div class="rephrase">
            <div><strong>Original:</strong><br>{original_text}</div>
            <div><strong>Rephrased:</strong><br>{rephrased_text}</div>
          </div>
        </div>
        """
        document += entry

    document += """
    </body>
    </html>
    """

    with open("comparison.html", "w", encoding="utf-8") as outfile:
        outfile.write(document)

if __name__ == "__main__":
    main()
