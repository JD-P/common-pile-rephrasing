import json
import random
from datasets import load_dataset
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("rephrase_jsonl")
args = parser.parse_args()

dataset = load_dataset("jdpressman/comma_v0.1_training_dataset_sample_1B")

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

.rephrase {
  display: flex;
}

.rephrase div {
  padding: 2em;
  width: 600px;
}
</style>
</head>
<body>
"""

with open(args.rephrase_jsonl) as infile:
    rephrasings = []
    for line in infile.readlines():
        rephrasings.append(json.loads(line))

rephrase_index = {}
for idx, item in enumerate(rephrasings):
    try:
        rephrase_index[item["index"]].append(idx)
    except KeyError:
        rephrase_index[item["index"]] = []
        rephrase_index[item["index"]].append(idx)

for i in range(50):
    choice = random.randrange(len(dataset["train"]))
    while choice not in rephrase_index:
        choice = random.randrange(len(dataset["train"]))
    entry = f'<div class="rephrase-set" id="{choice}">\n'
    original_text = dataset["train"][choice]["text"]
    original_text = original_text.replace("<", "&lt;")
    original_text = original_text.replace(">", "&gt;")
    for rephrase_idx in rephrase_index[choice]:
        entry += f'<div class="rephrase">'
        rephrasing = rephrasings[rephrase_idx]
        rephrase_text = "[" + rephrasing["template"] + "] " + rephrasing["text"]
        rephrase_text = rephrase_text.replace("<", "&lt;")
        rephrase_text = rephrase_text.replace(">", "&gt;")
        entry += f"<div>{original_text}</div>\n"
        entry += f"<div>{rephrase_text}</div>\n"
        entry += "</div>"
    entry += "</div>"
    document += entry
    
document += "</body></html>"
    
with open("demo.html", "w") as outfile:
    outfile.write(document)
    outfile.flush()
