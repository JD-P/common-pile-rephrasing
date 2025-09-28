import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import Trainer

# Load primary dataset
primary_ds = load_dataset("jdpressman/comma_v0.1_training_dataset_sample_1B_qwen3_4b_rephrase", split="train")

# Load replay dataset and take 5% sample
replay_ds = load_dataset("jdpressman/comma_v0.1_training_dataset_sample_1B", split="train")
# replay_ds = replay_ds.select(range(int(0.05 * len(replay_ds))))  # 5% replay:cite[4]

# Combine datasets
combined_ds = concatenate_datasets([primary_ds, replay_ds])

tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-1t")
model = AutoModelForCausalLM.from_pretrained(
    "common-pile/comma-v0.1-1t",
    device_map="auto",  # Automatically loads across multiple GPUs:cite[1]
    dtype="auto",
)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=4096,  # Sequence length as per paper
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_ds = replay_ds.map(tokenize_function, batched=True)

# Calculate warmup steps (1% of total steps)
total_steps = 1000
warmup_steps = int(0.01 * total_steps)  # 10 steps

training_args = TrainingArguments(
    output_dir="./comma-finetuned",
    per_device_train_batch_size=8,  # Effective batch size = 8 GPUs × 8 = 64
    gradient_accumulation_steps=8,   # 64 × 8 = 512 (paper's batch size)
    learning_rate=1e-3,              # Max LR after warmup
    weight_decay=0.1,                # AdamW weight decay:cite[10]
    num_train_epochs=1,              # Steps-based training
    max_steps=total_steps,           # 1000 steps
    warmup_steps=warmup_steps,       # Linear warmup
    lr_scheduler_type="cosine",      # Cosine decay after warmup:cite[7]
    fp16=False,                       # Use mixed precision
    bf16=True,
    logging_dir=os.path.abspath("./logs"),
    logging_steps=1,
    report_to="tensorboard",
    save_steps=100,
    # gradient_checkpointing=True,     # Save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
)

print("Logging directory from TrainingArguments:", trainer.args.logging_dir)

trainer.train()

model.save_pretrained("./comma-finetuned-final")
tokenizer.save_pretrained("./comma-finetuned-final")
