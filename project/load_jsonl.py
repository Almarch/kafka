import json
from datasets import Dataset

def load_jsonl(path, tokenizer, max_length):
    with open(path, "r", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokens = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

    tokens["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in ids]  # -100: magic number
        for ids in tokens["input_ids"]
    ]
        
    return Dataset.from_dict(tokens)