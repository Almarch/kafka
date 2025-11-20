from transformers import AutoTokenizer
from datasets import load_from_disk
import json, random

random.seed(42)
model_name = "./tinyllama_bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
gallica = load_from_disk("gallica")

for prog in [
    {   
        "name": "gallica_fullweight_1M_512t",
        "seg": 512,
        "n_blocks": 1_000_000_000,
    },
    {
        "name": "gallica_qlora_200K_2048t",
        "seg": 2048,
        "n_blocks": 200_000,
    }
]:
    gallica_json = []

    i = 0
    while i < prog["n_blocks"]:
        item = gallica[random.randint(0, len(gallica) - 1)]
        text = item["complete_text"]
        
        if len(text) < prog["seg"] * 6: # gallica contains very small texts
            continue
            
        max_start = len(text) - prog["seg"] * 6
        char_start = random.randint(0, max_start)
        substring = text[char_start:char_start + prog["seg"] * 6]
        tok = tokenizer.encode(substring, add_special_tokens=False)
        
        if len(tok) < prog["seg"]:
            continue
        
        start = random.randint(0, len(tok) - prog["seg"])
        block_tokens = tok[start:start+prog["seg"]]
        decoded = tokenizer.decode(block_tokens)
        gallica_json.append(
            json.dumps({"text": decoded}, ensure_ascii=False)
        )
        i += 1
        
    with open(prog["name"] + ".jsonl", "w", encoding="utf-8") as f:
        for line in gallica_json:
            f.write(line + "\n")

