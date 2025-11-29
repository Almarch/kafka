from transformers import AutoTokenizer
from datasets import load_from_disk
import json, random

def prepare_data(
    input_path = "gallica",
    item_name = "complete_text",
    seg = 512,
    n_blocks = 1_000,
    seed = 42,
    model = "./tinyllama_bf16"
):

    random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model)
    data = load_from_disk(input_path)

    json_out = []
    
    i = 0
    while i < n_blocks:
        item = data[random.randint(0, len(data) - 1)]
        text = item[item_name]
        
        if len(text) < seg * 6: # if data contains very small texts
            continue
            
        max_start = len(text) - seg * 6
        char_start = random.randint(0, max_start)
        substring = text[char_start:char_start + seg * 6]
        tok = tokenizer.encode(substring, add_special_tokens=False)
        
        if len(tok) < seg:
            continue
        
        start = random.randint(0, len(tok) - seg)
        block_tokens = tok[start:start+seg]
        decoded = tokenizer.decode(block_tokens)
        json_out.append(
            json.dumps({"text": decoded}, ensure_ascii=False)
        )
        i += 1
        
    return json_out