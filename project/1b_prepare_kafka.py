from transformers import AutoTokenizer
from datasets import load_from_disk
import re, json, random

seg = 2048
stride = 680 # => 3 sets, 300 blocks
n_gutenberg_blocks = 200 # ~ 40%
name = "3_kafka_40pc_gutenberg_2048t"

random.seed(42)
model_name = "./tinyllama_bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)

## Kafka resource

with open("chateau.txt", "r", encoding="utf-8") as f:
    kafka = f.read()
kafka = re.sub(r"–\s*\d+\s*–\n", "", kafka) # remove page numbers
kafka = re.sub(r"-\n", "", kafka) # fix split words
kafka = re.sub(r'\n(?!–)', ' ', kafka) # remove line breaks

tok = tokenizer.encode(kafka, add_special_tokens=False)
blocks = [
    tok[i:i+seg]
    for i in range(0, len(tok) - seg + 1, stride)
]

print("Number of Kafka blocks: " + str(len(blocks)))

kafka_json = [
    json.dumps({"text": tokenizer.decode(tok_block)}, ensure_ascii=False)
    for tok_block in blocks
]

## Gutenberg resource

gutenberg = load_from_disk("gutenberg")

gutenberg_json = []

i = 0
while i < n_gutenberg_blocks:
    item = gutenberg[random.randint(0, len(gutenberg) - 1)]
    text = item["text"]

    max_start = len(text) - seg * 6
    char_start = random.randint(0, max_start)
    substring = text[char_start:char_start + seg * 6]
    tok = tokenizer.encode(substring, add_special_tokens=False)
    
    if len(tok) < seg:
        continue
    
    start = random.randint(0, len(tok) - seg)
    block_tokens = tok[start:start+seg]
    decoded = tokenizer.decode(block_tokens)
    gutenberg_json.append(
        json.dumps({"text": decoded}, ensure_ascii=False)
    )
    i += 1

## Merge and save
combined = kafka_json + gutenberg_json
random.shuffle(combined)
with open(name + ".jsonl", "w", encoding="utf-8") as f:
    for line in combined:
        f.write(line + "\n")
