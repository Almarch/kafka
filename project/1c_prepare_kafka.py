from transformers import AutoTokenizer
from datasets import load_from_disk
import re, json, random
from prepare_data import prepare_data

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
gutenberg_json = prepare_data(
    input_path = "gutenberg",
    seg = seg,
    n_blocks = n_gutenberg_blocks,
    item_name = "text",
)

## Merge and save
combined = kafka_json + gutenberg_json
random.shuffle(combined)
with open(name + ".jsonl", "w", encoding="utf-8") as f:
    for line in combined:
        f.write(line + "\n")
