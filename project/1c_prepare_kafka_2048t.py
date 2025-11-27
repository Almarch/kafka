from transformers import AutoTokenizer
from datasets import load_from_disk
import re, json, random
from prepare_data import prepare_data

seg = 2048
stride = 512 # => 4 pseudo-epochs
name = "kafka_2048t"

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

## shuffle & save
random.shuffle(kafka_json)
with open(name + ".jsonl", "w", encoding="utf-8") as f:
    for line in kafka_json:
        f.write(line + "\n")
