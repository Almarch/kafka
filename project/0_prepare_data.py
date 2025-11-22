from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
print(torch.cuda.is_bf16_supported())

### Load the model

model_name = "TinyLlama/TinyLlama_v1.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

tokenizer.save_pretrained("tinyllama_bf16")
model.save_pretrained("tinyllama_bf16")

### Load all training data

gallica = load_dataset("PleIAs/French-PD-Books", split = "train", streaming=False)
gallica.save_to_disk("gallica")

