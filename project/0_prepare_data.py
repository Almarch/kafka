from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
print(torch.cuda.is_bf16_supported())

### Load the model(s)

# Tinyllama
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

# Mistral
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)
tokenizer.save_pretrained("mistral_bf16")
model.save_pretrained("mistral_bf16")

# model_name = "deepseek-ai/deepseek-llm-7b-base"
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16
# )
# tokenizer.save_pretrained("deepseek7b_bf16")
# model.save_pretrained("deepseek7b_bf16")

# model_name = "openlm-research/open_llama_3b_v2"
# tokenizer = AutoTokenizer.from_pretrained(
#     model_name
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16
# )
# tokenizer.save_pretrained("openllama3b_bf16")
# model.save_pretrained("openllama3b_bf16")

### Load all training data

gallica = load_dataset("PleIAs/French-PD-Books", split = "train", streaming=False)
gallica.save_to_disk("gallica")

