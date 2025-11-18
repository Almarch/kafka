from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch
import json

model_name = "openllama_f16"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r = 32,          # rank: W_new = W_frozen + ΔW with ΔW ≈ A × B where A_{3200 × r} and B_{r × 3200} (3200 = d, hidden dimension of OpenLlama)
    lora_alpha=32,   # scaling factor = alpha/r = 32/32 = 1.0
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"      # MLP layers
    ],
    lora_dropout=0.05,    # 5% of LoRA weights randomly dropped during training (regularization)
    bias="none",          # Don't train bias terms
    task_type="CAUSAL_LM" # next token
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    
    tokens = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=512
    )
    tokens["labels"] = [ids.copy() for ids in tokens["input_ids"]]
    
    return Dataset.from_dict(tokens)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="kafka_openllama",
    
    # Batch & Accumulation
    per_device_train_batch_size=8,        # Real batch size per GPU => ~8Go VRAM
    gradient_accumulation_steps=4,        # Accumulate k steps → effective batch = k * device_batch_size = 32
    
    # Epochs control
    num_train_epochs=1,                   # 1 epoch
    max_steps=-1,                         # -1 = use num_train_epochs instead of fixed steps
    
    # Learning rate
    learning_rate=3e-5,                   # Max LR (will be modulated by scheduler)
    lr_scheduler_type="cosine",           # Cosine annealing: smooth decay from max to 0
    warmup_ratio=0.05,                    # 5% of steps for warmup (prevents initial shock)
    
    # Optimizer
    optim="paged_adamw_8bit",             # 8-bit Adam: saves ~20% VRAM vs standard Adam
    
    # Gradient clipping
    max_grad_norm=0.3,                    # Clip gradients to prevent exploding gradients
    
    # Precision
    bf16=True,                            # Use bfloat16 (Ampere+ GPUs: A100, RTX 30xx+)
    fp16=False,                           # Don't use float16 (bf16 is better)
    
    # Logging & Checkpoints
    logging_steps=50,                     # Print logs every 50 steps
    save_steps=500,                       # Save checkpoint every 500 steps
    save_total_limit=3,                   # Keep only 3 latest checkpoints (saves disk space)
    
    # DataLoader optimizations
    dataloader_pin_memory=True,           # Faster GPU transfers (if enough RAM)
    dataloader_num_workers=2,             # Parallel data loading (2 CPU threads)
    remove_unused_columns=False,          # Don't auto-remove columns (we handle it manually)
    
    # Monitoring
    report_to="none"                      # No WandB/TensorBoard (set "tensorboard" if needed)
)

train_dataset = load_jsonl_dataset("train.jsonl")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

model = model.merge_and_unload()
final_path = "kafkallama"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
