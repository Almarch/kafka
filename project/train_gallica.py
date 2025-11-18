from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from load_jsonl import load_jsonl

model_name = "./tinyllama_bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
).to("cuda")

train_dataset = load_jsonl("train_gallica.jsonl", tokenizer, 512)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model.gradient_checkpointing_enable() # trades computation time for VRAM

training_args = TrainingArguments(
    output_dir="train_gallica",
    
    # Batch & Accumulation
    per_device_train_batch_size=2,       # Real batch size per GPU => ~8Go VRAM
    gradient_accumulation_steps=16,      # Accumulate k steps → effective batch = k * device_batch_size = 32
    
    # Epochs control
    num_train_epochs=1,                   # 1 epoch
    max_steps=-1,                         # -1 = use num_train_epochs instead of fixed steps
    
    # Learning rate
    learning_rate=1e-5,                   # Max LR (will be modulated by scheduler)
    lr_scheduler_type="cosine",           # Cosine annealing: smooth decay from max to 0
    warmup_ratio=0.05,                    # 5% of steps for warmup (prevents initial shock)
    
    # Optimizer
    optim="paged_adamw_8bit",             # 8-bit Adam: saves ~20% VRAM vs standard Adam
    
    # Gradient clipping
    max_grad_norm=0.3,                    # Clip gradients to prevent exploding gradients
    
    # Precision
    bf16=True,                            # Use bfloat16 (Ampere+ GPUs: A100, RTX 30xx+)
    fp16=False,                           # Don't use float16
    
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

final_path = "gallica_tinyllama_bf16"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
