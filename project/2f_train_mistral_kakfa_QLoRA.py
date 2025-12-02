from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
from load_jsonl import load_jsonl
import torch

lora = "./mistral_qlora_gallica_100K_512t"
base_model = "./mistral_bf16"
tokenizer = AutoTokenizer.from_pretrained("./mistral_bf16")

datasource = "kafka_512t.jsonl"
train_dataset = load_jsonl(datasource, tokenizer, 512)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute en bf16
    bnb_4bit_use_double_quant=True,         # Nested quantization
    bnb_4bit_quant_type="nf4"               # NormalFloat4
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model = PeftModel.from_pretrained(base_model, lora)

model = prepare_model_for_kbit_training(model)

model.print_trainable_parameters()

model.gradient_checkpointing_enable() # trades computation time for VRAM

training_args = TrainingArguments(
    output_dir="train_mistral_kafka",
    
    # Batch & Accumulation
    per_device_train_batch_size=8,        # Real batch size per GPU => ~8Go VRAM
    gradient_accumulation_steps=1,        # Accumulate k steps → effective batch = k * device_batch_size = 32
    num_train_epochs = 10,               # number of epochs

    # Learning rate
    learning_rate=3e-5,                   # Max LR (will be modulated by scheduler)
    lr_scheduler_type="cosine",           # Cosine annealing: smooth decay from max to 0
    warmup_ratio=0.20,                    # 20% of steps for warmup (prevents initial shock)
    
    # Optimizer
    optim="paged_adamw_8bit",             # 8-bit Adam: saves ~20% VRAM vs standard Adam
    
    # Gradient clipping
    max_grad_norm=1,                      # Clip gradients to prevent exploding gradients
    
    # Precision
    bf16=True,                            # Use bfloat16 (Ampere+ GPUs: A100, RTX 30xx+)
    fp16=False,                           # Don't use float16 (bf16 is better)
    
    # Logging & Checkpoints
    logging_steps=10,                      # Print logs every 50 steps
    save_steps=100,                       # Save checkpoint every 500 steps
    save_total_limit=3,                   # Keep only 3 latest checkpoints (saves disk space)
    
    # DataLoader optimizations
    dataloader_pin_memory=True,           # Faster GPU transfers (if enough RAM)
    dataloader_num_workers=16,            # Parallel data loading (CPU threads)
    dataloader_prefetch_factor=8,         # preload x batches per worker
    
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

model = model.merge_and_unload()
final_path = "kafstral"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)