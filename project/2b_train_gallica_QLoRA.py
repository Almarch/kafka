from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from load_jsonl import load_jsonl

model_name = "./tinyllama_bf16_gallica_fullweight_1M_512t"
tokenizer = AutoTokenizer.from_pretrained("./tinyllama_bf16")

datasource = "gallica_qlora_250K_2048t.jsonl"
train_dataset = load_jsonl(datasource, tokenizer, 2048)
with open(datasource, 'rb') as f:
    n_rows = sum(1 for _ in f)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
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
    r = 128,          # rank: W_new = W_frozen + ΔW with ΔW ≈ A × B where A_{d × r} and B_{r × d} (d: hidden dimension of the base model)
    lora_alpha=128,   # scaling factor = alpha/r = 1.0
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

model.gradient_checkpointing_enable() # trades computation time for VRAM

training_args = TrainingArguments(
    output_dir="train_kafka",
    
    # Batch & Accumulation
    per_device_train_batch_size=1,        # Real batch size per GPU => ~8Go VRAM
    gradient_accumulation_steps=32,       # Accumulate k steps → effective batch = k * device_batch_size = 32
    max_steps=n_rows // 32,                # number of samples / number of effective batches
    
    # Learning rate
    learning_rate=1e-5,                   # Max LR (will be modulated by scheduler)
    lr_scheduler_type="cosine",           # Cosine annealing: smooth decay from max to 0
    warmup_ratio=0.10,                    # 5% of steps for warmup (prevents initial shock)
    
    # Optimizer
    optim="paged_adamw_8bit",             # 8-bit Adam: saves ~20% VRAM vs standard Adam
    
    # Gradient clipping
    max_grad_norm=1,                    # Clip gradients to prevent exploding gradients
    
    # Precision
    bf16=True,                            # Use bfloat16 (Ampere+ GPUs: A100, RTX 30xx+)
    fp16=False,                           # Don't use float16 (bf16 is better)
    
    # Logging & Checkpoints
    logging_steps=20,                     # Print logs every 50 steps
    save_steps=200,                       # Save checkpoint every 500 steps
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

model.save_pretrained("lora_gallica_qlora_250K_2048t")