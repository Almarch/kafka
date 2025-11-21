from datasets import load_dataset

def load_jsonl(path, tokenizer, max_length, epochs = 1):

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=path, split="train", streaming=True)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.repeat(epochs)
    return dataset