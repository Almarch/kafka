from datasets import load_dataset

def load_jsonl(path, tokenizer, max_length):

    with open(path, 'rb') as f:
        n_rows = sum(1 for _ in f)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=path, split="train", streaming=True)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    class MonkeyClass(type(dataset)):
        def __len__(self):
            return n_rows
    
    dataset.__class__ = MonkeyClass
    
    return dataset
