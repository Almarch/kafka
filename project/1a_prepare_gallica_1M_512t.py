from prepare_data import prepare_data

json_out = prepare_data(
    seg = 512,
    n_blocks = 1_000_000,
)

with open("gallica_1M_512t.jsonl", "w", encoding="utf-8") as f:
    for line in json_out:
        f.write(line + "\n")
