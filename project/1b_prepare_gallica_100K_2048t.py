from prepare_data import prepare_data

json_out = prepare_data(
    seg = 2048,
    n_blocks = 100_000,
)

with open("gallica_100K_2048t.jsonl", "w", encoding="utf-8") as f:
    for line in json_out:
        f.write(line + "\n")
