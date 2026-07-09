import json

with open("notebooks/dense_net_201.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    if cell_type == "code":
        source = cell["source"]
        first_line = source[0].strip() if source else ""
        print(f"Cell {idx} (code): {first_line}")
    else:
        source = cell["source"]
        first_line = source[0].strip() if source else ""
        print(f"Cell {idx} (markdown): {first_line}")
