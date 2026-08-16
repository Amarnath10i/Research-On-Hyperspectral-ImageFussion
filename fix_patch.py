import json

notebooks = [
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal1\notebooks\DAETF_Net_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal2\notebooks\unfoldfusion_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal3\notebooks\continuumfusion_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal4\notebooks\zerofusion_Kaggle_GPU.ipynb"
]

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if any("%%writefile" in line and "engine.py" in line for line in cell["source"]):
                new_source = []
                for line in cell["source"]:
                    if 'ckpt_path = os.path.join(cfg.out_dir, f"{cfg.name}_checkpoint.pth")' in line:
                        line = line.replace('cfg.name', "getattr(cfg, 'name', 'model')")
                    new_source.append(line)
                cell["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Fixed cfg.name bug!")
