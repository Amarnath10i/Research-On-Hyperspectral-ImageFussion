import json
import re
import os

notebooks = [
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal1\notebooks\DAETF_Net_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal2\notebooks\unfoldfusion_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal3\notebooks\continuumfusion_Kaggle_GPU.ipynb",
    r"d:\projects\research-on-hyperspectral--image-fusion\proposal4\notebooks\zerofusion_Kaggle_GPU.ipynb"
]

def patch_engine(source):
    code = "".join(source)
    
    resume_code = """    start_step = 1
    ckpt_path = os.path.join(cfg.out_dir, f"{cfg.name}_checkpoint.pth")
    if os.path.exists(ckpt_path):
        log_fn(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema_model" in locals() and "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        best = ckpt.get("best", -1e9)
        history = ckpt.get("history", history)

    model.train()
"""
    code = re.sub(
        r'    best, t0 = -1e9, time\.time\(\)\n\n    model\.train\(\)\n',
        f'    best, t0 = -1e9, time.time()\n\n{resume_code}',
        code
    )
    
    code = code.replace(
        'for step, batch in enumerate(loader, start=1):',
        'for step, batch in enumerate(loader, start=start_step):'
    )
    
    save_code = """            history["loss"].append(logs["total"])

            save_dict = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best": best,
                "history": history,
            }
            if "ema_model" in locals():
                save_dict["ema_model"] = locals()["ema_model"].state_dict()
            torch.save(save_dict, ckpt_path)
"""
    code = re.sub(
        r'            history\["loss"\]\.append\(logs\["total"\]\)\n',
        save_code,
        code
    )

    return [line + "\n" if not line.endswith("\n") else line for line in code.splitlines(keepends=True)]

for nb_path in notebooks:
    print(f"Processing {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            if any("%%writefile" in line and "engine.py" in line for line in cell["source"]):
                print("  Found engine.py, patching...")
                cell["source"] = patch_engine(cell["source"])
            elif "unfoldfusion" in nb_path and any("def _basis(" in line for line in cell["source"]):
                print("  Found unfoldfusion model.py, fixing SVD issue...")
                new_source = []
                for line in cell["source"]:
                    if "u, _, _ = torch.linalg.svd(cov)" in line:
                        new_source.append("        u, _, _ = torch.linalg.svd(cov.float())\n")
                    else:
                        new_source.append(line)
                cell["source"] = new_source

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

print("Done patching notebooks!")
