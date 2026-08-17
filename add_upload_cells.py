"""
Adds a 'upload checkpoints back to Kaggle' cell to proposal notebooks,
and creates kaggle_ckpt_* directories with dataset-metadata.json for
proposal3 and proposal4 so they can resume from checkpoints too.
"""
import json, os

def make_upload_cell(out_dir_name, dataset_id, dataset_title):
    code = (
        f'## 15. Upload checkpoints back to Kaggle dataset (for resuming next session)\n'
        f'import os, glob, shutil, subprocess, json\n\n'
        f'_ckpt_src = "{out_dir_name}"\n'
        f'_dataset_id = "{dataset_id}"\n'
        f'_upload_dir = os.path.join(os.getcwd(), "ckpt_upload")\n'
        f'os.makedirs(_upload_dir, exist_ok=True)\n\n'
        f'for _f in glob.glob(os.path.join(_ckpt_src, "*.pth")):\n'
        f'    shutil.copy(_f, _upload_dir)\n'
        f'    print(f"  staged {{os.path.basename(_f)}}")\n\n'
        f'_meta = {{\n'
        f'    "title": "{dataset_title}",\n'
        f'    "id": _dataset_id,\n'
        f'    "licenses": [{{"name": "CC0-1.0"}}]\n'
        f'}}\n'
        f'with open(os.path.join(_upload_dir, "dataset-metadata.json"), "w") as _mf:\n'
        f'    json.dump(_meta, _mf, indent=2)\n\n'
        f'_result = subprocess.run(\n'
        f'    ["kaggle", "datasets", "version", "-p", _upload_dir,\n'
        f'     "-m", "checkpoint auto-save"],\n'
        f'    capture_output=True, text=True\n'
        f')\n'
        f'print(_result.stdout)\n'
        f'if _result.returncode != 0:\n'
        f'    print("Upload failed:", _result.stderr)\n'
        f'else:\n'
        f'    print("Checkpoints uploaded to", _dataset_id)\n'
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code
    }

notebooks = [
    (
        "proposal1/notebooks/DAETF_Net_Kaggle_GPU.ipynb",
        "daetf_out",
        "amarnath10chinu/daetf-net-checkpoint",
        "DAETF-Net Checkpoint",
    ),
    (
        "proposal2/notebooks/unfoldfusion_Kaggle_GPU.ipynb",
        "unfoldfusion_out",
        "amarnath10chinu/unfoldfusion-checkpoint",
        "UnfoldFusion Checkpoint",
    ),
    (
        "proposal3/notebooks/continuumfusion_Kaggle_GPU.ipynb",
        "continuumfusion_out",
        "amarnath10chinu/continuumfusion-checkpoint",
        "ContinuumFusion Checkpoint",
    ),
    (
        "proposal4/notebooks/zerofusion_Kaggle_GPU.ipynb",
        "zerofusion_out",
        "amarnath10chinu/zerofusion-checkpoint",
        "ZeroFusion Checkpoint",
    ),
]

for nb_path, out_dir, dataset_id, dataset_title in notebooks:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Check if upload cell already exists
    last_src = "".join(nb["cells"][-1].get("source", []))
    if "datasets version" in last_src or "ckpt_upload" in last_src:
        print(f"Already has upload cell: {nb_path}")
    else:
        nb["cells"].append(make_upload_cell(out_dir, dataset_id, dataset_title))
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Added upload cell to: {nb_path}")

# Create kaggle_ckpt directories + dataset-metadata.json for proposal3 and proposal4
ckpt_dirs = [
    ("kaggle_ckpt_continuum", "amarnath10chinu/continuumfusion-checkpoint", "ContinuumFusion Checkpoint"),
    ("kaggle_ckpt_zero",      "amarnath10chinu/zerofusion-checkpoint",      "ZeroFusion Checkpoint"),
]
for d, dataset_id, title in ckpt_dirs:
    os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(d, "dataset-metadata.json")
    meta = {"title": title, "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Created {meta_path}")

print("Done.")
