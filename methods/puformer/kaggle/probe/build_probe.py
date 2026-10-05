"""Write a CPU Kaggle kernel that inspects the CAVE / Harvard datasets, the Nikon D700 response and the
baseline repositories before anything is trained.

    python build_probe.py && kaggle kernels push -p .
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''import glob, json, os, subprocess
import numpy as np, scipy.io as sio, h5py

def mat_info(p):
    try:
        m = sio.loadmat(p)
        return {k: (v.shape, str(v.dtype), float(np.min(v)), float(np.max(v))) for k, v in m.items()
                if not k.startswith('__') and isinstance(v, np.ndarray)}
    except NotImplementedError:
        with h5py.File(p, 'r') as f:
            return {k: (f[k].shape, str(f[k].dtype)) for k in f.keys()}

for name in ('hyperspectral-image-restoration-based-on-cave', 'harvard-hsi-2'):
    roots = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    print('==', name, roots)
    files = sorted(glob.glob(f'{roots[0]}/**/*.mat', recursive=True))
    print(len(files), 'mat files')
    for p in files:
        if '/HSI/' in p:
            print(os.path.relpath(p, roots[0]), mat_info(p), flush=True)

subprocess.run('wget -q -O /tmp/resp.mat "https://raw.githubusercontent.com/qingma2016/DCTransformer/main/response%20coefficient.mat"', shell=True)
m = sio.loadmat('/tmp/resp.mat')
for k, v in m.items():
    if not k.startswith('__'):
        print('RESPONSE', k, v.shape, v.dtype)
        np.set_printoptions(precision=10, suppress=False, linewidth=200)
        print(repr(np.asarray(v, np.float64)))

for url in ('https://github.com/Deng-shangqi/PSRT', 'https://github.com/qingma2016/DCTransformer',
            'https://github.com/meiruni/MIMFormer', 'https://github.com/J-FHu/Fusformer'):
    d = '/tmp/ext/' + url.rsplit('/', 1)[1]
    subprocess.run(['git', 'clone', '--depth', '1', url, d])
    print(url, subprocess.run(['git', '-C', d, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip())
import torch, timm, einops
print('torch', torch.__version__, 'timm', timm.__version__, 'einops', einops.__version__)
'''


def main():
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")),
              cells=[dict(cell_type="code", metadata={}, execution_count=None, outputs=[], id="c0", source=CODE)])
    with open(os.path.join(HERE, "puformer-data-probe.ipynb"), "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
    meta = dict(id="amarnath10chinu/puformer-data-probe", title="PUFormer data probe",
                code_file="puformer-data-probe.ipynb", language="python", kernel_type="notebook", is_private=True,
                enable_gpu=False, enable_internet=True,
                dataset_sources=["liptee/hyperspectral-image-restoration-based-on-cave",
                                 "nikeshreddypatlolla/harvard-hsi-2"], competition_sources=[], kernel_sources=[])
    with open(os.path.join(HERE, "kernel-metadata.json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
