import json

def insert_ckpt_restore(nb_path, dataset_id, train_module_name):
    nb = json.load(open(nb_path, encoding='utf-8'))
    
    # Check if restore cell already exists
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if 'ckpt_dir = "/kaggle/input/' in src:
            print(f'Already has restore cell: {nb_path}')
            return

    # Find the cell that calls train()
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        if 'model, history' in src and train_module_name + '.train' in src:
            break
            
    # The restore code
    dataset_name = dataset_id.split('/')[-1]
    restore_code = f'''import shutil\nimport glob\nimport os\nckpt_dir = "/kaggle/input/{dataset_name}"\nif os.path.exists(ckpt_dir):\n    os.makedirs(cfg.out_dir, exist_ok=True)\n    for f in glob.glob(os.path.join(ckpt_dir, "*.pth")):\n        shutil.copy(f, cfg.out_dir)\n        print(f"Restored checkpoint {{os.path.basename(f)}}")\n'''
    
    restore_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': restore_code
    }
    
    nb['cells'].insert(i, restore_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Added restore cell to: {nb_path}')

insert_ckpt_restore('proposal3/notebooks/continuumfusion_Kaggle_GPU.ipynb', 'amarnath10chinu/continuumfusion-checkpoint', 'continuumfusion')
insert_ckpt_restore('proposal4/notebooks/zerofusion_Kaggle_GPU.ipynb', 'amarnath10chinu/zerofusion-checkpoint', 'zerofusion')
