# -*- coding: utf-8 -*-
"""Fill the manuscript's experiment tables from the Kaggle run's JSON outputs."""
import json, os, sys

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def fmt_row(name, m):
    return f"| {name:<26} | {m['psnr']:6.3f} | {m['ssim']:.4f} | {m['sam']:6.3f} | {m['ergas']:8.3f} |"

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else '.'
    # results_all_datasets.json has: indomain{DS:{method:{psnr,...}}}, cross, r_id, config
    try:
        allr = load_json(os.path.join(base, 'results_all_datasets.json'))
    except FileNotFoundError:
        print('results_all_datasets.json not found; try downloading kernel output first.')
        return

    print('=' * 78)
    print('IN-DOMAIN RESULTS (papers protocol, 3-band MSI)')
    print('=' * 78)
    for ds, methods in allr.get('indomain', {}).items():
        print(f'\n## {ds}')
        print('| Method | PSNR | SSIM | SAM | ERGAS |')
        print('|---|---|---|---|---|')
        for name, m in methods.items():
            print(fmt_row(name, m))

    cross = allr.get('cross', {})
    if cross:
        print('\n## Cross-domain zero-shot')
        print('| Direction | PSNR | SSIM | SAM | ERGAS |')
        print('|---|---|---|---|---|')
        for k in ('CAVE->HARVARD', 'HARVARD->CAVE'):
            if k in cross:
                print(fmt_row(k, cross[k]))

    rid = allr.get('r_id', {})
    if rid:
        print('\n## r_id (observation-identifiable rank)')
        for ds, info in rid.items():
            print(f'  {ds:<10} mean r_id={info["mean"]:.1f}  per-scene={info["rows"]}')

if __name__ == '__main__':
    main()