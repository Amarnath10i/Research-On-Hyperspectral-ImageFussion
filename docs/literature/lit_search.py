"""Crossref harvest: IEEE journal papers (2026+) on HSI-MSI fusion / super-resolution."""
import json, time, urllib.parse, urllib.request

QUERIES = [
    "hyperspectral multispectral image fusion",
    "hyperspectral image super-resolution fusion network",
    "hyperspectral pansharpening deep learning",
    "hyperspectral image fusion transformer",
    "hyperspectral image fusion mamba state space",
    "hyperspectral image fusion diffusion model",
    "hyperspectral image fusion unfolding network",
    "hyperspectral super-resolution unsupervised degradation",
    "hyperspectral multispectral fusion tensor low-rank",
    "hyperspectral image fusion cross-attention spectral",
]

BASE = "https://api.crossref.org/works"
seen = {}

for q in QUERIES:
    params = {
        "query.bibliographic": q,
        "filter": "from-pub-date:2026-01-01,type:journal-article",
        "rows": "100",
        "select": "DOI,title,container-title,volume,page,published,author,publisher",
        "mailto": "lit-review@nitandhra.ac.in",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
    except Exception as e:
        print("FAIL", q, e)
        continue
    items = data["message"]["items"]
    print(f"[{q}] -> {len(items)}")
    for it in items:
        doi = it["DOI"]
        if doi in seen:
            continue
        seen[doi] = it
    time.sleep(0.5)

print("\nUNIQUE:", len(seen))

# Keep IEEE (DOI prefix 10.1109) published 2026 or later
out = []
for doi, it in seen.items():
    if not doi.startswith("10.1109/"):
        continue
    yr = it.get("published", {}).get("date-parts", [[0]])[0][0] or 0
    if yr < 2026:
        continue
    out.append({
        "doi": doi,
        "year": yr,
        "journal": (it.get("container-title") or [""])[0],
        "title": (it.get("title") or [""])[0],
        "volume": it.get("volume", ""),
        "pages": it.get("page", ""),
        "authors": "; ".join(
            f"{a.get('family','')}, {a.get('given','')}".strip(", ")
            for a in it.get("author", [])
        ),
    })

out.sort(key=lambda x: (x["journal"], x["title"]))
json.dump(out, open("lit_raw.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("IEEE 2026+:", len(out))
for o in out:
    print(f"{o['year']} | {o['journal'][:48]:48} | {o['title'][:80]}")
