"""Track 2: IEEE 2026 papers on HSI + LiDAR / multimodal joint classification fusion."""
import json, re, time, urllib.parse, urllib.request

QUERIES = [
    "hyperspectral lidar joint classification fusion network",
    "masked self attention fusion hyperspectral lidar",
    "multimodal remote sensing fusion classification transformer",
    "hyperspectral SAR multimodal fusion classification",
    "cross-modal attention hyperspectral lidar DSM classification",
    "multimodal fusion network hyperspectral image classification mamba",
    "hyperspectral and lidar data fusion deep learning",
    "multi-source remote sensing data fusion classification network",
]
BASE = "https://api.crossref.org/works"
seen = {}
for q in QUERIES:
    url = BASE + "?" + urllib.parse.urlencode({
        "query.bibliographic": q,
        "filter": "from-pub-date:2026-01-01,type:journal-article",
        "rows": "100",
        "select": "DOI,title,container-title,volume,page,published,author",
        "mailto": "lit-review@nitandhra.ac.in",
    })
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
    except Exception as e:
        print("FAIL", q, e); continue
    items = data["message"]["items"]
    print(f"[{q}] -> {len(items)}")
    for it in items:
        seen.setdefault(it["DOI"], it)
    time.sleep(0.5)

HIGH_IF = {
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "IEEE Transactions on Image Processing",
    "IEEE Transactions on Geoscience and Remote Sensing",
    "IEEE Transactions on Neural Networks and Learning Systems",
    "IEEE Transactions on Multimedia",
    "IEEE Transactions on Circuits and Systems for Video Technology",
}
# must be multimodal: HSI paired with another modality
MODAL = re.compile(r"lidar|sar\b|dsm|multimodal|multi-modal|multisource|multi-source|"
                   r"cross-modal|crossmodal|multi-sensor|optical.{0,12}sar|heterogeneous", re.I)
TOPIC = re.compile(r"hyperspectral|multispectral|remote sensing|spectral", re.I)

out = []
for doi, it in seen.items():
    if not doi.startswith("10.1109/"):
        continue
    yr = (it.get("published", {}).get("date-parts", [[0]])[0] or [0])[0] or 0
    jr = (it.get("container-title") or [""])[0]
    ti = (it.get("title") or [""])[0]
    if yr < 2026 or jr not in HIGH_IF:
        continue
    if not (MODAL.search(ti) and TOPIC.search(ti)):
        continue
    out.append({"doi": doi, "year": yr, "journal": jr, "title": ti,
                "volume": it.get("volume", ""), "pages": it.get("page", ""),
                "authors": "; ".join(
                    f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                    for a in it.get("author", []))})

print("\nmultimodal candidates:", len(out))
EMAIL = "lit-review@nitandhra.ac.in"
for c in out:
    try:
        with urllib.request.urlopen(
                f"https://api.unpaywall.org/v2/{c['doi']}?email={EMAIL}", timeout=45) as r:
            u = json.load(r)
        c["is_oa"], c["oa_status"] = u.get("is_oa"), u.get("oa_status")
    except Exception as e:
        c["is_oa"], c["oa_status"] = None, f"lookup-failed: {e}"
    print(f"{str(c['is_oa']):5} {str(c['oa_status'])[:12]:12} {c['journal'][:42]:42} {c['title'][:70]}")
    time.sleep(0.3)

json.dump(out, open("lit_multimodal.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\nCLOSED:", sum(1 for c in out if c.get("is_oa") is False), "/", len(out))
