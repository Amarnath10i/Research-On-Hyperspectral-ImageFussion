"""Verify open-access status of candidate DOIs via Unpaywall."""
import json, re, time, urllib.request

HIGH_IF = {
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "IEEE Transactions on Image Processing",
    "IEEE Transactions on Geoscience and Remote Sensing",
    "IEEE Transactions on Neural Networks and Learning Systems",
    "IEEE Transactions on Multimedia",
    "IEEE Transactions on Circuits and Systems for Video Technology",
}
EXCL = re.compile(
    r"classif|segmentation|target detection|anomaly|change detection|unmixing|"
    r"denois|medical|gesture|nitrogen|asteroid|federated|micro-", re.I)
INCL = re.compile(r"fusion|super-?resolution|pansharpen|sharpening", re.I)

d = json.load(open("lit_raw.json", encoding="utf-8"))
cands = [
    x for x in d
    if x["journal"] in HIGH_IF
    and INCL.search(x["title"]) and not EXCL.search(x["title"])
    and re.search(r"hyperspectral|multispectral|spectral", x["title"], re.I)
]
print("candidates:", len(cands))

EMAIL = "lit-review@nitandhra.ac.in"
for c in cands:
    url = f"https://api.unpaywall.org/v2/{c['doi']}?email={EMAIL}"
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            u = json.load(r)
        c["is_oa"] = u.get("is_oa")
        c["oa_status"] = u.get("oa_status")
        c["published_date"] = u.get("published_date")
        c["journal_is_oa"] = u.get("journal_is_oa")
    except Exception as e:
        c["is_oa"] = None
        c["oa_status"] = f"lookup-failed: {e}"
    print(f"{str(c['is_oa']):5} {str(c.get('oa_status'))[:14]:14} {c['doi']:32} {c['title'][:60]}")
    time.sleep(0.3)

json.dump(cands, open("lit_candidates.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
closed = [c for c in cands if c.get("is_oa") is False]
print(f"\nCLOSED ACCESS (OA=No): {len(closed)} / {len(cands)}")
