"""Which NEON AOP files can be read anonymously from a Kaggle kernel: direct object URLs on the public Google
Cloud Storage bucket for the SERC tile 368000_4306000 (the tile of NEON's own tutorials), every year and visit."""
import requests

BASE = "https://storage.googleapis.com/neon-aop-products"
TILE = "368000_4306000"
hits = []
for year in (2016, 2017, 2019, 2021, 2022):
    for visit in range(1, 7):
        root = f"{BASE}/{year}/FullSite/D02/{year}_SERC_{visit}/L3"
        urls = {"hsi": f"{root}/Spectrometer/Reflectance/NEON_D02_SERC_DP3_{TILE}_reflectance.h5",
                "rgb": f"{root}/Camera/Mosaic/{year}_SERC_{visit}_{TILE}_image.tif"}
        for kind, url in urls.items():
            try:
                r = requests.get(url, headers={"Range": "bytes=0-15"}, timeout=30)
                code = r.status_code
            except Exception as e:
                code = type(e).__name__
            if code in (200, 206):
                size = r.headers.get("content-range", "")
                hits.append((kind, year, visit, size))
            print(kind, year, visit, code, flush=True)
print("HITS:", hits)
