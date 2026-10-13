"""Similarity and AI-text screening of the compiled paper (runs on Kaggle with internet and two T4 GPUs).

    python integrity_check.py --pdf main.pdf supplementary.pdf --bib references.bib --own README.md ... --out DIR

Similarity, as iThenticate measures it: the text of each PDF, without its reference list, is compared with the full
texts (arXiv or open-access PDFs) and abstracts (OpenAlex) of every cited work, and with the project's own public
README files. A match is a run of at least 8 identical words; the similarity index is the share of the paper's words
inside a match. iThenticate also searches paywalled publisher full texts and the open web, which this check cannot.

AI-text screening: two trained detectors (desklib/ai-text-detector-v1.01, fakespot-ai/roberta-base-ai-text-
detection-v1) and the zero-shot Binoculars score (Qwen2.5-1.5B observer, Qwen2.5-1.5B-Instruct performer). Each is
also run on human-written abstracts of the cited works published before 2022 and on abstracts written by an
instruction-tuned model (Phi-3.5-mini-instruct) for the same titles, so the paper's scores can be read against both.
"""
import argparse
import difflib
import io
import json
import os
import re
import time
import urllib.parse

import numpy as np
import requests

UA = {"User-Agent": "puformer-integrity-check/1.0"}
N = 8                                    # minimum match length in words


# ---------------------------------------------------------------------------------------------- bibliography
def bib_entries(path):
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start, depth, i = m.end(), 1, m.end()
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body, fields = text[start:i - 1], {}
        for f in re.finditer(r"(\w+)\s*=\s*", body):
            j = f.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                d, k = 1, j + 1
                while d and k < len(body):
                    d += {"{": 1, "}": -1}.get(body[k], 0)
                    k += 1
                val = body[j + 1:k - 1]
            elif body[j] == '"':
                k = body.index('"', j + 1)
                val = body[j + 1:k]
            else:
                val = re.match(r"[^,\n]*", body[j:]).group(0)
            fields.setdefault(f.group(1).lower(), val.strip())
        fields["key"], fields["type"] = m.group(2), m.group(1).lower()
        out.append(fields)
    return out


def plain(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s or "")
    return re.sub(r"\s+", " ", re.sub(r"[{}$\\]", "", s)).strip()


def similar(a, b):
    return difflib.SequenceMatcher(None, plain(a).lower(), plain(b).lower()).ratio()


# ---------------------------------------------------------------------------------------------- web sources
def get(url, timeout=40, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=timeout, **kw)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2)
    return None


def openalex(entry):
    doi, title = entry.get("doi", "").strip(), plain(entry.get("title", ""))
    w = None
    if doi:
        r = get("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/"))
        w = r.json() if r is not None and r.ok else None
    if w is None and title:
        r = get("https://api.openalex.org/works", params={"search": title[:200], "per-page": 3})
        if r is not None and r.ok:
            cands = [c for c in r.json().get("results", []) if similar(c.get("title") or "", title) > 0.85]
            w = cands[0] if cands else None
    if w is None:
        return None
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((p, word) for word, ps in inv.items() for p in ps)
    arxiv, pdfs = None, []
    for loc in w.get("locations") or []:
        url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", url)
        if m:
            arxiv = m.group(1)
        if loc.get("pdf_url"):
            pdfs.append(loc["pdf_url"])
    return dict(title=w.get("title"), year=w.get("publication_year"), abstract=" ".join(x for _, x in pos),
                arxiv=arxiv, pdfs=pdfs)


def arxiv_by_title(title):
    q = re.sub(r"[^A-Za-z0-9 ]", " ", plain(title))
    q = " AND ".join(f"ti:{w}" for w in q.split() if len(w) > 3)[:300]
    r = get("http://export.arxiv.org/api/query", params={"search_query": q, "max_results": 3})
    time.sleep(3)                                                    # arXiv asks for 3 s between calls
    if r is None or not r.ok:
        return None
    for e in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>http://arxiv.org/abs/([^<v]+)", e)
        if t and i and similar(t.group(1), title) > 0.85:
            return i.group(1)
    return None


def pdf_text(data):
    import fitz
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def fetch_pdf(url):
    r = get(url, timeout=60)
    if r is None or not r.ok or not r.content.startswith(b"%PDF") or len(r.content) > 40e6:
        return ""
    return pdf_text(r.content)


# ---------------------------------------------------------------------------------------------- text
def paper_text(pdf):
    text = pdf_text(open(pdf, "rb").read())
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    refs = [m.start() for m in re.finditer(r"\n\s*R\s?E\s?F\s?E\s?R\s?E\s?N\s?C\s?E\s?S\s*\n", text, re.I)]
    return text[:refs[-1]] if refs else text


def words(text):
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+", text)]


def shingles(text):
    w = [x for x, _, _ in words(text)]
    return {hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)}


def similarity(text, sources):
    """sources: {name: text}. Returns the index, the share per source and the matched passages."""
    w = words(text)
    toks = [x for x, _, _ in w]
    grams = [hash(tuple(toks[i:i + N])) for i in range(len(toks) - N + 1)]
    union = np.zeros(len(toks), bool)
    per, passages = {}, []
    for name, src in sources.items():
        sset = shingles(src)
        cov = np.zeros(len(toks), bool)
        for i, g in enumerate(grams):
            if g in sset:
                cov[i:i + N] = True
        if not cov.any():
            continue
        union |= cov
        per[name] = round(100 * cov.mean(), 2)
        idx = np.flatnonzero(cov)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for r in runs:
            passages.append(dict(source=name, words=len(r), text=text[w[r[0]][1]:w[r[-1]][2]].replace("\n", " ")))
    passages.sort(key=lambda p: -p["words"])
    return dict(words=len(toks), index=round(100 * union.mean(), 2),
                per_source=dict(sorted(per.items(), key=lambda kv: -kv[1])), passages=passages)


def chunks(text, lo=120, hi=230):
    """Prose chunks of lo..hi words on sentence boundaries; table and equation fragments are dropped."""
    text = re.sub(r"\s+", " ", text)
    out, cur = [], []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text):
        cur.append(s)
        n = sum(len(x.split()) for x in cur)
        if n >= lo:
            c = " ".join(cur)
            toks = c.split()
            alpha = sum(t.strip(".,;:()").isalpha() for t in toks) / max(len(toks), 1)
            if alpha > 0.75 and n <= hi * 1.5:
                out.append(c)
            cur = []
    return out


# ---------------------------------------------------------------------------------------------- detectors
def desklib_scores(texts, device="cuda:0"):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    name = "desklib/ai-text-detector-v1.01"
    cfg, tok = AutoConfig.from_pretrained(name), AutoTokenizer.from_pretrained(name)
    base = AutoModel.from_config(cfg)
    head = torch.nn.Linear(cfg.hidden_size, 1)
    sd = load_file(hf_hub_download(name, "model.safetensors"))
    miss = base.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
    assert len(miss.missing_keys) < 3, miss.missing_keys[:5]
    head.load_state_dict({k[11:]: v for k, v in sd.items() if k.startswith("classifier.")})
    base, head = base.to(device).eval(), head.to(device).eval()
    out = []
    with torch.no_grad():
        for t in texts:
            e = tok(t, truncation=True, max_length=768, return_tensors="pt").to(device)
            h = base(**e)[0]
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(float(torch.sigmoid(head((h * m).sum(1) / m.sum(1)))[0, 0]))
    del base, head
    torch.cuda.empty_cache()
    return out


def fakespot_scores(texts, device="cuda:0"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = "fakespot-ai/roberta-base-ai-text-detection-v1"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    labels = {i: str(l).lower() for i, l in model.config.id2label.items()}
    ai = [i for i, l in labels.items() if "ai" in l.split("_") or "machine" in l or l in ("ai", "fake", "generated")]
    ai = ai[0] if ai else 1
    out = []
    with torch.no_grad():
        for t in texts:
            e = tok(t, truncation=True, max_length=512, return_tensors="pt").to(device)
            out.append(float(torch.softmax(model(**e).logits, -1)[0, ai]))
    del model
    torch.cuda.empty_cache()
    return out, labels


def binoculars_scores(texts):
    """log-perplexity of the performer over the observer-performer cross-perplexity; lower = more machine-like."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    obs = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16).to("cuda:0").eval()
    per = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16).to("cuda:1").eval()
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, truncation=True, max_length=512, return_tensors="pt").input_ids
            lo = obs(ids.to("cuda:0")).logits[0, :-1].float()
            lp = per(ids.to("cuda:1")).logits[0, :-1].float().to("cuda:0")
            tgt = ids[0, 1:].to("cuda:0")
            ppl = torch.nn.functional.cross_entropy(lp, tgt)
            xppl = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1).mean()
            out.append(float(ppl / xppl))
    del obs, per
    torch.cuda.empty_cache()
    return out


def generate_ai_abstracts(titles):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "microsoft/Phi-3.5-mini-instruct"
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda:0").eval()
    out = []
    for i in range(0, len(titles), 8):
        prompts = [tok.apply_chat_template([{"role": "user", "content":
                   f"Write the abstract (about 200 words, one paragraph, no title) of the research paper \"{t}\"."}],
                   tokenize=False, add_generation_prompt=True) for t in titles[i:i + 8]]
        e = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=320, do_sample=True, temperature=0.7, top_p=0.9,
                               pad_token_id=tok.pad_token_id)
        out += [tok.decode(x[e.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    del model
    torch.cuda.empty_cache()
    return [o for o in out if len(o.split()) >= 80]


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", nargs="+", required=True)
    ap.add_argument("--bib", required=True)
    ap.add_argument("--own", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-ai", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "report.md"), "w")

    def say(*x):
        line = " ".join(str(y) for y in x)
        print(line, flush=True)
        log.write(line + "\n")

    # 1. sources
    sources, meta, human = {}, [], []
    for e in bib_entries(a.bib):
        title = plain(e.get("title", ""))
        w = openalex(e) or {}
        arx = w.get("arxiv") or (re.sub(r"v\d+$", "", e["eprint"]) if e.get("eprint") else None)
        if not arx and title and e["type"] in ("article", "inproceedings", "misc", "techreport"):
            arx = arxiv_by_title(title)
        full = fetch_pdf(f"https://arxiv.org/pdf/{arx}") if arx else ""
        if arx:
            time.sleep(1)
        for u in ([] if full else w.get("pdfs", [])[:2]):
            full = fetch_pdf(u)
            if full:
                break
        text = (w.get("abstract") or "") + "\n" + full
        kind = "full text" if full else ("abstract" if w.get("abstract") else "none")
        meta.append(dict(key=e["key"], title=title, arxiv=arx, source=kind, words=len(text.split())))
        if text.strip():
            sources[e["key"]] = text
        if w.get("abstract") and (w.get("year") or 9999) <= 2021 and len(w["abstract"].split()) >= 100:
            human.append(dict(key=e["key"], title=title, text=w["abstract"]))
        say(f"source {e['key']}: {kind} ({len(text.split())} words){' arXiv ' + arx if arx else ''}")
    for p in a.own:
        sources["own:" + os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)] = open(p, encoding="utf-8").read()
    json.dump(meta, open(os.path.join(a.out, "sources.json"), "w"), indent=1)
    n_full = sum(m["source"] == "full text" for m in meta)
    n_abs = sum(m["source"] == "abstract" for m in meta)
    say(f"\n## Sources\n{len(meta)} cited works: {n_full} full texts, {n_abs} abstracts only, "
        f"{len(meta) - n_full - n_abs} not found; {len(a.own)} own public files.")

    # 2. similarity
    texts = {}
    for pdf in a.pdf:
        name = os.path.splitext(os.path.basename(pdf))[0]
        texts[name] = paper_text(pdf)
        r = similarity(texts[name], sources)
        cited = {k: v for k, v in r["per_source"].items() if not k.startswith("own:")}
        r_cited = similarity(texts[name], {k: v for k, v in sources.items() if not k.startswith("own:")})
        json.dump(r, open(os.path.join(a.out, f"similarity_{name}.json"), "w"), indent=1)
        say(f"\n## Similarity: {name}.pdf ({r['words']} words, reference list excluded, matches of >= {N} words)")
        say(f"similarity index, all sources: {r['index']}%")
        say(f"similarity index, cited works only: {r_cited['index']}%")
        say("largest single sources: " + ", ".join(f"{k} {v}%" for k, v in list(r["per_source"].items())[:10]))
        say(f"largest cited source: {max(cited.values()) if cited else 0}%")
        say(f"matched passages ({len(r['passages'])}), longest first:")
        for p in r["passages"][:40]:
            say(f"- [{p['source']}, {p['words']} words] {p['text'][:300]}")

    if a.no_ai:
        return
    # 3. AI-text screening
    import torch
    print("GPUs:", torch.cuda.device_count(), flush=True)
    paper_chunks = chunks(texts[os.path.splitext(os.path.basename(a.pdf[0]))[0]])
    human = human[:40]
    try:
        ai = generate_ai_abstracts([h["title"] for h in human[:24]])
    except Exception as ex:
        say(f"AI control set could not be generated ({type(ex).__name__}: {str(ex)[:200]})")
        ai = []
    sets = {"paper (main body)": paper_chunks, "human abstracts (cited, <= 2021)": [h["text"] for h in human],
            "AI-written abstracts (Phi-3.5)": ai}
    say(f"\n## AI-text screening\nchunks: paper {len(paper_chunks)}, human {len(human)}, AI {len(ai)}")
    allt = [t for v in sets.values() for t in v]
    res = {}
    for det, fn in (("desklib", desklib_scores), ("fakespot", lambda t: fakespot_scores(t)[0]),
                    ("binoculars", binoculars_scores)):
        try:
            sc = fn(allt)
        except Exception as ex:                                  # a detector that fails to load is reported, not fatal
            say(f"{det}: failed ({type(ex).__name__}: {str(ex)[:200]})")
            continue
        i, res[det] = 0, {}
        for name, v in sets.items():
            res[det][name] = sc[i:i + len(v)]
            i += len(v)
    json.dump(dict(sets=sets, scores=res), open(os.path.join(a.out, "ai_scores.json"), "w"), indent=1)
    for det, d in res.items():
        h, g = np.array(d["human abstracts (cited, <= 2021)"]), np.array(d["AI-written abstracts (Phi-3.5)"])
        if det == "binoculars" and len(g) == 0:
            flag = lambda x: x <= np.percentile(h, 5)
            say(f"\n{det} (lower = more machine-like; flagged below the 5th percentile of the human controls)")
        elif det == "binoculars":                                # lower = machine-like; threshold from the controls
            cand = np.sort(np.concatenate([h, g]))
            acc = [((h > c).mean() + (g <= c).mean()) / 2 for c in cand]
            thr = float(cand[int(np.argmax(acc))])
            flag = lambda x: x <= thr
            say(f"\n{det} (lower = more machine-like; threshold {thr:.3f} separates the controls with "
                f"balanced accuracy {max(acc):.2f})")
        else:
            flag = lambda x: x >= 0.5
            say(f"\n{det} (probability of AI text; flagged at >= 0.5)")
        for name, v in d.items():
            v = np.array(v)
            say(f"- {name}: mean {v.mean():.3f}, median {np.median(v):.3f}, flagged {100 * flag(v).mean():.0f}% "
                f"of {len(v)}")
        pv = np.array(d["paper (main body)"])
        order = np.argsort(pv) if det == "binoculars" else np.argsort(-pv)
        say("  most machine-like paper chunks:")
        for k in order[:5]:
            say(f"  - {pv[k]:.3f}: {sets['paper (main body)'][k][:160]}...")
    log.close()


if __name__ == "__main__":
    main()
