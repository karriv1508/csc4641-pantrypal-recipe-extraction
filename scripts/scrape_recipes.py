"""
CSC 4641 - Recipe Information Extraction (PantryPal)
Self-collected data: scrape recent recipes from recipe blogs with `recipe-scrapers`.

What it does
  1. Reads each site's sitemap to find recipe page URLs (newest first, last 5 years).
  2. Checks robots.txt before every request and waits between requests.
  3. Saves the raw HTML page (scraped_raw/html/) - the "original data".
  4. Parses it with recipe-scrapers (schema.org JSON-LD) and writes
       scraped_raw/scraped_recipes_raw.jsonl     (fields exactly as extracted)
       data_processed/scraped_clean.jsonl        (same schema as recipenlg_subset.jsonl)

Run (from the Final Project folder, on your own computer):
  pip install recipe-scrapers requests
  python scripts/scrape_recipes.py --per-site 120

For course research use only - scraped pages are not redistributed.
"""
import argparse, json, re, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import robotparser
from xml.etree import ElementTree as ET

import requests
from recipe_scrapers import scrape_html

sys.path.insert(0, str(Path(__file__).parent))
from clean_data import clean_line, TOKEN_RE, norm_token  # same cleaning as RecipeNLG

UA = "MSOE-CSC4641-student-project (course research; contact via MSOE)"
SITES = ["https://www.budgetbytes.com", "https://cookieandkate.com", "https://pinchofyum.com"]
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
SKIP_URL = re.compile(r"/(category|tag|author|page|about|contact|privacy|shop|meal-plan|roundup)s?/|/\d{4}/\d{2}/?$", re.I)


def get(session, url, rp, delay):
    if rp and not rp.can_fetch(UA, url):
        return None
    time.sleep(delay)
    r = session.get(url, timeout=20)
    if "charset" not in r.headers.get("Content-Type", "").lower():
        r.encoding = "utf-8"  # avoid requests' latin-1 default garbling ½ into Â½
    return r.text if r.status_code == 200 else None


def sitemap_urls(session, site, rp, delay, since):
    """Collect (url, lastmod) from post sitemaps, newest first."""
    found = []
    for idx in ("/sitemap_index.xml", "/sitemap.xml", "/wp-sitemap.xml"):
        xml = get(session, site + idx, rp, delay)
        if xml:
            break
    else:
        return found
    root = ET.fromstring(xml.encode())
    children = [l.text.strip() for l in root.findall("sm:sitemap/sm:loc", NS)]
    if not children:
        children = [site + idx]
    children = [c for c in children if re.search(r"post|recipe", c, re.I)] or children
    for child in children:
        xml = get(session, child, rp, delay)
        if not xml:
            continue
        for u in ET.fromstring(xml.encode()).findall("sm:url", NS):
            loc = u.find("sm:loc", NS).text.strip()
            lm = u.find("sm:lastmod", NS)
            lm = lm.text.strip() if lm is not None else ""
            if SKIP_URL.search(loc) or loc.rstrip("/") == site:
                continue
            if lm and lm[:10] < since:
                continue
            found.append((loc, lm))
    found.sort(key=lambda x: x[1], reverse=True)
    return found


def safe(fn):
    try:
        return fn()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--per-site", type=int, default=120)
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--sites", nargs="*", default=SITES)
    a = ap.parse_args()

    root = Path(a.root)
    html_dir = root / "scraped_raw" / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    (root / "data_processed").mkdir(exist_ok=True)
    since = (datetime.now(timezone.utc) - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    s = requests.Session()
    s.headers["User-Agent"] = UA
    raw_f = open(root / "scraped_raw" / "scraped_recipes_raw.jsonl", "w", encoding="utf-8")
    out_f = open(root / "data_processed" / "scraped_clean.jsonl", "w", encoding="utf-8")
    stats = {}

    for site in a.sites:
        host = re.sub(r"https?://(www\.)?", "", site).strip("/")
        rp = robotparser.RobotFileParser(site + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        urls = sitemap_urls(s, site, rp, a.delay, since)
        print(f"{host}: {len(urls)} candidate URLs", flush=True)
        st = {"candidates": len(urls), "kept": 0, "not_recipe": 0, "blocked_or_error": 0}
        for url, lastmod in urls:
            if st["kept"] >= a.per_site:
                break
            try:
                html = get(s, url, rp, a.delay)
            except requests.RequestException:
                html = None
            if not html:
                st["blocked_or_error"] += 1
                continue
            sc = safe(lambda: scrape_html(html, org_url=url, supported_only=False))
            ings = safe(sc.ingredients) if sc else None
            if not ings:
                st["not_recipe"] += 1
                continue
            n = st["kept"]
            rid = f"scraped-{host.split('.')[0]}-{n:04d}"
            (html_dir / f"{rid}.html").write_text(html, encoding="utf-8")
            raw = {"id": rid, "url": url, "site": host, "lastmod": lastmod,
                   "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "title": safe(sc.title), "ingredients": ings,
                   "instructions": safe(sc.instructions_list), "yields": safe(sc.yields),
                   "total_time": safe(sc.total_time)}
            raw_f.write(json.dumps(raw, ensure_ascii=False) + "\n")

            clean = [clean_line(x) for x in ings]
            clean = [x for x in clean if x and re.search(r"[A-Za-z]", x)]
            out_f.write(json.dumps({
                "id": rid, "source": "self-scraped", "origin": host, "link": url,
                "title": clean_line(raw["title"] or ""), "raw_ingredients": ings,
                "ingredients": clean,
                "tokens": [[norm_token(t) for t in TOKEN_RE.findall(x)] for x in clean],
                "tags": None, "label_type": "unlabeled",
            }, ensure_ascii=False) + "\n")
            st["kept"] += 1
            if st["kept"] % 20 == 0:
                print(f"  {host}: {st['kept']} recipes", flush=True)
        stats[host] = st
        print(f"{host}: {st}", flush=True)

    raw_f.close(); out_f.close()
    json.dump(stats, open(root / "scraped_raw" / "scrape_stats.json", "w"), indent=2)
    print("TOTAL recipes:", sum(v["kept"] for v in stats.values()))


if __name__ == "__main__":
    main()
