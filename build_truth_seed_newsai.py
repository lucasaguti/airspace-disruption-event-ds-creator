#!/usr/bin/env python3
"""
Build event-first seed dataset using NewsAPI.ai (Event Registry) Article Search API
and verify against corridor counts.

Auth/env:
  - NEWSAPI_AI_TOKEN   (required)
  - NEWS_AI_ENDPOINT   (optional; default https://newsapi.ai)

Design goals:
  - High recall under a token/search budget (default max burn 1500 out of 2000)
  - Resume-safe: every API page is cached to disk; restarts do NOT re-spend credits
  - Always produces partial raw caches; final CSV written only after clustering+verification

Notes on "credits":
  NewsAPI.ai bills by "search actions" (a request). Archive article search costs depend on
  the number of *years searched* (e.g., 5 tokens per searched year). See pricing page.

This script keeps your downstream pipeline/columns the same as your GDELT build:
  docs columns (raw docs CSV): box_id,url,title,title_norm,domain,language,published_utc,intent_bundle
  final CSV columns: event_id,box_id,event_start_utc,event_end_utc,rep_title,total_articles,unique_domains,
                     unique_languages,baseline,extremum,delta_pct_peak,direction,duration_min,impact_pass,
                     top_domains,top_urls,fin_label,notes
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from urllib.parse import urlparse

DEFAULT_NEWS_AI_ENDPOINT = "https://eventregistry.org"
ENDPOINT_ENV = "NEWS_AI_ENDPOINT"

# ---------------------------
# Intents (same idea as your GDELT script)
# ---------------------------

DEFAULT_INTENT_BUNDLES: Dict[str, List[str]] = {
    "AIRSPACE": [
        "airspace closed", "airspace closure", "airspace restricted", "flight ban",
        "no-fly zone", "airspace reopened", "airspace reopens"
    ],
    "OPS": [
        "flight cancellations", "rerouted", "rerouted flights", "diverted", "diverted flights",
        "ATC outage", "air traffic control outage", "radar failure", "NATS", "EUROCONTROL regulation"
    ],
    "CONFLICT": [
        "missile attack", "rocket attack", "airstrike", "drone attack", "intercepted drone",
        "military operation", "shootdown", "surface-to-air missile", "SAM"
    ],
    "GNSS": [
        "GPS jamming", "GNSS jamming", "GPS spoofing", "GNSS spoofing",
        "navigation interference", "signal interference"
    ],
}

DEFAULT_NEGATIVE_TERMS = [
    "plane crash", "air crash", "aircraft crash", "accident", "fatalities",
    "earnings", "ticket prices", "passenger", "CEO",
]

# ---------------------------
# Utilities
# ---------------------------

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_date_ymd(s: str) -> dt.datetime:
    # UTC midnight
    return dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

def chunk_date_ranges(start_dt: dt.datetime, end_dt: dt.datetime, chunk_days: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    ranges: List[Tuple[dt.datetime, dt.datetime]] = []
    cur = start_dt
    delta = dt.timedelta(days=chunk_days)
    while cur < end_dt:
        nxt = min(cur + delta, end_dt)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges

def split_ranges_on_year_boundary(ranges: List[Tuple[dt.datetime, dt.datetime]]) -> List[Tuple[dt.datetime, dt.datetime]]:
    """Split ranges so none cross a Jan 1 boundary (reduces archive token burn)."""
    out: List[Tuple[dt.datetime, dt.datetime]] = []
    for (a, b) in ranges:
        cur = a
        while cur < b:
            next_year = dt.datetime(cur.year + 1, 1, 1, tzinfo=dt.timezone.utc)
            nxt = min(b, next_year)
            out.append((cur, nxt))
            cur = nxt
    return out

def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    xs = list(xs)
    return [xs[i:i+n] for i in range(0, len(xs), n)]

_ws = re.compile(r"\s+")
def norm_title(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[“”\"'`]", "", s)
    s = re.sub(r"[^a-z0-9\s\-:/]", " ", s)
    s = _ws.sub(" ", s)
    return s.strip()

def sha1_json(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(b).hexdigest()

def load_watchbox_terms(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # enforce list-of-strings
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            continue
        out[k] = [str(x).strip() for x in v if str(x).strip()]
    return out

def select_intent_bundles(mode: str) -> Dict[str, List[str]]:
    if mode == "all":
        return DEFAULT_INTENT_BUNDLES
    if mode == "core":
        return {k: DEFAULT_INTENT_BUNDLES[k] for k in ["AIRSPACE", "OPS"]}
    if mode == "risk":
        return {k: DEFAULT_INTENT_BUNDLES[k] for k in ["CONFLICT", "GNSS"]}
    raise ValueError(mode)

# ---------------------------
# NewsAPI.ai (Event Registry) client
# ---------------------------

@dataclass
class TokenBudget:
    max_burn: int
    burned: int = 0

    def add(self, n: int) -> None:
        self.burned += int(n)

    def would_exceed(self, n: int) -> bool:
        return (self.burned + int(n)) > self.max_burn

def _newsai_base() -> str:
    return os.getenv(ENDPOINT_ENV, DEFAULT_NEWS_AI_ENDPOINT).strip().rstrip("/")


def _newsai_getarticles_url() -> str:
    base = _newsai_base()
    if base.endswith("/api/v1/article/getArticles"):
        return base
    if base.endswith("/api/v1/article"):
        return base + "/getArticles"
    return base + "/api/v1/article/getArticles"


def _newsai_token() -> str:
    tok = os.getenv("NEWSAPI_AI_TOKEN", "").strip()
    if not tok:
        raise SystemExit("Missing env var NEWSAPI_AI_TOKEN.")
    return tok

def _parse_req_tokens(v: Any) -> int:
    """
    Event Registry may return req-tokens like '1', '1.000', or even ''.
    Convert safely to an integer token count.
    """
    if v is None:
        return 0
    s = str(v).strip()
    if not s:
        return 0
    try:
        # handles "1.000"
        return int(float(s))
    except Exception:
        # last resort: keep digits only
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else 0

def _newsai_suggest_locations_fast(endpoint_base: str, api_key: str, text: str, *, lang: str = "eng") -> str:
    """
    Resolve a free-text place string to an Event Registry location URI using suggestLocationsFast.

    Returns:
      - location uri string (best guess) or "" if not found.
    Robust to API returning either:
      - {"locations": [...]}  (dict wrapper)
      - [...]                (top-level list)
    """
    base = (endpoint_base or "").rstrip("/")
    url = f"{base}/api/v1/suggestLocationsFast"

    payload = {
        "apiKey": api_key,
        "prefix": text,
        "lang": lang,          # REQUIRED by API (your error)
        "count": 1,            # we only need the best match
    }

    data, _ = _newsai_post_json(url, payload, timeout_s=30, sleep_s=0.0)

    # API sometimes returns list, sometimes dict wrapper
    if isinstance(data, list):
        locs = data
    elif isinstance(data, dict):
        locs = data.get("locations") or data.get("results") or []
    else:
        locs = []

    for loc in locs:
        if not isinstance(loc, dict):
            continue
        uri = loc.get("uri") or loc.get("wikiUri") or loc.get("conceptUri")
        if uri:
            return str(uri)

    return ""



def _newsai_suggest_concepts_fast(endpoint_base: str, api_key: str, text: str) -> Optional[str]:
    url = endpoint_base.rstrip("/") + "/api/v1/suggestConceptsFast"
    payload = {
        "action": "suggestConceptsFast",
        "apiKey": api_key,
        "prefix": text,
        "count": 1,
        "lang": "eng",        # REQUIRED
        "conceptLang": "eng", # keep (fine)
    }
    data, _ = _newsai_post_json(url, payload, timeout_s=30, sleep_s=0.0)
    cs = data.get("concepts") or []
    if not cs:
        return None
    return (cs[0].get("uri") or "").strip() or None

def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def resolve_location_uris(
    terms: List[str],
    *,
    endpoint_base: str,
    api_key: str,
    cache_path: str,
) -> List[str]:
    cache = _load_json(cache_path)
    out: List[str] = []
    for t in terms:
        k = f"loc::{t}".lower()
        if k in cache:
            if cache[k]:
                out.append(cache[k])
            continue
        uri = _newsai_suggest_locations_fast(endpoint_base, api_key, t)
        cache[k] = uri
        if uri:
            out.append(uri)
    _save_json_atomic(cache_path, cache)
    # unique preserve order
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def resolve_concept_uris(
    terms: List[str],
    *,
    endpoint_base: str,
    api_key: str,
    cache_path: str,
) -> List[str]:
    cache = _load_json(cache_path)
    out: List[str] = []
    for t in terms:
        k = f"con::{t}".lower()
        if k in cache:
            if cache[k]:
                out.append(cache[k])
            continue
        uri = _newsai_suggest_concepts_fast(endpoint_base, api_key, t)
        cache[k] = uri
        if uri:
            out.append(uri)
    _save_json_atomic(cache_path, cache)
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _newsai_post_json(
    url: str,
    payload: Dict[str, Any],
    *,
    timeout_s: int = 60,
    sleep_s: float = 0.0,
    max_tries: int = 6,
) -> Tuple[Dict[str, Any], int]:
    """
    POST JSON to NewsAPI.ai/EventRegistry; returns (data, req_tokens_int).
    - Handles HTML/non-JSON responses cleanly
    - Parses req-tokens header robustly (can be '1.000')
    - Retries transient failures with backoff
    """
    last_err: Optional[str] = None
    last_status: Optional[int] = None
    last_ct: str = ""
    last_head: str = ""

    for attempt in range(1, max_tries + 1):
        r = None
        try:
            r = requests.post(url, json=payload, timeout=timeout_s, allow_redirects=True)

            last_status = r.status_code
            last_ct = (r.headers.get("content-type") or "").lower()
            last_head = (r.text or "")[:400].replace("\n", " ").strip()

            # req-tokens can be missing, int-like, or float-like ("1.000")
            rt = r.headers.get("req-tokens", "") or ""
            _raw = (r.headers.get("req-tokens") or "0").strip()
            try:
                req_tokens = int(_raw)
            except ValueError:
                try:
                    req_tokens = int(float(_raw))
                except ValueError:
                    req_tokens = 0


            # Transient HTTP statuses
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code} ct={last_ct} head={last_head!r}"
                backoff = min(30.0, (2 ** (attempt - 1)) + (0.25 * attempt))
                time.sleep(backoff)
                continue

            # Hard HTTP errors: raise with helpful info
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} ct={last_ct} head={last_head!r}")

            # Must be JSON
            try:
                data = r.json()
            except Exception:
                raise RuntimeError(f"Non-JSON response (status={r.status_code}, ct={last_ct}) head={last_head!r}")

            # EventRegistry returns {"error": "..."} with 200 sometimes
            if isinstance(data, dict) and data.get("error"):
                msg = str(data.get("error"))
                raise RuntimeError(msg)

            if sleep_s > 0:
                time.sleep(sleep_s)

            return data, req_tokens

        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = str(e)
            if attempt < max_tries:
                time.sleep(min(15.0, 1.5 * attempt))
                continue
            raise RuntimeError(
                f"NewsAPI.ai request failed after retries: {last_err} "
                f"(status={last_status}, ct={last_ct}, head={last_head!r})"
            ) from e

    raise RuntimeError(f"NewsAPI.ai request failed: {last_err}")


def cache_path(raw_dir: str, cache_key: str) -> str:
    safe_mkdir(raw_dir)
    return os.path.join(raw_dir, f"{cache_key}.json")

def fetch_newsai_articles(
    *,
    query: Dict[str, Any],
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    maxrecords: int,
    max_pages: int,
    sleep_s: float,
    raw_dir: str,
    budget: TokenBudget,
    resume: bool,
) -> List[Dict[str, Any]]:
    """
    Page through NewsAPI.ai Article Search results for a given query object.
    Uses raw cache per page so reruns don't spend credits again.
    Returns a list of normalized article dicts: url,title,domain,language,published_utc.
    """
    base = _newsai_base()
    token = _newsai_token()
    url = _newsai_getarticles_url()

    out: List[Dict[str, Any]] = []

    # Paging loop
    page = 1
    while page <= max_pages:
        payload = {
            "apiKey": token,
            "query": query,
            "articlesCount": int(maxrecords),
            "articlesPage": int(page),
            "articlesSortBy": "date",
            "articlesSortByAsc": True,
        }

        cache_key = sha1_json({"url": url, "payload": payload})
        cp = cache_path(raw_dir, cache_key)

        if resume and os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                data = json.load(f)
            req_tokens = 0
        else:
            # Budget guard (best-effort): if we have no header estimate, still allow,
            # but stop if we exceed after the response.
            data, req_tokens = _newsai_post_json(url, payload, timeout_s=60, sleep_s=sleep_s)
            # Save atomically
            tmp = cp + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, cp)

        if req_tokens:
            if budget.would_exceed(req_tokens):
                print(f"[budget] would exceed max burn ({budget.burned}+{req_tokens}>{budget.max_burn}); stopping cleanly.", flush=True)
                break
            budget.add(req_tokens)

        arts = (data.get("articles") or {}).get("results") or []
        if not arts:
            break

        for a in arts:
            # Response commonly contains: url, title, lang, dateTimePub/date, source:{uri,title}, ...
            a_url = a.get("url") or a.get("link")
            if not a_url:
                continue
            title = a.get("title") or ""
            lang = a.get("lang") or a.get("language") or ""
            pub = a.get("dateTimePub") or a.get("date") or a.get("publishedAt") or ""
            # normalize pub to UTC ISO if possible
            pub_iso = ""
            if pub:
                try:
                    # Event Registry uses "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
                    if len(pub) == 10:
                        pub_dt = dt.datetime.strptime(pub, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
                    else:
                        # tolerate Z or offset
                        pub_dt = dt.datetime.fromisoformat(pub.replace("Z", "+00:00"))
                        if pub_dt.tzinfo is None:
                            pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
                        pub_dt = pub_dt.astimezone(dt.timezone.utc)
                    pub_iso = pub_dt.isoformat()
                except Exception:
                    pub_iso = ""

            src = a.get("source") or {}
            domain = src.get("uri") or src.get("title") or urlparse(a_url).netloc

            out.append({
                "url": a_url,
                "title": title,
                "domain": domain,
                "language": lang,
                "published_utc": pub_iso,
            })

        # Stop if we've reached the server-reported last page
        meta = data.get("articles") or {}
        pages = meta.get("pages")
        if pages and page >= int(pages):
            break

        page += 1

    return out

# ---------------------------
# Downstream pipeline (copied from your v1.1 structure)
# ---------------------------

def articles_to_frame(arts: List[Dict[str, Any]], box_id: str) -> pd.DataFrame:
    if not arts:
        return pd.DataFrame()

    rows = []
    for a in arts:
        url = a.get("url", "")
        if not url:
            continue
        title = a.get("title", "")
        rows.append({
            "box_id": box_id,
            "url": url,
            "title": title,
            "title_norm": norm_title(title),
            "domain": a.get("domain", ""),
            "language": a.get("language", ""),
            "published_utc": a.get("published_utc", ""),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["url"].astype(bool)]
    return df

def make_aggregates(df_docs: pd.DataFrame, bucket_hours: int = 24) -> pd.DataFrame:
    df = df_docs.copy()
    # Parse published_utc; drop rows without timestamp
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]
    df["bucket_utc"] = df["ts_utc"].dt.floor(f"{bucket_hours}H")

    agg = (
        df.groupby(["box_id", "bucket_utc"])
          .agg(
              total_articles=("url", "count"),
              unique_domains=("domain", lambda s: len(set([x for x in s if x]))),
              unique_languages=("language", lambda s: len(set([x for x in s if x]))),
              rep_title=("title", lambda s: s.iloc[0] if len(s) else ""),
          )
          .reset_index()
    )
    return agg

def select_seed_candidates(
    agg: pd.DataFrame,
    min_total_articles: int,
    min_unique_domains: int,
    top_k_per_box: int,
) -> pd.DataFrame:
    df = agg[(agg["total_articles"] >= min_total_articles) & (agg["unique_domains"] >= min_unique_domains)].copy()
    if df.empty:
        return df
    df = df.sort_values(["box_id", "total_articles", "unique_domains"], ascending=[True, False, False])
    # keep top K per box to cap compute
    df = df.groupby("box_id", as_index=False).head(top_k_per_box)
    return df

def jaccard(a: str, b: str) -> float:
    sa = set(norm_title(a).split())
    sb = set(norm_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def cluster_candidates(cands: pd.DataFrame, title_sim_thresh: float, merge_within_hours: int) -> List[Dict[str, Any]]:
    if cands.empty:
        return []
    # simple greedy clustering
    cands = cands.sort_values(["box_id", "bucket_utc"]).reset_index(drop=True)
    clusters: List[Dict[str, Any]] = []

    for _, row in cands.iterrows():
        placed = False
        for c in clusters:
            if c["box_id"] != row["box_id"]:
                continue
            # time proximity
            dt_hours = abs((row["bucket_utc"] - c["event_end_utc"]).total_seconds()) / 3600.0
            if dt_hours > merge_within_hours:
                continue
            if jaccard(row["rep_title"], c["rep_title"]) >= title_sim_thresh:
                # merge
                c["event_end_utc"] = max(c["event_end_utc"], row["bucket_utc"])
                c["total_articles"] += int(row["total_articles"])
                c["unique_domains"] = max(int(c["unique_domains"]), int(row["unique_domains"]))
                c["unique_languages"] = max(int(c["unique_languages"]), int(row["unique_languages"]))
                placed = True
                break
        if not placed:
            clusters.append({
                "event_id": f"{row['box_id']}::{row['bucket_utc'].isoformat()}",
                "box_id": row["box_id"],
                "event_start_utc": row["bucket_utc"],
                "event_end_utc": row["bucket_utc"],
                "rep_title": row["rep_title"],
                "total_articles": int(row["total_articles"]),
                "unique_domains": int(row["unique_domains"]),
                "unique_languages": int(row["unique_languages"]),
            })
    return clusters

def attach_top_sources(events: List[Dict[str, Any]], df_docs: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    if not events or df_docs.empty:
        return events
    df = df_docs.copy()
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]

    for e in events:
        box = e["box_id"]
        start = e["event_start_utc"]
        end = e["event_end_utc"] + pd.Timedelta(hours=24)
        dfe = df[(df["box_id"] == box) & (df["ts_utc"] >= start) & (df["ts_utc"] <= end)].copy()
        if dfe.empty:
            e["top_domains"] = "[]"
            e["top_urls"] = "[]"
            continue
        top_domains = (dfe["domain"].value_counts().head(top_n).index.tolist())
        top_urls = (dfe.drop_duplicates(subset=["url"])["url"].head(top_n).tolist())
        e["top_domains"] = json.dumps(top_domains)
        e["top_urls"] = json.dumps(top_urls)
    return events

def load_counts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df[df["ts_utc"].notna()].copy()
    return df

def compute_baseline_window(df_counts: pd.DataFrame, box_id: str, center: pd.Timestamp, hours_before: int = 72) -> float:
    start = center - pd.Timedelta(hours=hours_before)
    sub = df_counts[(df_counts["box_id"] == box_id) & (df_counts["ts_utc"] >= start) & (df_counts["ts_utc"] < center)]
    if sub.empty:
        return float("nan")
    return float(sub["count"].median())

def compute_extremum_window(df_counts: pd.DataFrame, box_id: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    sub = df_counts[(df_counts["box_id"] == box_id) & (df_counts["ts_utc"] >= start) & (df_counts["ts_utc"] <= end)]
    if sub.empty:
        return float("nan")
    return float(sub["count"].min())

def verify_events_against_counts(
    events: List[Dict[str, Any]],
    counts_df: pd.DataFrame,
    window_pad_hours: int,
    min_baseline: float,
    shock_thresh: float,
) -> pd.DataFrame:
    rows = []
    for e in events:
        box = e["box_id"]
        start = pd.to_datetime(e["event_start_utc"], utc=True)
        end = pd.to_datetime(e["event_end_utc"], utc=True) + pd.Timedelta(hours=window_pad_hours)
        baseline = compute_baseline_window(counts_df, box, start, hours_before=72)
        extremum = compute_extremum_window(counts_df, box, start, end)
        if math.isnan(baseline) or math.isnan(extremum) or baseline <= 0:
            continue
        delta_pct = (extremum - baseline) / baseline
        direction = "DOWN" if delta_pct < 0 else "UP"
        impact_pass = (baseline >= min_baseline) and (abs(delta_pct) >= shock_thresh)
        rows.append({
            "event_id": e["event_id"],
            "box_id": box,
            "event_start_utc": start.isoformat(),
            "event_end_utc": end.isoformat(),
            "rep_title": e.get("rep_title", ""),
            "total_articles": e.get("total_articles", 0),
            "unique_domains": e.get("unique_domains", 0),
            "unique_languages": e.get("unique_languages", 0),
            "baseline": baseline,
            "extremum": extremum,
            "delta_pct_peak": delta_pct,
            "direction": direction,
            "duration_min": int((end - start).total_seconds() // 60),
            "impact_pass": bool(impact_pass),
            "top_domains": e.get("top_domains", "[]"),
            "top_urls": e.get("top_urls", "[]"),
            "fin_label": "",
            "notes": "",
        })
    return pd.DataFrame(rows)

def build_complex_query_from_terms(
    locality_terms: List[str],
    intent_terms: List[str],
    *,
    endpoint_base: str,
    api_key: str,
    uri_cache_path: str,
    date_start: str,
    date_end: str,
) -> Dict[str, Any]:
    loc_uris = resolve_location_uris(
        locality_terms,
        endpoint_base=endpoint_base,
        api_key=api_key,
        cache_path=uri_cache_path,
    )
    intent_uris = resolve_concept_uris(
        intent_terms,
        endpoint_base=endpoint_base,
        api_key=api_key,
        cache_path=uri_cache_path,
    )

    q: Dict[str, Any] = {
        "$query": {
            "dateStart": date_start,
            "dateEnd": date_end,
        },
        "$filter": {
            "isDuplicate": "skipDuplicates",
        },
    }

    # These do NOT count as "keywords" in the free-plan sense
    if loc_uris:
        q["$query"]["locationUri"] = {"$or": loc_uris}
    if intent_uris:
        q["$query"]["conceptUri"] = {"$or": intent_uris}

    return q



# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build event-first seed dataset from NewsAPI.ai Article Search and verify against corridor counts.")
    ap.add_argument("--watchbox-terms", required=True, help="JSON file mapping box_id -> locality_terms[]")
    ap.add_argument("--counts-path", required=True, help="Counts file with ts_utc, box_id, count (optional __global__).")
    ap.add_argument("--out-csv", default="truth_seed_v1.csv", help="Output CSV path.")
    ap.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="UTC end date YYYY-MM-DD (exclusive).")

    ap.add_argument("--intent-mode", choices=["all", "core", "risk"], default="all")
    ap.add_argument("--intent-split", action="store_true")
    ap.add_argument("--intent-bundles", default="AIRSPACE,OPS,CONFLICT,GNSS")
    ap.add_argument("--intent-shard-size", type=int, default=8)
    ap.add_argument("--loc-shard-size", type=int, default=8)

    ap.add_argument("--chunk-days", type=int, default=365, help="Date chunk size. Use 365 for year-wide; script will split on year boundaries.")
    ap.add_argument("--maxrecords", type=int, default=100, help="NewsAPI.ai articlesCount per page.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Polite sleep between requests (NewsAPI.ai).")
    ap.add_argument("--max-pages", type=int, default=12, help="Max pages per query shard.")

    ap.add_argument("--use-negatives", action="store_true", help="Include negative terms (may reduce recall). Default off.")
    ap.add_argument("--max-token-burn", type=int, default=1500, help="Stop before exceeding this many tokens/search credits (best-effort).")
    ap.add_argument("--resume", action="store_true", help="Reuse cached raw pages in --raw-dir (prevents credit waste).")
    ap.add_argument("--save-raw", action="store_true", help="Save raw caches and per-box docs CSV for debugging.")
    ap.add_argument("--raw-dir", default="raw_newsai", help="Directory to store raw caches.")

    ap.add_argument("--min-total-articles", type=int, default=3)
    ap.add_argument("--min-unique-domains", type=int, default=1)
    ap.add_argument("--top-k-per-box", type=int, default=500)
    ap.add_argument("--title-sim", type=float, default=0.85)
    ap.add_argument("--merge-hours", type=int, default=48)
    ap.add_argument("--bucket-hours", type=int, default=24)
    ap.add_argument("--min-baseline", type=float, default=20.0)
    ap.add_argument("--shock-thresh", type=float, default=0.25)
    ap.add_argument("--window-pad-hours", type=int, default=6)

    args = ap.parse_args()

    watchbox_terms = load_watchbox_terms(args.watchbox_terms)
    start_dt = parse_date_ymd(args.start_date)
    end_dt = parse_date_ymd(args.end_date)

    intent_bundles = select_intent_bundles(args.intent_mode)

    ranges = chunk_date_ranges(start_dt, end_dt, args.chunk_days)
    ranges = split_ranges_on_year_boundary(ranges)  # IMPORTANT for archive token burn

    bundles_to_run = ["ALL"]
    if args.intent_split:
        bundles_to_run = [b.strip().upper() for b in args.intent_bundles.split(",") if b.strip()]
        for b in bundles_to_run:
            if b not in DEFAULT_INTENT_BUNDLES:
                raise SystemExit(f"Unknown bundle '{b}'. Valid: {list(DEFAULT_INTENT_BUNDLES.keys())}")

    budget = TokenBudget(max_burn=int(args.max_token_burn))
    all_docs: List[pd.DataFrame] = []

    for box_id, locality_terms in watchbox_terms.items():
        locality_terms = [t for t in locality_terms if t]
        if not locality_terms:
            continue

        loc_shards = chunk_list(locality_terms, args.loc_shard_size)

        box_docs_parts: List[pd.DataFrame] = []

        for bundle in bundles_to_run:
            if bundle == "ALL":
                intent_terms: List[str] = []
                for _, ts in intent_bundles.items():
                    intent_terms.extend(ts)
            else:
                intent_terms = list(DEFAULT_INTENT_BUNDLES[bundle])

            intent_shards = chunk_list(intent_terms, args.intent_shard_size)

            for (rs, re_) in ranges:
                date_start = rs.date().isoformat()
                date_end = re_.date().isoformat()

                for li, loc_shard in enumerate(loc_shards, 1):
                    for ii, intent_shard in enumerate(intent_shards, 1):
                        print(
                            f"[fetch] box={box_id} bundle={bundle} range={date_start}..{date_end} "
                            f"loc={li}/{len(loc_shards)} intent={ii}/{len(intent_shards)} "
                            f"L={len(loc_shard)} I={len(intent_shard)} burn={budget.burned}/{budget.max_burn}",
                            flush=True,
                        )

                        kw = len(loc_shard) + len(intent_shard) + (len(DEFAULT_NEGATIVE_TERMS) if args.use_negatives else 0)
                        if kw > 15:
                            raise SystemExit(f"Query would use {kw} keywords (>15 free-tier cap). Reduce shard sizes or disable negatives.")


                        endpoint_base = _newsai_base()
                        api_key = _newsai_token()
                        uri_cache_path = os.path.join(args.raw_dir, "_uri_cache.json")
                        
                        q = build_complex_query_from_terms(
                            loc_shard,
                            intent_shard,
                            endpoint_base=endpoint_base,
                            api_key=api_key,
                            uri_cache_path=uri_cache_path,
                            date_start=date_start,
                            date_end=date_end,
                        )


                        # Cache directory is per box/bundle/date shard so resumes don't waste credits
                        shard_dir = os.path.join(args.raw_dir, f"box={box_id}", f"bundle={bundle}", f"{date_start}_{date_end}", f"loc{li}_intent{ii}")
                        if args.save_raw:
                            safe_mkdir(shard_dir)

                        arts = fetch_newsai_articles(
                            query=q,
                            start_dt=rs,
                            end_dt=re_,
                            maxrecords=args.maxrecords,
                            max_pages=args.max_pages,
                            sleep_s=args.sleep,
                            raw_dir=shard_dir if args.save_raw else os.path.join(args.raw_dir, "_cache"),
                            budget=budget,
                            resume=bool(args.resume),
                        )

                        df_part = articles_to_frame(arts, box_id=box_id)
                        if df_part.empty:
                            continue
                        df_part["intent_bundle"] = bundle
                        box_docs_parts.append(df_part)

                        # If we're nearing budget, break early but keep what we have
                        if budget.burned >= budget.max_burn:
                            print("[budget] reached max burn; stopping fetch loops cleanly.", flush=True)
                            break
                    if budget.burned >= budget.max_burn:
                        break
                if budget.burned >= budget.max_burn:
                    break
            if budget.burned >= budget.max_burn:
                break

        if box_docs_parts:
            df_box = pd.concat(box_docs_parts, ignore_index=True)
            all_docs.append(df_box)

            if args.save_raw:
                safe_mkdir(args.raw_dir)
                outp = os.path.join(args.raw_dir, f"docs_{box_id}_{args.start_date}_{args.end_date}.csv")
                df_box.to_csv(outp, index=False)

        print(f"[docs] {box_id}: {sum(len(x) for x in box_docs_parts)} rows (burn={budget.burned})", flush=True)

        if budget.burned >= budget.max_burn:
            break

    if not all_docs:
        raise SystemExit("No documents retrieved. Check token/env, terms, and date range (archive access may be required).")

    df_docs = pd.concat(all_docs, ignore_index=True).drop_duplicates(subset=["url"])
    print(f"[docs] total unique docs: {len(df_docs)} (burn={budget.burned})", flush=True)

    agg = make_aggregates(df_docs, bucket_hours=args.bucket_hours)
    cands = select_seed_candidates(
        agg,
        min_total_articles=args.min_total_articles,
        min_unique_domains=args.min_unique_domains,
        top_k_per_box=args.top_k_per_box,
    )
    print(f"[cands] {len(cands)} candidate aggregates", flush=True)

    events = cluster_candidates(cands, title_sim_thresh=args.title_sim, merge_within_hours=args.merge_hours)
    events = attach_top_sources(events, df_docs, top_n=5)
    print(f"[events] {len(events)} seed clusters", flush=True)

    counts_df = load_counts(args.counts_path)
    df_out = verify_events_against_counts(
        events,
        counts_df,
        window_pad_hours=args.window_pad_hours,
        min_baseline=args.min_baseline,
        shock_thresh=args.shock_thresh,
    )

    if df_out.empty:
        raise SystemExit("No events matched counts. (Possible: not enough docs; budget stop; terms mismatch.)")

    df_out = df_out.sort_values(
        ["impact_pass", "unique_domains", "total_articles", "event_start_utc"],
        ascending=[False, False, False, True],
    )
    df_out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(df_out)} rows -> {args.out_csv} (burn={budget.burned})", flush=True)


if __name__ == "__main__":
    main()
