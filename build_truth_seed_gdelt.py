#!/usr/bin/env python3
"""
Event-first seed dataset builder (GDELT DOC 2.0 API) for airspace disruptions.
Adds --intent-mode flag to control query length/coverage.

Implements the "event-first seed recipe":
  1) Pull GDELT items for corridor locality terms + aviation disruption intent bundles
  2) Rank by (unique domains, mentions proxy, cross-language volume)
  3) Cluster near-duplicates into one "event"
  4) Verify each seed event against corridor counts (count shock) and export CSV

This v1.1 update adds robustness for real-world GDELT responses:
  - Handles non-JSON / blank / HTML responses with retries + backoff
  - Adds User-Agent header
  - Adds query-length guards: cap locality/intents terms to avoid 414/URI-too-long
  - Emits helpful diagnostics when the API returns non-JSON

No API key is required.

GDELT DOC API notes:
  - Use startdatetime/enddatetime (UTC) to backfill beyond rolling windows.
  - maxrecords is typically <= 250 per request in ArtList mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_INTENT_BUNDLES = {
    "AIRSPACE": [
        "airspace closed", "airspace closure", "airspace restricted", "flight ban",
        "FIR closed", "NOTAM", "no-fly zone", "restricted airspace", "airspace reopened"
    ],
    "OPS": [
        "flights suspended", "flight cancellations", "rerouted", "rerouted flights",
        "diverted", "diverted flights", "ATC outage", "air traffic control outage",
        "radar failure", "NATS", "EUROCONTROL regulation"
    ],
    "CONFLICT": [
        "missile", "drone", "air strike", "airstrike", "military exercise",
        "live-fire", "intercept", "shootdown"
    ],
    "GNSS": [
        "GPS jamming", "GNSS interference", "GPS spoofing", "GNSS spoofing"
    ]
}

def select_intent_bundles(mode: str) -> Dict[str, List[str]]:
    """
    intent-mode:
      - all  : AIRSPACE + OPS + CONFLICT + GNSS (default)
      - core : AIRSPACE + OPS (closures / NOTAM / reroutes / ATC outages)
      - risk : CONFLICT + GNSS (missiles/drones + GPS/GNSS interference)
    """
    m = (mode or "all").lower().strip()
    if m == "core":
        return {k: DEFAULT_INTENT_BUNDLES[k] for k in ["AIRSPACE", "OPS"] if k in DEFAULT_INTENT_BUNDLES}
    if m == "risk":
        return {k: DEFAULT_INTENT_BUNDLES[k] for k in ["CONFLICT", "GNSS"] if k in DEFAULT_INTENT_BUNDLES}
    return DEFAULT_INTENT_BUNDLES

DEFAULT_NEGATIVE_TERMS = [
    "plane crash", "air crash", "aircraft crash", "accident", "fatalities",
    "earnings", "ticket prices", "passenger", "CEO",
]

DEFAULT_HEADERS = {
    "User-Agent": "airspace-seed-builder/1.1 (GDELT DOC API)",
    "Accept": "application/json,text/plain,*/*",
}

# ----------------------------- Utilities -----------------------------

def parse_date_ymd(s: str) -> dt.datetime:
    # YYYY-MM-DD assumed UTC midnight
    return dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc)

def to_gdelt_dt(d: dt.datetime) -> str:
    d = d.astimezone(dt.timezone.utc)
    return d.strftime("%Y%m%d%H%M%S")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_title(t: str) -> str:
    t = (t or "").lower().strip()
    t = re.sub(r"[\u2018\u2019]", "'", t)
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for w in ["breaking", "live", "update", "exclusive", "watch", "video"]:
        t = re.sub(rf"\b{w}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonical_domain(domain: str) -> str:
    d = (domain or "").lower().strip()
    d = re.sub(r"^www\.", "", d)
    return d

def seq_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def chunk_date_ranges(start: dt.datetime, end: dt.datetime, days: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt
    return out

def looks_like_html(s: str) -> bool:
    s = (s or "").lstrip()
    return s.startswith("<!DOCTYPE") or s.startswith("<html") or s.startswith("<")

# ----------------------------- Config Loading -----------------------------

def load_watchbox_terms(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "boxes" in obj:
        out = {}
        for box_id, box in obj["boxes"].items():
            out[box_id] = list(box.get("locality_terms", []))
        return out
    return {k: list(v) for k, v in obj.items()}

# ----------------------------- Query Building -----------------------------

def build_query(
    locality_terms: List[str],
    intent_bundles: Dict[str, List[str]],
    negative_terms: List[str],
    max_locality_terms: int = 25,
    max_intent_terms: int = 40,
    max_negative_terms: int = 12,
) -> str:
    """
    Broad query:
      (locality OR ...) (intent OR ...) -(neg OR ...)
    with hard caps to keep URL length reasonable.
    """
    def qterm(x: str) -> str:
        x = x.strip()
        if not x:
            return ""
        if " " in x:
            return f'"{x}"'
        return x

    loc_terms = [t for t in locality_terms if t and t.strip()][:max_locality_terms]

    intents_flat: List[str] = []
    for _, terms in intent_bundles.items():
        intents_flat.extend([t for t in terms if t and t.strip()])
    intents_flat = intents_flat[:max_intent_terms]

    neg_terms = [t for t in negative_terms if t and t.strip()][:max_negative_terms]

    loc = " OR ".join(qterm(t) for t in loc_terms)
    intent = " OR ".join(qterm(t) for t in intents_flat)
    neg = " OR ".join(qterm(t) for t in neg_terms)

    q = f"({loc}) ({intent})"
    if neg:
        q += f" -({neg})"
    return q

# ----------------------------- GDELT DOC Fetch -----------------------------

def _request_json_with_retries(
    params: Dict[str, str],
    sleep_s: float,
    max_attempts: int = 6,
) -> Dict[str, Any]:
    """
    Robust GET with:
      - retry on 429/5xx
      - retry on blank/non-JSON bodies (common during transient issues)
      - helpful diagnostics if repeatedly non-JSON
    """
    backoff = sleep_s
    last_text = ""
    last_status = None
    last_url = None

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(
                GDELT_DOC_ENDPOINT,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=60,
            )
            last_status = r.status_code
            last_url = r.url
            text = r.text or ""
            last_text = text[:400].replace("\n", " ")

            if r.status_code == 429:
                time.sleep(max(2.0, backoff))
                backoff *= 1.6
                continue
            if 500 <= r.status_code <= 599:
                time.sleep(max(2.0, backoff))
                backoff *= 1.6
                continue

            # Some errors return HTML or empty body with 200
            if not text.strip():
                time.sleep(max(1.0, backoff))
                backoff *= 1.4
                continue
            if looks_like_html(text):
                # Often indicates 414/blocked/proxy or a server-side error page
                time.sleep(max(2.0, backoff))
                backoff *= 1.4
                continue

            # Parse JSON
            return r.json()

        except requests.exceptions.JSONDecodeError:
            time.sleep(max(2.0, backoff))
            backoff *= 1.4
            continue
        except requests.RequestException:
            time.sleep(max(2.0, backoff))
            backoff *= 1.6
            continue

    raise RuntimeError(
        "GDELT returned non-JSON repeatedly.\n"
        f"Last HTTP status: {last_status}\n"
        f"Last URL length: {len(last_url) if last_url else 'n/a'}\n"
        f"Last URL: {last_url}\n"
        f"Last response head: {last_text}\n"
        "Fixes:\n"
        "  - Reduce query size (lower --max-locality-terms / --max-intent-terms)\n"
        "  - Increase --sleep and/or reduce --max-pages\n"
        "  - Reduce --chunk-days (e.g., 7)\n"
    )

def fetch_gdelt_artlist(
    query: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    maxrecords: int,
    sleep_s: float,
    max_pages: int,
) -> List[Dict[str, Any]]:
    """
    Fetch articles using startdatetime/enddatetime paging.
    We page by moving startdatetime forward to last seen seendate + 1 second.
    """
    if maxrecords > 250:
        raise ValueError("GDELT DOC API maxrecords should be <= 250 for ArtList.")

    results: List[Dict[str, Any]] = []
    cursor_start = start_dt
    pages = 0
    seen_urls = set()

    while cursor_start < end_dt and pages < max_pages:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(maxrecords),
            "sort": "DateAsc",
            "startdatetime": to_gdelt_dt(cursor_start),
            "enddatetime": to_gdelt_dt(end_dt),
        }

        data = _request_json_with_retries(params=params, sleep_s=sleep_s)
        articles = data.get("articles", []) or []
        if not articles:
            break

        new_articles = []
        for a in articles:
            url = a.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            new_articles.append(a)

        if not new_articles:
            break

        results.extend(new_articles)
        pages += 1
        time.sleep(sleep_s)

        # Advance cursor by last seendate we can parse
        last_dt = None
        for a in reversed(articles):
            sd = a.get("seendate") or a.get("sourceCollectionDate") or ""
            if not sd:
                continue
            try:
                if re.match(r"^\d{14}$", sd):
                    last_dt = dt.datetime.strptime(sd, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
                else:
                    last_dt = dt.datetime.fromisoformat(sd.replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=dt.timezone.utc)
                    else:
                        last_dt = last_dt.astimezone(dt.timezone.utc)
                break
            except Exception:
                continue

        if last_dt is None:
            break
        cursor_start = last_dt + dt.timedelta(seconds=1)

        if len(articles) < maxrecords:
            break

    return results

def articles_to_frame(articles: List[Dict[str, Any]], box_id: str) -> pd.DataFrame:
    rows = []
    for a in articles:
        url = a.get("url") or ""
        title = a.get("title") or ""
        domain = canonical_domain(a.get("domain") or "")
        lang = a.get("language") or a.get("lang") or ""
        seendate = a.get("seendate") or ""

        published = None
        if seendate:
            try:
                if re.match(r"^\d{14}$", seendate):
                    published = dt.datetime.strptime(seendate, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
                else:
                    published = dt.datetime.fromisoformat(seendate.replace("Z", "+00:00"))
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=dt.timezone.utc)
                    else:
                        published = published.astimezone(dt.timezone.utc)
            except Exception:
                published = None

        rows.append({
            "box_id": box_id,
            "url": url,
            "title": title,
            "title_norm": normalize_title(title),
            "domain": domain,
            "language": lang,
            "published_utc": published.isoformat().replace("+00:00", "Z") if published else "",
        })

    df = pd.DataFrame(rows)
    return df[df["url"].astype(bool)]

# ----------------------------- Ranking + Clustering -----------------------------

@dataclasses.dataclass
class SeedEvent:
    event_id: str
    box_id: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    rep_title: str
    rep_title_norm: str
    total_articles: int
    unique_domains: int
    unique_languages: int
    top_urls: List[str]
    top_domains: List[str]

def make_aggregates(df_docs: pd.DataFrame, bucket_hours: int = 24) -> pd.DataFrame:
    df = df_docs.copy()
    df["published_dt"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_dt"])
    df["bucket"] = df["published_dt"].dt.floor(f"{bucket_hours}H")

    agg = df.groupby(["box_id", "bucket", "title_norm"]).agg(
        total_articles=("url", "count"),
        unique_domains=("domain", pd.Series.nunique),
        unique_languages=("language", pd.Series.nunique),
        example_title=("title", "first"),
        min_published=("published_dt", "min"),
        max_published=("published_dt", "max"),
    ).reset_index()
    return agg

def select_seed_candidates(
    agg: pd.DataFrame,
    min_total_articles: int = 3,
    min_unique_domains: int = 1,
    top_k_per_box: int = 500,
) -> pd.DataFrame:
    df = agg.copy()
    df = df[(df["total_articles"] >= min_total_articles) & (df["unique_domains"] >= min_unique_domains)]
    df = df.sort_values(
        ["box_id", "unique_domains", "unique_languages", "total_articles"],
        ascending=[True, False, False, False],
    )
    df["rank_in_box"] = df.groupby("box_id").cumcount() + 1
    return df[df["rank_in_box"] <= top_k_per_box]

def cluster_candidates(
    cands: pd.DataFrame,
    title_sim_thresh: float = 0.85,
    merge_within_hours: int = 48,
) -> List[SeedEvent]:
    """
    Simple clustering per box using title similarity + temporal proximity.
    """
    window = dt.timedelta(hours=merge_within_hours)
    events: List[SeedEvent] = []

    df = cands.copy()
    df["min_published"] = pd.to_datetime(df["min_published"], utc=True)
    df["max_published"] = pd.to_datetime(df["max_published"], utc=True)
    df = df.sort_values(["box_id", "min_published"])

    for box_id, g in df.groupby("box_id"):
        clusters = []

        def add_row(cl, row):
            cl["rows"].append(row)
            cl["start"] = min(cl["start"], row["min_published"])
            cl["end"] = max(cl["end"], row["max_published"])

        for _, r in g.iterrows():
            row = r.to_dict()
            t = row["min_published"].to_pydatetime()

            # expire old clusters
            clusters = [cl for cl in clusters if (t - cl["end"].to_pydatetime()) <= window]

            best = None
            best_sim = 0.0
            for cl in clusters:
                sim = seq_similarity(row["title_norm"], cl["rep_title_norm"])
                if sim > best_sim:
                    best_sim = sim
                    best = cl

            if best is not None and best_sim >= title_sim_thresh:
                add_row(best, row)
                # update rep if this candidate is stronger
                if (row["unique_domains"], row["total_articles"]) > (best["rep_unique_domains"], best["rep_total_articles"]):
                    best["rep_title"] = row["example_title"]
                    best["rep_title_norm"] = row["title_norm"]
                    best["rep_unique_domains"] = int(row["unique_domains"])
                    best["rep_total_articles"] = int(row["total_articles"])
            else:
                clusters.append({
                    "start": row["min_published"],
                    "end": row["max_published"],
                    "rep_title": row["example_title"],
                    "rep_title_norm": row["title_norm"],
                    "rep_unique_domains": int(row["unique_domains"]),
                    "rep_total_articles": int(row["total_articles"]),
                    "rows": [row],
                })

        # finalize
        for cl in clusters:
            start = cl["start"].to_pydatetime().astimezone(dt.timezone.utc)
            end = cl["end"].to_pydatetime().astimezone(dt.timezone.utc)
            event_id = f"seed:airspace:{box_id}:{start.strftime('%Y%m%dT%H%M%SZ')}"
            # v1: sum aggregates as a proxy; later we recompute from actual docs in attach_top_sources
            total_articles = int(sum(int(x["total_articles"]) for x in cl["rows"]))
            unique_domains = int(sum(int(x["unique_domains"]) for x in cl["rows"]))
            unique_languages = int(sum(int(x["unique_languages"]) for x in cl["rows"]))
            events.append(SeedEvent(
                event_id=event_id,
                box_id=box_id,
                start_utc=start,
                end_utc=end,
                rep_title=cl["rep_title"],
                rep_title_norm=cl["rep_title_norm"],
                total_articles=total_articles,
                unique_domains=unique_domains,
                unique_languages=unique_languages,
                top_urls=[],
                top_domains=[],
            ))

    return events

def attach_top_sources(events: List[SeedEvent], docs: pd.DataFrame, top_n: int = 5) -> List[SeedEvent]:
    df = docs.copy()
    df["published_dt"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_dt"])

    for ev in events:
        # Use +/- 12h slack to gather sources
        start = pd.Timestamp(ev.start_utc - dt.timedelta(hours=12))
        end = pd.Timestamp(ev.end_utc + dt.timedelta(hours=12))

        sub = df[(df["box_id"] == ev.box_id) & (df["published_dt"] >= start) & (df["published_dt"] <= end)].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("published_dt")
        ev.top_urls = sub["url"].head(top_n).tolist()
        ev.top_domains = sub["domain"].value_counts().head(top_n).index.tolist()

        # Replace proxies with actuals within window
        ev.total_articles = int(len(sub))
        ev.unique_domains = int(sub["domain"].nunique())
        ev.unique_languages = int(sub["language"].nunique())

    return events

# ----------------------------- Verification vs Corridor Counts -----------------------------

def load_counts(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("ts_utc") or cols.get("timestamp") or cols.get("ts") or cols.get("time")
    box_col = cols.get("box_id") or cols.get("box") or cols.get("corridor") or cols.get("region")
    cnt_col = cols.get("count") or cols.get("n") or cols.get("aircraft_count") or cols.get("aircraft")

    if not (ts_col and box_col and cnt_col):
        raise ValueError(f"Counts file must have timestamp, box_id, count columns. Found: {df.columns.tolist()}")

    df = df.rename(columns={ts_col: "ts_utc", box_col: "box_id", cnt_col: "count"})
    df["ts_dt"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts_dt"])
    df["box_id"] = df["box_id"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(float)
    return df[["ts_dt", "box_id", "count"]]

def compute_baseline(series: pd.DataFrame, t0: dt.datetime, weeks: int = 4) -> float:
    ts = pd.Timestamp(t0)
    how = int(ts.dayofweek) * 24 + int(ts.hour)
    s = series.copy()
    s["how"] = s["ts_dt"].dt.dayofweek * 24 + s["ts_dt"].dt.hour
    cutoff = ts - pd.Timedelta(weeks=weeks)
    hist = s[(s["ts_dt"] >= cutoff) & (s["ts_dt"] < ts) & (s["how"] == how)]
    if hist.empty:
        hist = s[(s["ts_dt"] >= ts - pd.Timedelta(days=7)) & (s["ts_dt"] < ts)]
    if hist.empty:
        return float("nan")
    return float(hist["value"].median())

def verify_events_against_counts(
    events: List[SeedEvent],
    counts_df: pd.DataFrame,
    window_pad_hours: int = 6,
    min_baseline: float = 20.0,
    shock_thresh: float = 0.25,
) -> pd.DataFrame:
    dfc = counts_df.copy()
    has_global = (dfc["box_id"] == "__global__").any()

    if has_global:
        g = dfc[dfc["box_id"] == "__global__"][["ts_dt", "count"]].rename(columns={"count": "global"})
        dfc = dfc[dfc["box_id"] != "__global__"]
        dfc = dfc.merge(g, on="ts_dt", how="left")
        dfc["global"] = dfc["global"].ffill().bfill()
        dfc["value"] = dfc["count"] / dfc["global"].replace(0, np.nan)
    else:
        dfc["value"] = dfc["count"]

    out_rows = []
    for ev in events:
        t_start = ev.start_utc - dt.timedelta(hours=window_pad_hours)
        t_end = ev.end_utc + dt.timedelta(hours=window_pad_hours)

        s = dfc[(dfc["box_id"] == ev.box_id) & (dfc["ts_dt"] >= pd.Timestamp(t_start)) & (dfc["ts_dt"] <= pd.Timestamp(t_end))].copy()
        if s.empty:
            continue
        s = s.sort_values("ts_dt")

        baseline = compute_baseline(s, ev.start_utc)
        if math.isnan(baseline):
            baseline = float(s["value"].median())

        vmin = float(s["value"].min())
        vmax = float(s["value"].max())

        delta_drop = (vmin - baseline) / baseline if baseline else 0.0
        delta_spike = (vmax - baseline) / baseline if baseline else 0.0
        if abs(delta_drop) >= abs(delta_spike):
            delta_peak = delta_drop
            extremum = vmin
            direction = "DROP"
        else:
            delta_peak = delta_spike
            extremum = vmax
            direction = "SPIKE"

        if baseline and baseline > 0:
            dev = (s["value"] - baseline) / baseline
            mask = dev.abs() >= shock_thresh
            if mask.any():
                dur = (s.loc[mask, "ts_dt"].max() - s.loc[mask, "ts_dt"].min()).total_seconds() / 60.0
            else:
                dur = 0.0
        else:
            dur = 0.0

        impact_pass = (baseline >= min_baseline) and (abs(delta_peak) >= shock_thresh)

        out_rows.append({
            "event_id": ev.event_id,
            "box_id": ev.box_id,
            "event_start_utc": ev.start_utc.isoformat().replace("+00:00", "Z"),
            "event_end_utc": ev.end_utc.isoformat().replace("+00:00", "Z"),
            "rep_title": ev.rep_title,
            "total_articles": ev.total_articles,
            "unique_domains": ev.unique_domains,
            "unique_languages": ev.unique_languages,
            "baseline": baseline,
            "extremum": extremum,
            "delta_pct_peak": float(delta_peak),
            "direction": direction,
            "duration_min": float(dur),
            "impact_pass": bool(impact_pass),
            "top_domains": json.dumps(ev.top_domains, ensure_ascii=False),
            "top_urls": json.dumps(ev.top_urls, ensure_ascii=False),
            "fin_label": "",
            "notes": "",
        })

    return pd.DataFrame(out_rows)

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build event-first seed dataset from GDELT DOC API and verify against corridor counts.")
    ap.add_argument("--watchbox-terms", required=True, help="JSON file mapping box_id -> locality_terms[]")
    ap.add_argument("--counts-path", required=True, help="Counts file with ts_utc, box_id, count (optional __global__).")
    ap.add_argument("--out-csv", default="truth_seed_v1.csv", help="Output CSV path.")
    ap.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="UTC end date YYYY-MM-DD (exclusive).")

    ap.add_argument(
      "--intent-mode",
      choices=["all", "core", "risk"],
      default="all",
      help="Which intent bundle set to use: all (default), core (AIRSPACE+OPS), risk (CONFLICT+GNSS).",
  )
  
    ap.add_argument("--chunk-days", type=int, default=7, help="Fetch GDELT in date chunks to reduce timeouts (try 7).")
    ap.add_argument("--maxrecords", type=int, default=250, help="GDELT maxrecords per request (<=250).")
    ap.add_argument("--sleep", type=float, default=1.2, help="Sleep seconds between API calls (increase if 429).")
    ap.add_argument("--max-pages", type=int, default=120, help="Max paging iterations per chunk (reduce if throttled).")

    ap.add_argument("--min-total-articles", type=int, default=3, help="Ranking threshold (broad).")
    ap.add_argument("--min-unique-domains", type=int, default=1, help="Ranking threshold (broad).")
    ap.add_argument("--top-k-per-box", type=int, default=500, help="Keep top-K aggregates per box for clustering.")

    ap.add_argument("--title-sim", type=float, default=0.85, help="Title similarity threshold for clustering.")
    ap.add_argument("--merge-hours", type=int, default=48, help="Merge candidates within this many hours into an event cluster.")
    ap.add_argument("--bucket-hours", type=int, default=24, help="Aggregation bucket size for ranking.")

    ap.add_argument("--min-baseline", type=float, default=20.0, help="Min baseline to accept impact event.")
    ap.add_argument("--shock-thresh", type=float, default=0.25, help="Abs(delta_pct) threshold for count shock (broad).")
    ap.add_argument("--window-pad-hours", type=int, default=6, help="Pad hours around event window for verification.")

    ap.add_argument("--max-locality-terms", type=int, default=25, help="Cap locality terms to avoid oversized queries.")
    ap.add_argument("--max-intent-terms", type=int, default=40, help="Cap intent terms to avoid oversized queries.")
    ap.add_argument("--max-negative-terms", type=int, default=12, help="Cap negative terms to avoid oversized queries.")

    ap.add_argument("--save-raw", action="store_true", help="Save raw docs CSV for debugging.")
    ap.add_argument("--raw-dir", default="raw_docs", help="Directory to store raw docs.")
    args = ap.parse_args()

    watchbox_terms = load_watchbox_terms(args.watchbox_terms)
    start_dt = parse_date_ymd(args.start_date)
    end_dt = parse_date_ymd(args.end_date)

    intent_bundles = select_intent_bundles(args.intent_mode)

    ranges = chunk_date_ranges(start_dt, end_dt, args.chunk_days)

    all_docs = []
    for box_id, locality_terms in watchbox_terms.items():
        if not locality_terms:
            continue

        q = build_query(
            locality_terms=locality_terms,
            intent_bundles=DEFAULT_INTENT_BUNDLES,
            intent_bundles=intent_bundles,
            negative_terms=DEFAULT_NEGATIVE_TERMS,
            max_locality_terms=args.max_locality_terms,
            max_intent_terms=args.max_intent_terms,
            max_negative_terms=args.max_negative_terms,
        )

        box_docs_parts = []
        for (rs, re_) in ranges:
            arts = fetch_gdelt_artlist(
                query=q,
                start_dt=rs,
                end_dt=re_,
                maxrecords=args.maxrecords,
                sleep_s=args.sleep,
                max_pages=args.max_pages,
            )
            df_part = articles_to_frame(arts, box_id=box_id)
            box_docs_parts.append(df_part)

        if box_docs_parts:
            df_box = pd.concat(box_docs_parts, ignore_index=True)
            all_docs.append(df_box)

            if args.save_raw:
                safe_mkdir(args.raw_dir)
                outp = os.path.join(args.raw_dir, f"docs_{box_id}_{args.start_date}_{args.end_date}.csv")
                df_box.to_csv(outp, index=False)

        print(f"[docs] {box_id}: {sum(len(x) for x in box_docs_parts)} rows")

    if not all_docs:
        raise SystemExit("No documents retrieved. Check terms/date range.")

    df_docs = pd.concat(all_docs, ignore_index=True).drop_duplicates(subset=["url"])
    print(f"[docs] total unique docs: {len(df_docs)}")

    agg = make_aggregates(df_docs, bucket_hours=args.bucket_hours)
    cands = select_seed_candidates(
        agg,
        min_total_articles=args.min_total_articles,
        min_unique_domains=args.min_unique_domains,
        top_k_per_box=args.top_k_per_box
    )
    print(f"[cands] {len(cands)} candidate aggregates")

    events = cluster_candidates(cands, title_sim_thresh=args.title_sim, merge_within_hours=args.merge_hours)
    events = attach_top_sources(events, df_docs, top_n=5)
    print(f"[events] {len(events)} seed clusters")

    counts_df = load_counts(args.counts_path)
    df_out = verify_events_against_counts(
        events,
        counts_df,
        window_pad_hours=args.window_pad_hours,
        min_baseline=args.min_baseline,
        shock_thresh=args.shock_thresh,
    )

    if df_out.empty:
        raise SystemExit("No events matched counts. Ensure box_id names match between watchbox terms and counts file.")

    df_out = df_out.sort_values(
        ["impact_pass", "unique_domains", "total_articles", "event_start_utc"],
        ascending=[False, False, False, True]
    )
    df_out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(df_out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
