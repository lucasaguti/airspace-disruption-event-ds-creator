#!/usr/bin/env python3
"""
Event-first seed dataset builder (GDELT DOC 2.0 API) for airspace disruptions.

Implements the "event-first seed recipe":
  1) Pull GDELT items for corridor locality terms + aviation disruption intent bundles
  2) Rank by (unique domains, mentions proxy, cross-language volume)
  3) Cluster near-duplicates into one "event"
  4) Verify each seed event against corridor counts (count shock) and export CSV

This is intentionally v1:
  - broad query threshold
  - light clustering
  - verification is a *sanity check*, not a perfect truth labeler

Notes on GDELT DOC API:
  - By default searches a rolling recent window; you can control time using
    either TIMES PAN (e.g., 90d, 1y) OR startdatetime/enddatetime.
  - TIMESPAN cannot be combined with startdatetime/enddatetime.

References:
  - GDELT DOC 2.0 API Debuts! (TIMESPAN, ArtList, maxrecords up to 250)
  - DOC/GEO 2.0 updates (timespan=1y)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import gzip
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

DEFAULT_NEGATIVE_TERMS = [
    # Helps remove lots of non-disruption aviation news
    "plane crash", "air crash", "aircraft crash", "accident", "fatalities",
    "earnings", "ticket prices", "passenger", "airline shares", "CEO",
]

# ----------------------------- Utilities -----------------------------

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def parse_date(s: str) -> dt.datetime:
    # Accept YYYY-MM-DD (assumed UTC midnight)
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
    # remove boilerplate terms
    for w in ["breaking", "live", "update", "exclusive", "watch", "video"]:
        t = re.sub(rf"\b{w}\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canonical_domain(domain: str) -> str:
    d = (domain or "").lower().strip()
    d = re.sub(r"^www\.", "", d)
    return d

def seq_similarity(a: str, b: str) -> float:
    # 0..1 using SequenceMatcher
    return difflib.SequenceMatcher(None, a, b).ratio()

def chunk_date_ranges(start: dt.datetime, end: dt.datetime, days: int) -> List[Tuple[dt.datetime, dt.datetime]]:
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=days), end)
        out.append((cur, nxt))
        cur = nxt
    return out

# ----------------------------- Config Loading -----------------------------

def load_watchbox_terms(path: str) -> Dict[str, List[str]]:
    """
    Accepts JSON with:
      { "box_id": ["term1","term2", ...], ... }
    or:
      { "boxes": { "box_id": { "locality_terms": [...] } } }
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "boxes" in obj:
        out = {}
        for box_id, box in obj["boxes"].items():
            out[box_id] = list(box.get("locality_terms", []))
        return out
    # assume flat dict
    return {k: list(v) for k, v in obj.items()}

# ----------------------------- GDELT DOC Fetch -----------------------------

def build_query(locality_terms: List[str],
                intent_bundles: Dict[str, List[str]],
                negative_terms: List[str]) -> str:
    """
    Broad query:
      (locality OR locality OR ...) AND (intent OR intent OR ...) AND NOT (neg OR ...)
    """
    # Phrase-quote multiword terms.
    def qterm(x: str) -> str:
        x = x.strip()
        if " " in x:
            return f'"{x}"'
        return x

    loc = " OR ".join(qterm(t) for t in locality_terms if t.strip())
    intents = []
    for _, terms in intent_bundles.items():
        intents.extend(terms)
    intent = " OR ".join(qterm(t) for t in intents if t.strip())

    neg = " OR ".join(qterm(t) for t in negative_terms if t.strip())

    q = f"({loc}) ({intent})"
    if neg:
        q += f" -({neg})"
    return q

def fetch_gdelt_artlist(query: str,
                        start_dt: dt.datetime,
                        end_dt: dt.datetime,
                        maxrecords: int = 250,
                        sort: str = "DateAsc",
                        sleep_s: float = 0.5,
                        max_pages: int = 200) -> List[Dict[str, Any]]:
    """
    Fetch articles using startdatetime/enddatetime paging.
    We page by moving startdatetime forward to last seen seendate + 1 second.
    """
    assert maxrecords <= 250, "GDELT DOC API maxrecords is typically <= 250 in ArtList mode."
    results: List[Dict[str, Any]] = []

    cursor_start = start_dt
    pages = 0
    last_seen_urls = set()

    while cursor_start < end_dt and pages < max_pages:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(maxrecords),
            "sort": sort,
            "startdatetime": to_gdelt_dt(cursor_start),
            "enddatetime": to_gdelt_dt(end_dt),
        }
        r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(3.0)
            continue
        r.raise_for_status()
        data = r.json()

        articles = data.get("articles", []) or []
        if not articles:
            break

        # Dedup within & across pages
        new_articles = []
        for a in articles:
            url = a.get("url") or ""
            if not url or url in last_seen_urls:
                continue
            last_seen_urls.add(url)
            new_articles.append(a)

        if not new_articles:
            break

        results.extend(new_articles)
        pages += 1
        time.sleep(sleep_s)

        # Advance cursor
        # 'seendate' is typically like "2024-01-01 12:34:56" (UTC)
        last_dt = None
        for a in reversed(articles):
            sd = a.get("seendate") or a.get("sourceCollectionDate") or ""
            if sd:
                try:
                    # seendate sometimes "YYYYMMDDHHMMSS" in some modes; try both.
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
            # If we can't parse, stop to avoid infinite loops
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
        # Parse seendate with best-effort
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
    # Drop empties
    df = df[df["url"].astype(bool)]
    return df

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
    # Parse published_utc
    df["published_dt"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_dt"])
    # Time bucket
    bucket = f"{bucket_hours}H"
    df["bucket"] = df["published_dt"].dt.floor(bucket)

    agg = df.groupby(["box_id", "bucket", "title_norm"]).agg(
        total_articles=("url", "count"),
        unique_domains=("domain", pd.Series.nunique),
        unique_languages=("language", pd.Series.nunique),
        example_title=("title", "first"),
        min_published=("published_dt", "min"),
        max_published=("published_dt", "max"),
    ).reset_index()
    return agg

def select_seed_candidates(agg: pd.DataFrame,
                           min_total_articles: int = 3,
                           min_unique_domains: int = 1,
                           top_k_per_box: int = 500) -> pd.DataFrame:
    df = agg.copy()
    df = df[(df["total_articles"] >= min_total_articles) & (df["unique_domains"] >= min_unique_domains)]
    df = df.sort_values(["box_id", "unique_domains", "unique_languages", "total_articles"], ascending=[True, False, False, False])
    df["rank_in_box"] = df.groupby("box_id").cumcount() + 1
    df = df[df["rank_in_box"] <= top_k_per_box]
    return df

def cluster_candidates(cands: pd.DataFrame,
                       title_sim_thresh: float = 0.85,
                       merge_within_hours: int = 48) -> List[SeedEvent]:
    """
    Simple online clustering:
      - sort by time
      - maintain active clusters within window
      - attach to best matching cluster by title similarity
    """
    clusters = []  # list of dict cluster state
    window = dt.timedelta(hours=merge_within_hours)

    def cluster_add(cl, row):
        cl["rows"].append(row)
        cl["start"] = min(cl["start"], row["min_published"])
        cl["end"] = max(cl["end"], row["max_published"])

    # Convert times to python datetimes
    df = cands.copy()
    df["min_published"] = pd.to_datetime(df["min_published"], utc=True)
    df["max_published"] = pd.to_datetime(df["max_published"], utc=True)
    df = df.sort_values(["box_id", "min_published"])

    events: List[SeedEvent] = []

    for box_id, g in df.groupby("box_id"):
        active = []
        for _, r in g.iterrows():
            rp = r.to_dict()
            t = rp["min_published"].to_pydatetime()

            # expire old clusters
            active = [cl for cl in active if (t - cl["end"]) <= window]

            # find best match
            best = None
            best_sim = 0.0
            for cl in active:
                sim = seq_similarity(rp["title_norm"], cl["rep_title_norm"])
                if sim > best_sim:
                    best_sim = sim
                    best = cl

            if best is not None and best_sim >= title_sim_thresh:
                cluster_add(best, rp)
                # update representative if this row is "stronger"
                if (rp["unique_domains"], rp["total_articles"]) > (best["rep_unique_domains"], best["rep_total_articles"]):
                    best["rep_title"] = rp["example_title"]
                    best["rep_title_norm"] = rp["title_norm"]
                    best["rep_unique_domains"] = int(rp["unique_domains"])
                    best["rep_total_articles"] = int(rp["total_articles"])
            else:
                cl = {
                    "box_id": box_id,
                    "start": rp["min_published"].to_pydatetime(),
                    "end": rp["max_published"].to_pydatetime(),
                    "rep_title": rp["example_title"],
                    "rep_title_norm": rp["title_norm"],
                    "rep_unique_domains": int(rp["unique_domains"]),
                    "rep_total_articles": int(rp["total_articles"]),
                    "rows": [rp],
                }
                active.append(cl)

        # finalize clusters to SeedEvents
        for i, cl in enumerate(active):
            # aggregate metrics across cluster rows
            total_articles = int(sum(int(x["total_articles"]) for x in cl["rows"]))
            unique_domains = int(pd.Series([x for row in cl["rows"] for x in []]).nunique())  # placeholder
            # We'll set unique_domains using sum of per-row uniques as rough proxy (v1)
            unique_domains_proxy = int(sum(int(x["unique_domains"]) for x in cl["rows"]))
            unique_languages_proxy = int(sum(int(x["unique_languages"]) for x in cl["rows"]))

            event_id = f"seed:airspace:{box_id}:{cl['start'].strftime('%Y%m%dT%H%M%SZ')}"
            events.append(SeedEvent(
                event_id=event_id,
                box_id=box_id,
                start_utc=cl["start"].astimezone(dt.timezone.utc),
                end_utc=cl["end"].astimezone(dt.timezone.utc),
                rep_title=cl["rep_title"],
                rep_title_norm=cl["rep_title_norm"],
                total_articles=total_articles,
                unique_domains=unique_domains_proxy,
                unique_languages=unique_languages_proxy,
                top_urls=[],
                top_domains=[],
            ))

    return events

def attach_top_sources(events: List[SeedEvent], docs: pd.DataFrame, top_n: int = 5) -> List[SeedEvent]:
    df = docs.copy()
    df["published_dt"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_dt"])
    for ev in events:
        mask = (df["box_id"] == ev.box_id) & (df["published_dt"] >= pd.Timestamp(ev.start_utc)) & (df["published_dt"] <= pd.Timestamp(ev.end_utc))
        sub = df[mask]
        # if empty, expand window a bit
        if sub.empty:
            mask = (df["box_id"] == ev.box_id) & (df["published_dt"] >= pd.Timestamp(ev.start_utc - dt.timedelta(hours=12))) & (df["published_dt"] <= pd.Timestamp(ev.end_utc + dt.timedelta(hours=12)))
            sub = df[mask]

        if not sub.empty:
            # top urls by earliest
            sub = sub.sort_values("published_dt")
            ev.top_urls = sub["url"].head(top_n).tolist()
            ev.top_domains = sub["domain"].value_counts().head(top_n).index.tolist()
    return events

# ----------------------------- Verification vs Corridor Counts -----------------------------

def load_counts(path: str) -> pd.DataFrame:
    """
    Load corridor counts.
    Expected columns (flexible naming):
      - timestamp column: ts_utc | timestamp | ts
      - box_id column: box_id
      - count column: count | n | aircraft_count
    Supports .csv, .csv.gz, .parquet
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)

    # Normalize columns
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
    """
    Baseline = median of same hour-of-week over prior N weeks.
    Expects series columns: ts_dt, value
    """
    # target hour-of-week
    ts = pd.Timestamp(t0)
    how = int(ts.dayofweek) * 24 + int(ts.hour)
    s = series.copy()
    s["how"] = s["ts_dt"].dt.dayofweek * 24 + s["ts_dt"].dt.hour
    cutoff = ts - pd.Timedelta(weeks=weeks)
    hist = s[(s["ts_dt"] >= cutoff) & (s["ts_dt"] < ts) & (s["how"] == how)]
    if hist.empty:
        # fallback to prior week median
        hist = s[(s["ts_dt"] >= ts - pd.Timedelta(days=7)) & (s["ts_dt"] < ts)]
    if hist.empty:
        return float("nan")
    return float(hist["value"].median())

def verify_events_against_counts(events: List[SeedEvent],
                                counts_df: pd.DataFrame,
                                window_pad_hours: int = 6,
                                min_baseline: float = 20.0,
                                shock_thresh: float = 0.25) -> pd.DataFrame:
    """
    Verification:
      - if __global__ exists, use ratio=box/global (more robust)
      - else use box counts directly

    shock passes if abs(delta_pct_peak) >= shock_thresh and baseline >= min_baseline
    """
    dfc = counts_df.copy()
    has_global = (dfc["box_id"] == "__global__").any()

    # pivot global series for fast join
    if has_global:
        g = dfc[dfc["box_id"] == "__global__"][["ts_dt", "count"]].rename(columns={"count": "global"})
        dfc = dfc[dfc["box_id"] != "__global__"]
        dfc = dfc.merge(g, on="ts_dt", how="left")
        dfc["global"] = dfc["global"].fillna(method="ffill").fillna(method="bfill")
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

        # baseline at start
        baseline = compute_baseline(s.rename(columns={"value": "value"}), ev.start_utc)
        if math.isnan(baseline):
            baseline = float(s["value"].median())

        vmin = float(s["value"].min())
        vmax = float(s["value"].max())

        # delta relative to baseline: consider both drops and spikes
        delta_drop = (vmin - baseline) / baseline if baseline else 0.0
        delta_spike = (vmax - baseline) / baseline if baseline else 0.0
        # choose larger magnitude
        if abs(delta_drop) >= abs(delta_spike):
            delta_peak = delta_drop
            extremum = vmin
            direction = "DROP"
        else:
            delta_peak = delta_spike
            extremum = vmax
            direction = "SPIKE"

        # duration: time where value deviates beyond threshold
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
            "unique_domains_proxy": ev.unique_domains,
            "unique_languages_proxy": ev.unique_languages,
            "baseline": baseline,
            "extremum": extremum,
            "delta_pct_peak": float(delta_peak),
            "direction": direction,
            "duration_min": float(dur),
            "impact_pass": bool(impact_pass),
            "top_domains": json.dumps(ev.top_domains, ensure_ascii=False),
            "top_urls": json.dumps(ev.top_urls, ensure_ascii=False),
            "fin_label": "",  # leave blank for later
            "notes": "",
        })

    return pd.DataFrame(out_rows)

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build event-first seed dataset from GDELT DOC API and verify against corridor counts.")
    ap.add_argument("--watchbox-terms", required=True, help="JSON file mapping box_id -> locality_terms[]")
    ap.add_argument("--counts-path", required=True, help="Corridor counts file (.csv/.csv.gz/.parquet) with columns ts_utc, box_id, count (plus optional __global__).")
    ap.add_argument("--out-csv", default="truth_seed_v1.csv", help="Output CSV path.")
    ap.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="UTC end date YYYY-MM-DD (exclusive).")

    ap.add_argument("--chunk-days", type=int, default=14, help="Fetch GDELT in date chunks to reduce duplication/timeouts.")
    ap.add_argument("--maxrecords", type=int, default=250, help="GDELT maxrecords per request (<=250).")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between API calls.")
    ap.add_argument("--max-pages", type=int, default=200, help="Max paging iterations per chunk.")

    ap.add_argument("--min-total-articles", type=int, default=3, help="Ranking threshold (broad).")
    ap.add_argument("--min-unique-domains", type=int, default=1, help="Ranking threshold (broad).")
    ap.add_argument("--top-k-per-box", type=int, default=500, help="Keep top-K aggregates per box for clustering.")

    ap.add_argument("--title-sim", type=float, default=0.85, help="Title similarity threshold for clustering (0..1).")
    ap.add_argument("--merge-hours", type=int, default=48, help="Merge candidates within this many hours into an event cluster.")
    ap.add_argument("--bucket-hours", type=int, default=24, help="Aggregation bucket size for ranking.")

    ap.add_argument("--min-baseline", type=float, default=20.0, help="Min baseline value to accept a verified impact event.")
    ap.add_argument("--shock-thresh", type=float, default=0.25, help="Abs(delta_pct) threshold for a count shock (broad).")
    ap.add_argument("--window-pad-hours", type=int, default=6, help="Pad hours around event window for verification.")

    ap.add_argument("--save-raw", action="store_true", help="Save raw docs CSV for debugging.")
    ap.add_argument("--raw-dir", default="raw_docs", help="Directory to store raw docs.")
    args = ap.parse_args()

    watchbox_terms = load_watchbox_terms(args.watchbox_terms)

    start_dt = parse_date(args.start_date)
    end_dt = parse_date(args.end_date)

    # 1) Pull docs per box
    all_docs = []
    ranges = chunk_date_ranges(start_dt, end_dt, args.chunk_days)

    for box_id, locality_terms in watchbox_terms.items():
        if not locality_terms:
            continue
        q = build_query(locality_terms, DEFAULT_INTENT_BUNDLES, DEFAULT_NEGATIVE_TERMS)

        box_docs = []
        for (rs, re_) in ranges:
            arts = fetch_gdelt_artlist(
                query=q,
                start_dt=rs,
                end_dt=re_,
                maxrecords=args.maxrecords,
                sort="DateAsc",
                sleep_s=args.sleep,
                max_pages=args.max_pages,
            )
            df_part = articles_to_frame(arts, box_id=box_id)
            box_docs.append(df_part)

        if box_docs:
            df_box = pd.concat(box_docs, ignore_index=True)
            all_docs.append(df_box)

            if args.save_raw:
                safe_mkdir(args.raw_dir)
                outp = os.path.join(args.raw_dir, f"docs_{box_id}_{args.start_date}_{args.end_date}.csv")
                df_box.to_csv(outp, index=False)

        print(f"[docs] {box_id}: {sum(len(x) for x in box_docs)} rows")

    if not all_docs:
        raise SystemExit("No documents retrieved. Check terms/date range.")

    df_docs = pd.concat(all_docs, ignore_index=True).drop_duplicates(subset=["url"])
    print(f"[docs] total unique docs: {len(df_docs)}")

    # 2) Aggregate + rank
    agg = make_aggregates(df_docs, bucket_hours=args.bucket_hours)
    cands = select_seed_candidates(
        agg,
        min_total_articles=args.min_total_articles,
        min_unique_domains=args.min_unique_domains,
        top_k_per_box=args.top_k_per_box
    )
    print(f"[cands] {len(cands)} candidate aggregates")

    # 3) Cluster into seed events
    events = cluster_candidates(cands, title_sim_thresh=args.title_sim, merge_within_hours=args.merge_hours)
    events = attach_top_sources(events, df_docs, top_n=5)
    print(f"[events] {len(events)} seed clusters")

    # 4) Verify vs counts and export
    counts_df = load_counts(args.counts_path)
    df_out = verify_events_against_counts(
        events,
        counts_df,
        window_pad_hours=args.window_pad_hours,
        min_baseline=args.min_baseline,
        shock_thresh=args.shock_thresh
    )
    if df_out.empty:
        raise SystemExit("No events matched counts. Check box_id names match between watchbox terms and counts file.")
    # Keep broad: export both pass/fail so you can audit
    df_out = df_out.sort_values(["impact_pass", "unique_domains_proxy", "total_articles", "event_start_utc"], ascending=[False, False, False, True])
    df_out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(df_out)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()
