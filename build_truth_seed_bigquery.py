#!/usr/bin/env python3
"""
build_truth_seed_bigquery.py

Build NEWS-ONLY (reported) event-first seed dataset using GDELT BigQuery (GKG),
optimized for time-precision + downstream scoring (counts-free).

Adds:
A) Time-precise events (default 6h buckets) and per-event timing:
   - onset_utc: first publication timestamp in (padded) event window
   - peak_utc: hour bucket with max article rate
   - tail_utc: last publication timestamp in (padded) event window

B) Counts-free "confidence" features:
   - unique_domains (already used as a gate; default >=2)
   - burstiness = peak_rate_per_hour / mean_rate_per_hour
   - top_domain_share (syndication proxy)
   - intent_strength (strong vs weak phrases, per-doc and per-event ratios)
   - anchor locality hit (anchor terms vs country-only terms, per-event ratios)

C) Counts-free "impact proxy" features:
   - volume spike magnitude (total articles, peak rate)
   - breadth across outlets (unique_domains, domain entropy proxy)
   - strong words presence (closed/ban/grounded/suspended/NOTAM/closure/etc.)
   - multi-day persistence (duration)

Design goals:
  - Fast historical capture using BigQuery
  - Resume-safe cache per shard
  - Produce an event CSV suitable for later enrichment with counts (Phase 2)

Requirements:
  pip install google-cloud-bigquery pyarrow pandas

Auth:
  gcloud auth application-default login
  or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON

Inputs:
  --watchbox-terms watchbox_terms.json  (box_id -> locality_terms[])

Outputs:
  - Raw cache per shard: <raw-dir>/.../bq_<hash>.csv.gz
  - Optional per-box docs CSV: <raw-dir>/docs_<box>_<start>_<end>.csv
  - Final CSV: --out-csv (NEWS-ONLY events with confidence/impact features)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

# BigQuery imports (kept optional until runtime)
try:
    from google.cloud import bigquery
    from google.api_core.exceptions import BadRequest
except Exception:
    bigquery = None  # type: ignore
    BadRequest = Exception  # type: ignore


DEFAULT_GDELT_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# ---------------------------
# Intent bundles
# ---------------------------

DEFAULT_INTENT_BUNDLES: Dict[str, List[str]] = {
    "AIRSPACE": [
        "airspace closed",
        "airspace closure",
        "airspace restricted",
        "flight ban",
        "no-fly zone",
        "airspace reopened",
        "airspace reopens",
    ],
    "OPS": [
        "flight cancellations",
        "rerouted",
        "rerouted flights",
        "diverted",
        "diverted flights",
        "atc outage",
        "air traffic control outage",
        "radar failure",
        "eurocontrol regulation",
        "nats",
    ],
    "CONFLICT": [
        "missile attack",
        "rocket attack",
        "airstrike",
        "drone attack",
        "intercepted drone",
        "military operation",
        "shootdown",
        "surface-to-air missile",
        "sam",
    ],
    "GNSS": [
        "gps jamming",
        "gnss jamming",
        "gps spoofing",
        "gnss spoofing",
        "navigation interference",
        "signal interference",
    ],
}

# Theme regex (kept broad-ish, but you can disable by requiring title intent)
DEFAULT_THEME_REGEX = r"(?i)(AVIATION|AIRPORT|AIRLINE|FLIGHT|AIR_TRAFFIC|RADAR|EUROCONTROL|NATS|NOTAM|AIRSPACE|OVERFLIGHT|GPS|GNSS|JAMM|SPOOF|MISSILE|ROCKET|DRONE)"

# Strong vs weak intent phrases (counts-free "intent strength")
# You can tune these without changing pipeline wiring.
STRONG_INTENT_PHRASES = [
    "airspace closed",
    "airspace closure",
    "flight ban",
    "no-fly zone",
    "airspace restricted",
    "atc outage",
    "air traffic control outage",
    "radar failure",
    "gps jamming",
    "gnss jamming",
    "gps spoofing",
    "gnss spoofing",
    "shootdown",
    "surface-to-air missile",
    "missile attack",
]
WEAK_INTENT_PHRASES = [
    "rerouted",
    "rerouted flights",
    "diverted",
    "diverted flights",
    "flight cancellations",
    "eurocontrol regulation",
    "nats",
    "airspace reopened",
    "airspace reopens",
    "navigation interference",
    "signal interference",
]

# Strong words for "impact proxy"
STRONG_WORDS = [
    "closed",
    "closure",
    "ban",
    "grounded",
    "grounding",
    "suspended",
    "suspends",
    "halted",
    "halts",
    "notam",
    "no-fly",
    "restricted",
    "restriction",
    "overflight",
    "shutdown",
    "outage",
    "failure",
    "jamming",
    "spoofing",
    "missile",
    "rocket",
    "drone",
    "attack",
    "strike",
    "intercept",
]


# ---------------------------
# Utilities
# ---------------------------

_ws = re.compile(r"\s+")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_date_ymd(s: str) -> dt.datetime:
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


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    xs = list(xs)
    return [xs[i : i + n] for i in range(0, len(xs), n)]


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


def _term_boundary_pattern(term: str) -> str:
    """
    Build a boundary-safe pattern that matches `term` as a token-ish unit.
    RE2-safe (no lookbehind).
    """
    esc = re.escape(term.strip().lower())
    return rf"(?:^|[^a-z0-9]){esc}(?:[^a-z0-9]|$)"


def _re_union(terms: List[str]) -> str:
    """
    Build RE2-safe case-insensitive union regex with non-alphanumeric boundaries.

    Note: We LOWER() the target field in SQL, but we keep (?i) for safety if caller
    uses it elsewhere.
    """
    clean = [t.strip().lower() for t in terms if (t or "").strip()]
    if not clean:
        return r"(?i)$^"
    parts = [_term_boundary_pattern(t) for t in clean]
    return r"(?i)(" + "|".join(parts) + r")"


def cache_path(raw_dir: str, cache_key: str) -> str:
    safe_mkdir(raw_dir)
    return os.path.join(raw_dir, f"bq_{cache_key}.csv.gz")


def jaccard(a: str, b: str) -> float:
    sa = set(norm_title(a).split())
    sb = set(norm_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------
# Locality anchor classification (heuristic)
# ---------------------------

ANCHOR_HINTS = (
    "sea",
    "strait",
    "gulf",
    "channel",
    "canal",
    "island",
    "islands",
    "bay",
    "peninsula",
    "archipelago",
    "ocean",
    "mediterranean",
    "aegean",
    "fir",
    "adiz",
    "gap",
    "tunnel",
)

# Lightweight country list (extend as you add regions)
COMMON_COUNTRIES = {
    "egypt",
    "turkey",
    "india",
    "pakistan",
    "oman",
    "yemen",
    "sri lanka",
    "maldives",
    "serbia",
    "croatia",
    "montenegro",
    "albania",
    "north macedonia",
    "kosovo",
    "bulgaria",
    "romania",
    "lithuania",
    "latvia",
    "estonia",
    "finland",
    "poland",
    "greece",
    "cyprus",
    "israel",
    "lebanon",
    "syria",
    "jordan",
    "japan",
    "taiwan",
    "philippines",
    "vietnam",
    "china",
    "spain",
    "morocco",
    "france",
    "united kingdom",
    "uk",
    "u.k.",
    "uae",
    "united arab emirates",
    "qatar",
    "bahrain",
    "kuwait",
    "iran",
}


def classify_locality_terms(terms: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (anchor_terms, country_terms).
    - country_terms: those likely to be country names (broad)
    - anchor_terms: seas/straits/cities/regions (more precise)
    """
    anchor: List[str] = []
    country: List[str] = []
    for t in terms:
        tl = t.strip().lower()
        if not tl:
            continue
        if tl in COMMON_COUNTRIES:
            country.append(t)
            continue
        if any(h in tl for h in ANCHOR_HINTS):
            anchor.append(t)
            continue
        # default: treat as anchor (cities/regions are usually desirable)
        anchor.append(t)
    return anchor, country


# ---------------------------
# BigQuery fetch
# ---------------------------

def _bq_client(project: str):
    if bigquery is None:
        raise SystemExit("Missing google-cloud-bigquery. Install: pip install google-cloud-bigquery pyarrow")
    return bigquery.Client(project=project)


def _sql_with_partitiontime(table: str) -> str:
    return f"""
    --standardSQL
    WITH base AS (
      SELECT
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(`DATE` AS STRING), 'UTC') AS ts_utc,
        DocumentIdentifier AS url,
        REGEXP_EXTRACT(DocumentIdentifier, r'^https?://([^/]+)/') AS domain,
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS page_title,
        V2Themes AS themes,
        V2Locations AS locations
      FROM `{table}`
      WHERE
        DATE(_PARTITIONTIME) >= @date_start
        AND DATE(_PARTITIONTIME) < @date_end
        AND DocumentIdentifier IS NOT NULL
    )
    SELECT
      url, domain, ts_utc, page_title, themes, locations
    FROM base
    WHERE
      REGEXP_CONTAINS(LOWER(COALESCE(locations,'')), @loc_re)
      AND (
        (@require_title_intent AND REGEXP_CONTAINS(LOWER(COALESCE(page_title,'')), @intent_re))
        OR
        (NOT @require_title_intent AND (
          REGEXP_CONTAINS(LOWER(COALESCE(page_title,'')), @intent_re)
          OR REGEXP_CONTAINS(COALESCE(themes,''), @theme_re)
        ))
      )
    """


def _sql_without_partitiontime(table: str) -> str:
    return f"""
    --standardSQL
    WITH base AS (
      SELECT
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(`DATE` AS STRING), 'UTC') AS ts_utc,
        DocumentIdentifier AS url,
        REGEXP_EXTRACT(DocumentIdentifier, r'^https?://([^/]+)/') AS domain,
        REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS page_title,
        V2Themes AS themes,
        V2Locations AS locations,
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(`DATE` AS STRING), 1, 8)) AS d
      FROM `{table}`
      WHERE DocumentIdentifier IS NOT NULL
    )
    SELECT
      url, domain, ts_utc, page_title, themes, locations
    FROM base
    WHERE
      d >= @date_start
      AND d < @date_end
      AND REGEXP_CONTAINS(LOWER(COALESCE(locations,'')), @loc_re)
      AND (
        (@require_title_intent AND REGEXP_CONTAINS(LOWER(COALESCE(page_title,'')), @intent_re))
        OR
        (NOT @require_title_intent AND (
          REGEXP_CONTAINS(LOWER(COALESCE(page_title,'')), @intent_re)
          OR REGEXP_CONTAINS(COALESCE(themes,''), @theme_re)
        ))
      )
    """


def fetch_gdelt_bq_articles(
    *,
    gcp_project: str,
    table: str,
    locality_terms: List[str],
    intent_terms: List[str],
    date_start: str,  # YYYY-MM-DD (inclusive)
    date_end: str,    # YYYY-MM-DD (exclusive)
    raw_dir: str,
    resume: bool,
    theme_regex: str,
    require_title_intent: bool = False,
    max_rows: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Query GDELT GKG via BigQuery and return (articles, bytes_processed_if_known).

    Article dict includes:
      url,title,domain,published_utc,locations,themes
    """
    loc_re = _re_union(locality_terms)
    intent_re = _re_union(intent_terms)

    params = {
        "table": table,
        "date_start": date_start,
        "date_end": date_end,
        "loc_re": loc_re,
        "intent_re": intent_re,
        "theme_re": theme_regex,
        "require_title_intent": bool(require_title_intent),
        "max_rows": max_rows,
        "dry_run": dry_run,
    }

    cache_key = sha1_json(params)
    cp = cache_path(raw_dir, cache_key)

    if resume and os.path.exists(cp):
        df = pd.read_csv(cp, compression="gzip")
        bytes_processed = None
    else:
        client = _bq_client(gcp_project)

        sql = _sql_with_partitiontime(table)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date_start", "DATE", date_start),
                bigquery.ScalarQueryParameter("date_end", "DATE", date_end),
                bigquery.ScalarQueryParameter("loc_re", "STRING", loc_re),
                bigquery.ScalarQueryParameter("intent_re", "STRING", intent_re),
                bigquery.ScalarQueryParameter("theme_re", "STRING", theme_regex),
                bigquery.ScalarQueryParameter("require_title_intent", "BOOL", bool(require_title_intent)),
            ],
            use_query_cache=True,
            dry_run=bool(dry_run),
        )

        try:
            job = client.query(sql, job_config=job_config)
        except BadRequest as e:
            msg = str(e)
            if "_PARTITIONTIME" in msg or "Unrecognized name: _PARTITIONTIME" in msg:
                sql = _sql_without_partitiontime(table)
                job = client.query(sql, job_config=job_config)
            else:
                raise

        if dry_run:
            bytes_processed = int(getattr(job, "total_bytes_processed", 0) or 0)
            return [], bytes_processed

        df = job.to_dataframe(create_bqstorage_client=False)
        bytes_processed = int(getattr(job, "total_bytes_processed", 0) or 0)

        if max_rows is not None and len(df) > int(max_rows):
            df = df.head(int(max_rows))

        df.to_csv(cp, index=False, compression="gzip")

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        url = str(r.get("url") or "").strip()
        if not url:
            continue
        title = r.get("page_title") or ""
        title = html.unescape(str(title)) if title else ""
        domain = str(r.get("domain") or "").strip() or urlparse(url).netloc

        pub_iso = ""
        try:
            ts = r.get("ts_utc")
            if pd.notna(ts):
                pub_iso = pd.to_datetime(ts, utc=True).isoformat()
        except Exception:
            pub_iso = ""

        out.append(
            {
                "url": url,
                "title": title,
                "domain": domain,
                "published_utc": pub_iso,
                "locations": str(r.get("locations") or ""),
                "themes": str(r.get("themes") or ""),
            }
        )

    return out, None


# ---------------------------
# Downstream pipeline
# ---------------------------

def articles_to_frame(
    arts: List[Dict[str, Any]],
    box_id: str,
    intent_bundle: str,
    intent_terms_all: List[str],
    theme_regex: str,
    anchor_re: Optional[re.Pattern],
    country_re: Optional[re.Pattern],
) -> pd.DataFrame:
    if not arts:
        return pd.DataFrame()

    theme_pat = re.compile(theme_regex)
    intent_terms_all_l = [t.lower() for t in intent_terms_all]

    strong_terms_l = [t.lower() for t in STRONG_INTENT_PHRASES]
    weak_terms_l = [t.lower() for t in WEAK_INTENT_PHRASES]
    strong_words_l = STRONG_WORDS  # already lower

    rows = []
    for a in arts:
        url = a.get("url", "")
        if not url:
            continue

        title = a.get("title", "") or ""
        tl = title.lower()

        # Was this doc title-intent hit? (even if query allowed theme-only)
        title_intent_hit = any(p in tl for p in intent_terms_all_l)

        # Theme hit proxy (counts-free)
        themes = (a.get("themes", "") or "")
        theme_hit = bool(theme_pat.search(themes)) if themes else False

        # Intent strength
        intent_strength = "none"
        if any(p in tl for p in strong_terms_l):
            intent_strength = "strong"
        elif any(p in tl for p in weak_terms_l):
            intent_strength = "weak"

        # Strong word presence (impact proxy)
        strong_word_count = sum(1 for w in strong_words_l if w in tl)
        strong_word_any = strong_word_count > 0

        # Anchor/country locality hit (using V2Locations)
        locations = (a.get("locations", "") or "").lower()
        anchor_hit = bool(anchor_re.search(locations)) if (anchor_re and locations) else False
        country_hit = bool(country_re.search(locations)) if (country_re and locations) else False

        rows.append(
            {
                "box_id": box_id,
                "url": url,
                "title": title,
                "title_norm": norm_title(title),
                "domain": a.get("domain", ""),
                "published_utc": a.get("published_utc", ""),
                "intent_bundle": intent_bundle,
                "title_intent_hit": bool(title_intent_hit),
                "theme_hit": bool(theme_hit),
                "intent_strength": intent_strength,
                "strong_word_count": int(strong_word_count),
                "strong_word_any": bool(strong_word_any),
                "anchor_hit": bool(anchor_hit),
                "country_hit": bool(country_hit),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["url"].astype(bool)]


def make_aggregates(df_docs: pd.DataFrame, bucket_hours: int) -> pd.DataFrame:
    df = df_docs.copy()
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]
    df["bucket_utc"] = df["ts_utc"].dt.floor(f"{int(bucket_hours)}h")

    agg = (
        df.groupby(["box_id", "bucket_utc"])
        .agg(
            total_articles=("url", "count"),
            unique_domains=("domain", lambda s: len(set([x for x in s if x]))),
            rep_title=("title", lambda s: s.iloc[0] if len(s) else ""),
            title_intent_ratio=("title_intent_hit", "mean"),
            strong_intent_ratio=("intent_strength", lambda s: float((s == "strong").mean()) if len(s) else 0.0),
            anchor_doc_ratio=("anchor_hit", "mean"),
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
    df = df.groupby("box_id", as_index=False).head(top_k_per_box)
    return df


@dataclass
class Event:
    event_id: str
    box_id: str
    event_start_bucket_utc: pd.Timestamp
    event_end_bucket_utc: pd.Timestamp
    rep_title: str


def cluster_candidates(cands: pd.DataFrame, title_sim_thresh: float, merge_within_hours: int) -> List[Event]:
    if cands.empty:
        return []
    cands = cands.sort_values(["box_id", "bucket_utc"]).reset_index(drop=True)

    clusters: List[Event] = []
    for _, row in cands.iterrows():
        placed = False
        for c in clusters:
            if c.box_id != row["box_id"]:
                continue
            dt_hours = abs((row["bucket_utc"] - c.event_end_bucket_utc).total_seconds()) / 3600.0
            if dt_hours > merge_within_hours:
                continue
            if jaccard(row["rep_title"], c.rep_title) >= title_sim_thresh:
                c.event_end_bucket_utc = max(c.event_end_bucket_utc, row["bucket_utc"])
                placed = True
                break
        if not placed:
            clusters.append(
                Event(
                    event_id=f"{row['box_id']}::{pd.to_datetime(row['bucket_utc'], utc=True).isoformat()}",
                    box_id=row["box_id"],
                    event_start_bucket_utc=pd.to_datetime(row["bucket_utc"], utc=True),
                    event_end_bucket_utc=pd.to_datetime(row["bucket_utc"], utc=True),
                    rep_title=str(row["rep_title"] or ""),
                )
            )
    return clusters


def _safe_entropy_from_counts(counts: List[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        ent -= p * math.log(p + 1e-12)
    return ent


def enrich_events_from_docs(
    events: List[Event],
    df_docs: pd.DataFrame,
    *,
    bucket_hours: int,
    onset_pad_hours: int,
    peak_bin_hours: int,
    top_n: int,
) -> List[Dict[str, Any]]:
    if not events or df_docs.empty:
        return []

    df = df_docs.copy()
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]

    bucket_pad = pd.Timedelta(hours=int(bucket_hours))
    onset_pad = pd.Timedelta(hours=int(onset_pad_hours))
    peak_bin = pd.Timedelta(hours=int(peak_bin_hours))

    out: List[Dict[str, Any]] = []

    for e in events:
        # Expand window for better onset/tail capture without changing clustering logic.
        win_start = e.event_start_bucket_utc - onset_pad
        win_end_excl = e.event_end_bucket_utc + bucket_pad

        dfe = df[(df["box_id"] == e.box_id) & (df["ts_utc"] >= win_start) & (df["ts_utc"] < win_end_excl)].copy()
        if dfe.empty:
            continue

        onset = pd.to_datetime(dfe["ts_utc"].min(), utc=True)
        tail = pd.to_datetime(dfe["ts_utc"].max(), utc=True)

        # Peak (max article rate) over peak_bin_hours
        dfe["peak_bucket_utc"] = dfe["ts_utc"].dt.floor(f"{int(peak_bin_hours)}h")
        peak_counts = dfe.groupby("peak_bucket_utc").size().sort_values(ascending=False)
        peak_bucket = pd.to_datetime(peak_counts.index[0], utc=True)
        peak_rate = int(peak_counts.iloc[0])

        duration_hours = max(1.0, (tail - onset).total_seconds() / 3600.0)
        mean_rate = float(len(dfe)) / duration_hours
        burstiness = float(peak_rate) / (mean_rate + 1e-9)

        # Domain stats
        dom_counts = dfe["domain"].fillna("").astype(str)
        vc = dom_counts.value_counts()
        unique_domains = int((vc.index != "").sum()) if len(vc) else 0
        top_domain = str(vc.index[0]) if len(vc) else ""
        top_domain_share = float(vc.iloc[0]) / float(len(dfe)) if len(dfe) else 0.0
        domain_entropy = _safe_entropy_from_counts([int(x) for x in vc.values.tolist()]) if len(vc) else 0.0

        # Intent strength ratios + strong words
        strong_ratio = float((dfe["intent_strength"] == "strong").mean()) if len(dfe) else 0.0
        weak_ratio = float((dfe["intent_strength"] == "weak").mean()) if len(dfe) else 0.0
        title_intent_ratio = float(dfe["title_intent_hit"].mean()) if len(dfe) else 0.0
        strong_word_any_ratio = float(dfe["strong_word_any"].mean()) if len(dfe) else 0.0
        strong_word_count_sum = int(dfe["strong_word_count"].sum())

        # Locality anchor/country ratios (per-doc flags)
        anchor_hit_ratio = float(dfe["anchor_hit"].mean()) if len(dfe) else 0.0
        country_hit_ratio = float(dfe["country_hit"].mean()) if len(dfe) else 0.0

        # Persistence (impact proxy)
        span_days = float((tail - onset).total_seconds() / 86400.0)
        multi_day = span_days >= 1.0

        # "Impact proxy" score (counts-free; heuristic)
        # Tuned to prefer: multi-outlet + burst + strong language + persistence.
        impact_score = (
            0.60 * math.log1p(len(dfe))
            + 0.90 * math.log1p(unique_domains)
            + 0.70 * math.log1p(peak_rate)
            + 0.35 * min(burstiness, 20.0)
            + 1.00 * (1.0 if strong_ratio > 0 else 0.0)
            + 0.70 * (1.0 if strong_word_any_ratio >= 0.3 else 0.0)
            + 0.60 * (1.0 if multi_day else 0.0)
        )

        # "Confidence" score: locality anchor + title intent + multi-outlet + not heavily syndicated
        syndicated_flag = top_domain_share >= 0.75
        confidence_score = (
            0.80 * min(1.0, unique_domains / 4.0)
            + 0.60 * min(1.0, title_intent_ratio)
            + 0.50 * min(1.0, anchor_hit_ratio)
            - 0.60 * (1.0 if syndicated_flag else 0.0)
        )

        # Top sources
        top_domains = vc.head(top_n).index.tolist()
        top_urls = dfe.drop_duplicates(subset=["url"])["url"].head(top_n).tolist()

        out.append(
            {
                "event_id": e.event_id,
                "box_id": e.box_id,
                "rep_title": e.rep_title,

                # Time-precise fields
                "onset_utc": onset.isoformat(),
                "peak_utc": peak_bucket.isoformat(),
                "tail_utc": tail.isoformat(),

                # Legacy-style window (exclusive end)
                "event_start_utc": (e.event_start_bucket_utc).isoformat(),
                "event_end_utc": (e.event_end_bucket_utc + bucket_pad).isoformat(),
                "duration_min": int((tail - onset).total_seconds() // 60),

                # Volume / breadth
                "total_articles": int(len(dfe)),
                "unique_domains": int(unique_domains),
                "top_domain": top_domain,
                "top_domain_share": float(top_domain_share),
                "domain_entropy": float(domain_entropy),

                # Rates / burst
                "peak_rate_per_peakbin": int(peak_rate),
                "peak_bin_hours": int(peak_bin_hours),
                "mean_rate_per_hour": float(mean_rate),
                "burstiness": float(burstiness),

                # Intent / language strength
                "title_intent_ratio": float(title_intent_ratio),
                "strong_intent_ratio": float(strong_ratio),
                "weak_intent_ratio": float(weak_ratio),
                "strong_word_any_ratio": float(strong_word_any_ratio),
                "strong_word_count_sum": int(strong_word_count_sum),

                # Locality confidence
                "anchor_hit_ratio": float(anchor_hit_ratio),
                "country_hit_ratio": float(country_hit_ratio),

                # Impact proxy & confidence
                "multi_day": bool(multi_day),
                "span_days": float(span_days),
                "impact_score": float(impact_score),
                "confidence_score": float(confidence_score),
                "syndicated_flag": bool(syndicated_flag),

                # Evidence lists
                "top_domains": json.dumps(top_domains),
                "top_urls": json.dumps(top_urls),

                # Placeholders for Phase 2 (counts-derived)
                "baseline": float("nan"),
                "extremum": float("nan"),
                "delta_pct_peak": float("nan"),
                "direction": "",
                "impact_pass": "",

                "fin_label": "",
                "notes": "",
            }
        )

    return out


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build NEWS-ONLY seed dataset from GDELT BigQuery (GKG), with timing + confidence/impact features.")
    ap.add_argument("--gcp-project", required=True, help="GCP project ID used for BigQuery billing/auth.")
    ap.add_argument("--gdelt-table", default=DEFAULT_GDELT_TABLE, help=f"BigQuery table (default: {DEFAULT_GDELT_TABLE})")

    ap.add_argument("--watchbox-terms", required=True, help="JSON file mapping box_id -> locality_terms[]")

    ap.add_argument("--out-csv", default="news_seed_v2.csv", help="Output CSV path.")
    ap.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="UTC end date YYYY-MM-DD (exclusive).")

    ap.add_argument("--intent-mode", choices=["all", "core", "risk"], default="core")
    ap.add_argument("--intent-split", action="store_true")
    ap.add_argument("--intent-bundles", default="AIRSPACE,OPS,CONFLICT,GNSS")
    ap.add_argument("--intent-shard-size", type=int, default=999)
    ap.add_argument("--loc-shard-size", type=int, default=999)

    ap.add_argument("--chunk-days", type=int, default=31, help="Date chunk size (default 31).")
    ap.add_argument("--theme-regex", default=DEFAULT_THEME_REGEX, help="Regex for aviation/ops-ish themes (broad filter).")

    ap.add_argument("--require-title-intent", action="store_true", help="Only keep docs whose PAGE_TITLE matches intent terms (disables theme-only matches).")

    ap.add_argument("--resume", action="store_true", help="Reuse cached shard results (prevents re-querying BigQuery).")
    ap.add_argument("--save-raw", action="store_true", help="Save per-shard caches and per-box docs CSV.")
    ap.add_argument("--raw-dir", default="raw_bq", help="Directory to store shard caches and raw docs.")

    # Time precision
    ap.add_argument("--bucket-hours", type=int, default=6, help="Aggregation bucket size in hours (default 6).")
    ap.add_argument("--onset-pad-hours", type=int, default=12, help="Pad window backwards to capture earlier onset/tail (default 12).")
    ap.add_argument("--peak-bin-hours", type=int, default=1, help="Bin size for peak rate calculation in hours (default 1).")

    # Candidate gating
    ap.add_argument("--min-total-articles", type=int, default=3, help="Minimum articles in a bucket to be a candidate (default 3).")
    ap.add_argument("--min-unique-domains", type=int, default=2, help="Minimum unique domains in a bucket (default 2).")
    ap.add_argument("--top-k-per-box", type=int, default=800, help="Max candidate buckets per box (default 800).")

    # Clustering
    ap.add_argument("--title-sim", type=float, default=0.75, help="Jaccard similarity threshold for merging candidate buckets (default 0.75).")
    ap.add_argument("--merge-hours", type=int, default=24, help="Max gap (hours) for merging buckets into same event (default 24).")

    # Optional post-filters (events)
    ap.add_argument("--min-impact-score", type=float, default=float("-inf"), help="Drop events with impact_score below this (default disabled).")
    ap.add_argument("--min-confidence-score", type=float, default=float("-inf"), help="Drop events with confidence_score below this (default disabled).")
    ap.add_argument("--max-top-domain-share", type=float, default=0.0, help="If >0, drop events with top_domain_share >= this (syndication filter).")

    ap.add_argument("--dry-run", action="store_true", help="BigQuery dry run (estimate bytes scanned), no output files.")
    ap.add_argument("--max-rows-per-shard", type=int, default=0, help="Optional cap on rows per shard (0 = no cap).")

    args = ap.parse_args()

    watchbox_terms = load_watchbox_terms(args.watchbox_terms)
    start_dt = parse_date_ymd(args.start_date)
    end_dt = parse_date_ymd(args.end_date)

    intent_bundles = select_intent_bundles(args.intent_mode)
    ranges = chunk_date_ranges(start_dt, end_dt, int(args.chunk_days))

    bundles_to_run = ["ALL"]
    if args.intent_split:
        bundles_to_run = [b.strip().upper() for b in args.intent_bundles.split(",") if b.strip()]
        for b in bundles_to_run:
            if b not in DEFAULT_INTENT_BUNDLES:
                raise SystemExit(f"Unknown bundle '{b}'. Valid: {list(DEFAULT_INTENT_BUNDLES.keys())}")

    # Flatten intent terms for scoring
    intent_terms_all: List[str] = []
    for _, ts in intent_bundles.items():
        intent_terms_all.extend(ts)

    all_docs: List[pd.DataFrame] = []
    safe_mkdir(args.raw_dir)

    for box_id, locality_terms in watchbox_terms.items():
        locality_terms = [t for t in locality_terms if t]
        if not locality_terms:
            continue

        anchor_terms, country_terms = classify_locality_terms(locality_terms)
        anchor_pat = re.compile(_re_union(anchor_terms)) if anchor_terms else None
        country_pat = re.compile(_re_union(country_terms)) if country_terms else None

        loc_shards = chunk_list(locality_terms, int(args.loc_shard_size))
        box_docs_parts: List[pd.DataFrame] = []

        for bundle in bundles_to_run:
            if bundle == "ALL":
                intent_terms: List[str] = []
                for _, ts in intent_bundles.items():
                    intent_terms.extend(ts)
            else:
                intent_terms = list(DEFAULT_INTENT_BUNDLES[bundle])

            intent_shards = chunk_list(intent_terms, int(args.intent_shard_size))

            for (rs, re_) in ranges:
                date_start = rs.date().isoformat()
                date_end = re_.date().isoformat()

                for li, loc_shard in enumerate(loc_shards, 1):
                    for ii, intent_shard in enumerate(intent_shards, 1):
                        shard_dir = os.path.join(
                            args.raw_dir,
                            f"box={box_id}",
                            f"bundle={bundle}",
                            f"{date_start}_{date_end}",
                            f"loc{li}_intent{ii}",
                        )
                        if args.save_raw:
                            safe_mkdir(shard_dir)

                        print(
                            f"[fetch] box={box_id} bundle={bundle} range={date_start}..{date_end} "
                            f"loc={li}/{len(loc_shards)} intent={ii}/{len(intent_shards)} "
                            f"L={len(loc_shard)} I={len(intent_shard)}",
                            flush=True,
                        )

                        max_rows = int(args.max_rows_per_shard) if int(args.max_rows_per_shard) > 0 else None

                        arts, bytes_scanned = fetch_gdelt_bq_articles(
                            gcp_project=args.gcp_project,
                            table=args.gdelt_table,
                            locality_terms=loc_shard,
                            intent_terms=intent_shard,
                            date_start=date_start,
                            date_end=date_end,
                            raw_dir=shard_dir if args.save_raw else os.path.join(args.raw_dir, "_cache"),
                            resume=bool(args.resume),
                            theme_regex=str(args.theme_regex),
                            require_title_intent=bool(args.require_title_intent),
                            max_rows=max_rows,
                            dry_run=bool(args.dry_run),
                        )

                        if args.dry_run:
                            print(f"[dry-run] estimated bytes scanned: {bytes_scanned or 0}", flush=True)
                            continue

                        df_part = articles_to_frame(
                            arts,
                            box_id=box_id,
                            intent_bundle=bundle,
                            intent_terms_all=intent_terms_all,
                            theme_regex=str(args.theme_regex),
                            anchor_re=anchor_pat,
                            country_re=country_pat,
                        )
                        if df_part.empty:
                            continue
                        box_docs_parts.append(df_part)

        if args.dry_run:
            continue

        if box_docs_parts:
            df_box = pd.concat(box_docs_parts, ignore_index=True)
            all_docs.append(df_box)

            if args.save_raw:
                outp = os.path.join(args.raw_dir, f"docs_{box_id}_{args.start_date}_{args.end_date}.csv")
                df_box.to_csv(outp, index=False)

        print(f"[docs] {box_id}: {sum(len(x) for x in box_docs_parts)} rows", flush=True)

    if args.dry_run:
        print("[dry-run] complete.", flush=True)
        return

    if not all_docs:
        raise SystemExit("No documents retrieved. Check table name, terms, and date range.")

    df_docs = pd.concat(all_docs, ignore_index=True).drop_duplicates(subset=["box_id", "url"])
    print(f"[docs] total unique docs: {len(df_docs)}", flush=True)

    agg = make_aggregates(df_docs, bucket_hours=int(args.bucket_hours))
    cands = select_seed_candidates(
        agg,
        min_total_articles=int(args.min_total_articles),
        min_unique_domains=int(args.min_unique_domains),
        top_k_per_box=int(args.top_k_per_box),
    )
    print(f"[cands] {len(cands)} candidate aggregates", flush=True)

    events = cluster_candidates(cands, title_sim_thresh=float(args.title_sim), merge_within_hours=int(args.merge_hours))
    print(f"[events] {len(events)} seed clusters (pre-enrich)", flush=True)

    enriched = enrich_events_from_docs(
        events,
        df_docs,
        bucket_hours=int(args.bucket_hours),
        onset_pad_hours=int(args.onset_pad_hours),
        peak_bin_hours=int(args.peak_bin_hours),
        top_n=5,
    )
    if not enriched:
        raise SystemExit("No events produced after enrichment. (Possible: thresholds too strict; not enough docs.)")

    df_out = pd.DataFrame(enriched)

    # Optional post-filters
    if float(args.min_impact_score) != float("-inf"):
        df_out = df_out[df_out["impact_score"] >= float(args.min_impact_score)].copy()
    if float(args.min_confidence_score) != float("-inf"):
        df_out = df_out[df_out["confidence_score"] >= float(args.min_confidence_score)].copy()
    if float(args.max_top_domain_share) > 0:
        df_out = df_out[df_out["top_domain_share"] < float(args.max_top_domain_share)].copy()

    if df_out.empty:
        raise SystemExit("All events filtered out. Loosen post-filters or candidate thresholds.")

    # Sort: highest value first for review
    df_out = df_out.sort_values(
        ["impact_score", "confidence_score", "unique_domains", "total_articles", "onset_utc"],
        ascending=[False, False, False, False, True],
    )

    df_out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(df_out)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
