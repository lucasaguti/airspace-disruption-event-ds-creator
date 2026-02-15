#!/usr/bin/env python3
"""
Build NEWS-ONLY (reported) event-first seed dataset using GDELT BigQuery (GKG).

Design goals:
  - Fast, scalable historical capture using BigQuery (no API paging bottlenecks)
  - Broad capture (location + intent/title/themes), then reuse your existing pipeline:
      articles_to_frame → make_aggregates → select_seed_candidates → cluster_candidates → attach_top_sources
  - Resume-safe: every (box × date-chunk × loc-shard × intent-shard) query is cached to disk.
    Reruns do NOT re-query BigQuery when --resume is enabled.

Requirements:
  pip install google-cloud-bigquery pyarrow pandas

Auth:
  gcloud auth application-default login
  or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON

Inputs:
  --watchbox-terms watchbox_terms.json  (box_id -> locality_terms[])
  --counts-path backtest/derived/airspace_counts.csv.gz (ts_utc, box_id, count)

Outputs:
  - Raw cache per shard: <raw-dir>/.../bq_<hash>.csv.gz
  - Optional per-box docs CSV: <raw-dir>/docs_<box>_<start>_<end>.csv
  - Final CSV: --out-csv (NEWS-ONLY seed clusters; counts fields left blank)

Notes:
  - GKG doesn't reliably provide language per row; we leave language blank.
  - Page titles are extracted from Extras via <PAGE_TITLE>…</PAGE_TITLE>.
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
# Intents (same idea as your existing scripts)
# ---------------------------

DEFAULT_INTENT_BUNDLES: Dict[str, List[str]] = {
    "AIRSPACE": [
        "airspace closed", "airspace closure", "airspace restricted", "flight ban",
        "no-fly zone", "airspace reopened", "airspace reopens",
    ],
    "OPS": [
        "flight cancellations", "rerouted", "rerouted flights", "diverted", "diverted flights",
        "ATC outage", "air traffic control outage", "radar failure", "NATS", "EUROCONTROL regulation",
    ],
    "CONFLICT": [
        "missile attack", "rocket attack", "airstrike", "drone attack", "intercepted drone",
        "military operation", "shootdown", "surface-to-air missile", "SAM",
    ],
    "GNSS": [
        "GPS jamming", "GNSS jamming", "GPS spoofing", "GNSS spoofing",
        "navigation interference", "signal interference",
    ],
}

# Broad theme regex to reduce noise (still high recall)
DEFAULT_THEME_REGEX = r"(?i)(TRANSPORT|AVIATION|AIRPORT|AIRLINE|FLIGHT|AIR_TRAFFIC|RADAR|EUROCONTROL|NATS|GPS|GNSS|JAMM|SPOOF|MISSILE|ROCKET|DRONE)"

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
    return [xs[i:i + n] for i in range(0, len(xs), n)]


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


def _re_union(terms: List[str]) -> str:
    clean = [t.strip().lower() for t in terms if (t or "").strip()]
    if not clean:
        return r"(?i)$^"  # match nothing

    parts = []
    for t in clean:
        esc = re.escape(t)
        # require "term" to appear as its own token-ish unit in the locations string
        parts.append(rf"(?:^|[^a-z0-9]){esc}(?:[^a-z0-9]|$)")

    return r"(?i)(" + "|".join(parts) + r")"


def cache_path(raw_dir: str, cache_key: str) -> str:
    safe_mkdir(raw_dir)
    return os.path.join(raw_dir, f"bq_{cache_key}.csv.gz")


# ---------------------------
# BigQuery fetch
# ---------------------------

def _bq_client(project: str):
    if bigquery is None:
        raise SystemExit("Missing google-cloud-bigquery. Install: pip install google-cloud-bigquery pyarrow")
    return bigquery.Client(project=project)


def _sql_with_partitiontime(table: str) -> str:
    # Uses DATE(_PARTITIONTIME) pruning for partitioned tables.
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
        (@require_title_intent AND REGEXP_CONTAINS(LOWER(COALESCE(page_title, '')), @intent_re))
        OR
        (NOT @require_title_intent AND (
          REGEXP_CONTAINS(LOWER(COALESCE(page_title, '')), @intent_re)
          OR REGEXP_CONTAINS(COALESCE(themes, ''), @theme_re)
        ))
      )
    """


def _sql_without_partitiontime(table: str) -> str:
    # Fallback if _PARTITIONTIME isn't available; uses DATE field.
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
        (@require_title_intent AND REGEXP_CONTAINS(LOWER(COALESCE(page_title, '')), @intent_re))
        OR
        (NOT @require_title_intent AND (
          REGEXP_CONTAINS(LOWER(COALESCE(page_title, '')), @intent_re)
          OR REGEXP_CONTAINS(COALESCE(themes, ''), @theme_re)
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
    Each article dict: url,title,domain,language,published_utc
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

        out.append({
            "url": url,
            "title": title,
            "domain": domain,
            "language": "",
            "published_utc": pub_iso,
        })

    return out, None


# ---------------------------
# Downstream pipeline (same shape as your existing scripts)
# ---------------------------

def articles_to_frame(arts: List[Dict[str, Any]], box_id: str, intent_bundle: str) -> pd.DataFrame:
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
            "intent_bundle": intent_bundle,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["url"].astype(bool)]


def make_aggregates(df_docs: pd.DataFrame, bucket_hours: int = 24) -> pd.DataFrame:
    df = df_docs.copy()
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["ts_utc"] = ts[ts.notna()]
    df["bucket_utc"] = df["ts_utc"].dt.floor(f"{bucket_hours}h")
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
    cands = cands.sort_values(["box_id", "bucket_utc"]).reset_index(drop=True)
    clusters: List[Dict[str, Any]] = []
    for _, row in cands.iterrows():
        placed = False
        for c in clusters:
            if c["box_id"] != row["box_id"]:
                continue
            dt_hours = abs((row["bucket_utc"] - c["event_end_utc"]).total_seconds()) / 3600.0
            if dt_hours > merge_within_hours:
                continue
            if jaccard(row["rep_title"], c["rep_title"]) >= title_sim_thresh:
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
        top_domains = dfe["domain"].value_counts().head(top_n).index.tolist()
        top_urls = dfe.drop_duplicates(subset=["url"])["url"].head(top_n).tolist()
        e["top_domains"] = json.dumps(top_domains)
        e["top_urls"] = json.dumps(top_urls)
    return events


def events_to_news_seed_frame(events: List[Dict[str, Any]], bucket_hours: int) -> pd.DataFrame:
    """
    NEWS-ONLY output. Keeps columns compatible with the "truth seed" schema by leaving
    counts-derived fields blank/NA. This lets you add counts in phase 2 without
    breaking downstream consumers expecting these columns to exist.
    """
    rows: List[Dict[str, Any]] = []
    pad = pd.Timedelta(hours=int(bucket_hours))

    for e in events:
        start = pd.to_datetime(e["event_start_utc"], utc=True)
        # In clustering, event_end_utc is the last bucket timestamp; make end exclusive by adding bucket_hours.
        end_excl = pd.to_datetime(e["event_end_utc"], utc=True) + pad

        rows.append(
            {
                "event_id": e.get("event_id", ""),
                "box_id": e.get("box_id", ""),
                "event_start_utc": start.isoformat(),
                "event_end_utc": end_excl.isoformat(),
                "rep_title": e.get("rep_title", ""),
                "total_articles": int(e.get("total_articles", 0) or 0),
                "unique_domains": int(e.get("unique_domains", 0) or 0),
                "unique_languages": int(e.get("unique_languages", 0) or 0),

                # Counts-derived fields intentionally blank (Phase 2)
                "baseline": float("nan"),
                "extremum": float("nan"),
                "delta_pct_peak": float("nan"),
                "direction": "",
                "duration_min": int((end_excl - start).total_seconds() // 60),
                "impact_pass": "",

                "top_domains": e.get("top_domains", "[]"),
                "top_urls": e.get("top_urls", "[]"),
                "fin_label": "",
                "notes": "",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build NEWS-ONLY seed dataset from GDELT BigQuery (GKG).")
    ap.add_argument("--gcp-project", required=True, help="GCP project ID used for BigQuery billing/auth.")
    ap.add_argument("--gdelt-table", default=DEFAULT_GDELT_TABLE, help=f"BigQuery table (default: {DEFAULT_GDELT_TABLE})")

    ap.add_argument("--watchbox-terms", required=True, help="JSON file mapping box_id -> locality_terms[]")
    ap.add_argument("--out-csv", default="news_seed_v1.csv", help="Output CSV path.")
    ap.add_argument("--start-date", required=True, help="UTC start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end-date", required=True, help="UTC end date YYYY-MM-DD (exclusive).")

    ap.add_argument("--intent-mode", choices=["all", "core", "risk"], default="all")
    ap.add_argument("--intent-split", action="store_true")
    ap.add_argument("--intent-bundles", default="AIRSPACE,OPS,CONFLICT,GNSS")
    ap.add_argument("--intent-shard-size", type=int, default=999)
    ap.add_argument("--loc-shard-size", type=int, default=999)

    ap.add_argument("--chunk-days", type=int, default=30, help="Date chunk size (default 30).")
    ap.add_argument("--theme-regex", default=DEFAULT_THEME_REGEX, help="Regex for aviation/ops-ish themes (broad filter).")

    ap.add_argument(
        "--require-title-intent",
        action="store_true",
        help="Only keep docs whose PAGE_TITLE matches intent terms (disables theme-only matches).",
    )

    ap.add_argument("--resume", action="store_true", help="Reuse cached shard results (prevents re-querying BigQuery).")
    ap.add_argument("--save-raw", action="store_true", help="Save per-shard caches and per-box docs CSV.")
    ap.add_argument("--raw-dir", default="raw_bq", help="Directory to store shard caches and raw docs.")

    ap.add_argument("--min-total-articles", type=int, default=3)
    ap.add_argument("--min-unique-domains", type=int, default=1)
    ap.add_argument("--top-k-per-box", type=int, default=500)
    ap.add_argument("--title-sim", type=float, default=0.85)
    ap.add_argument("--merge-hours", type=int, default=48)
    ap.add_argument("--bucket-hours", type=int, default=24)
    ap.add_argument("--min-baseline", type=float, default=20.0)
    ap.add_argument("--shock-thresh", type=float, default=0.25)
    ap.add_argument("--window-pad-hours", type=int, default=6)

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

    all_docs: List[pd.DataFrame] = []
    safe_mkdir(args.raw_dir)

    for box_id, locality_terms in watchbox_terms.items():
        locality_terms = [t for t in locality_terms if t]
        if not locality_terms:
            continue

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

                        df_part = articles_to_frame(arts, box_id=box_id, intent_bundle=bundle)
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
    events = attach_top_sources(events, df_docs, top_n=5)
    print(f"[events] {len(events)} seed clusters", flush=True)

    df_out = events_to_news_seed_frame(events, bucket_hours=int(args.bucket_hours))
  
    if df_out.empty:
        raise SystemExit("No events produced. (Possible: thresholds too strict; not enough docs.)")

    df_out = df_out.sort_values(
        ["unique_domains", "total_articles", "event_start_utc"],
        ascending=[False, False, True],
    )
    df_out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {len(df_out)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
