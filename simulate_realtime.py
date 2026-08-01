"""
Real-Time Complaint Simulator
==============================
Replays historical BBMP complaints row by row to the /complaints endpoint,
simulating live citizen complaint ingestion for demo purposes.

Usage:
    # Default: replay synthetic_complaints.csv at 1 complaint/sec
    python simulate_realtime.py

    # Use real complaints.csv, faster speed, limit to 200 complaints
    python simulate_realtime.py --csv data/complaints.csv --delay 0.3 --limit 200

    # Replay only a specific category
    python simulate_realtime.py --category "Garbage and Unsanitary Practices"

In production this script is replaced by BBMP's Sahaaya portal or a
mobile app calling POST /complaints directly.
"""

import argparse
import time
import requests
import pandas as pd
import sys
from datetime import datetime

API_URL = "http://localhost:8000/complaints"
HEALTH_URL = "http://localhost:8000/health"


def wait_for_api(timeout: int = 60):
    """Wait until the API is ready before starting replay."""
    print("Waiting for API to be ready...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            if r.status_code == 200 and r.json().get("pipeline_ready"):
                print(" Ready!\n")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print("\nAPI not ready after timeout. Is the server running?")
    return False


def replay(csv_path: str, delay: float, limit: int, category: str):
    """Load CSV and POST each row to /complaints with a delay between each."""

    print(f"Loading complaints from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)

    # Filter by category if specified
    if category:
        if "category_title" in df.columns:
            df = df[df["category_title"].str.contains(category, case=False, na=False)]
            print(f"Filtered to category '{category}': {len(df)} complaints")
        else:
            print("Warning: 'category_title' column not found, ignoring --category filter")

    # Sort by created_at to replay in chronological order
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], format="mixed",
                                           dayfirst=False, errors="coerce")
        df = df.dropna(subset=["created_at"]).sort_values("created_at")

    if limit:
        df = df.head(limit)

    total = len(df)
    print(f"Replaying {total} complaints at {delay}s interval")
    print(f"Estimated time: {total * delay / 60:.1f} minutes\n")
    print(f"{'#':<6} {'Time':<22} {'Category':<40} {'Status':<12} {'Zone'}")
    print("-" * 100)

    success = 0
    failed  = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        payload = {
            "created_at":             str(row.get("created_at", datetime.now())),
            "latitude":               float(row.get("latitude", 12.97)),
            "longitude":              float(row.get("longitude", 77.59)),
            "category_id":            int(row.get("category_id", 0))
                                      if pd.notna(row.get("category_id")) else 0,
            "category_title":         str(row.get("category_title", "Others")),
            "sub_category_title":     str(row.get("sub_category_title", "Others")),
            "complaint_status_title": str(row.get("complaint_status_title", "Open")),
            "ward_id":                int(row.get("ward_id", 0))
                                      if pd.notna(row.get("ward_id")) else 0,
            "ward_title":             str(row.get("ward_title", "Unknown")),
            "civic_agency_title":     str(row.get("civic_agency_title", "BBMP")),
            "comment_count":          int(row.get("comment_count", 0))
                                      if pd.notna(row.get("comment_count")) else 0,
            "description":            str(row.get("description", ""))[:200],
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            if resp.status_code == 200:
                data    = resp.json()
                zone_id = data.get("zone_id", "?")
                cat     = payload["category_title"][:38]
                status  = payload["complaint_status_title"][:10]
                ts      = payload["created_at"][:19]
                print(f"{i:<6} {ts:<22} {cat:<40} {status:<12} Zone {zone_id}")
                success += 1
            else:
                print(f"{i:<6} ERROR {resp.status_code}: {resp.text[:80]}")
                failed += 1
        except requests.exceptions.ConnectionError:
            print(f"{i:<6} CONNECTION ERROR — is the API running at {API_URL}?")
            failed += 1
        except Exception as e:
            print(f"{i:<6} UNEXPECTED ERROR: {e}")
            failed += 1

        time.sleep(delay)

    print("\n" + "=" * 100)
    print(f"Replay complete: {success} accepted, {failed} failed out of {total} complaints")
    print(f"Check http://localhost:8000/risk to see updated zone risk scores")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate real-time complaint ingestion for demo purposes"
    )
    parser.add_argument(
        "--csv",
        default="data/synthetic_complaints.csv",
        help="Path to complaints CSV (default: data/synthetic_complaints.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between each complaint POST (default: 1.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of complaints to replay (default: 100)",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Filter complaints by category title substring (default: all)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip API health check and start immediately",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  BBMP Real-Time Complaint Simulator")
    print("=" * 60)
    print(f"  API endpoint : {API_URL}")
    print(f"  CSV file     : {args.csv}")
    print(f"  Delay        : {args.delay}s per complaint")
    print(f"  Limit        : {args.limit} complaints")
    print(f"  Category     : {args.category or 'All'}")
    print("=" * 60 + "\n")

    if not args.no_wait:
        if not wait_for_api():
            sys.exit(1)

    replay(args.csv, args.delay, args.limit, args.category)


if __name__ == "__main__":
    main()
