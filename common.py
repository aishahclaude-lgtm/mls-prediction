"""
Shared helpers for backfill.py and ingest.py.

All confirmed-working data source endpoints live here in one place so they're
easy to fix in one spot if a provider changes something.
"""
import os
import time
import requests
from supabase import create_client, Client

ASA_BASE = "https://app.americansocceranalysis.com/api/v1/mls"
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"

REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5  # be polite to free public APIs


def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_KEY environment variables "
            "(use the service_role key, not the anon key, since these scripts write data)."
        )
    return create_client(url, key)


def fetch_json(url: str, params: dict | None = None):
    """GET a URL and return parsed JSON, with a couple of retries."""
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                                 headers={"Accept": "application/json"})
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except Exception as e:  # noqa: BLE001 - we want to retry on anything and report at the end
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_err}")


def asa_get(path: str, params: dict | None = None):
    return fetch_json(f"{ASA_BASE}/{path}", params=params)


def espn_get(path: str, params: dict | None = None):
    return fetch_json(f"{ESPN_BASE}/{path}", params=params)


def id_map(supabase: Client, table: str, external_col: str) -> dict:
    """Return {external_id: internal_uuid} for a table, for joining ASA/ESPN ids to our rows.

    Paginated in batches of 1000 — PostgREST's default row cap would otherwise
    silently truncate large tables (e.g. players) and make already-inserted
    rows look "new" on the next run.
    """
    rows = []
    start = 0
    page = 1000
    while True:
        batch = supabase.table(table).select(f"id,{external_col}").range(start, start + page - 1).execute().data
        rows.extend(batch)
        if len(batch) < page:
            break
        start += page
    return {r[external_col]: r["id"] for r in rows if r.get(external_col)}


def fetch_all(supabase: Client, table: str, columns: str = "*", order_col: str | None = None) -> list:
    """Paginated select("*")-style fetch for any table larger than ~1000 rows."""
    rows = []
    start = 0
    page = 1000
    while True:
        q = supabase.table(table).select(columns)
        if order_col:
            q = q.order(order_col)
        batch = q.range(start, start + page - 1).execute().data
        rows.extend(batch)
        if len(batch) < page:
            break
        start += page
    return rows


def chunked(seq, size=200):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
