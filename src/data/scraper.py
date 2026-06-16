from __future__ import annotations

import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Research Project - WC2026 Tactical Simulator)"
}
RATE_LIMIT_SECONDS = 3.0
DEFAULT_TIMEOUT_SECONDS = 30


def fetch_html(url: str, *, session: requests.Session | None = None) -> str:
    """Fetch a page politely enough for one-off research scraping."""
    time.sleep(RATE_LIMIT_SECONDS)
    client = session or requests.Session()
    response = client.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def flatten_columns(columns: Iterable[object]) -> list[str]:
    flattened: list[str] = []

    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip()]
            flattened.append("_".join(parts))
        else:
            flattened.append(str(column).strip())

    return flattened


def extract_table(html: str, table_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": table_id})

    if table is None:
        raise ValueError(f"Table not found: {table_id}")

    dataframes = pd.read_html(StringIO(str(table)))
    if not dataframes:
        raise ValueError(f"Table has no readable rows: {table_id}")

    dataframe = dataframes[0]
    dataframe.columns = flatten_columns(dataframe.columns)
    return dataframe


def scrape_player_stats(fbref_url: str, *, table_id: str = "stats_standard_9") -> pd.DataFrame:
    html = fetch_html(fbref_url)
    return extract_table(html, table_id)


def scrape_national_team(country: str, fbref_squad_id: str) -> pd.DataFrame:
    slug = country.replace(" ", "-")
    url = f"https://fbref.com/en/squads/{fbref_squad_id}/{slug}-Stats"
    return scrape_player_stats(url)


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path
