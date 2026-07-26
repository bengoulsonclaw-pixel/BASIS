"""Shared free-text product finder for the dashboard's product-list pages.

Type a name, ticker, sector or region — "oil", "CLA", "metals", "EMEA" — to filter a
timeline / table / heatmap down to matching products. Multi-word queries are AND-matched
("nat gas" → only the two gas contracts). A small ALIASES map lets common words surface
products that don't literally contain them (e.g. "oil" → crude/gasoline/diesel).

Usage on a page:
    tickers, q = prodsearch.search_box(tickers, INSTRUMENTS, key="xxx_search", container=col)
    if q and not tickers:
        st.info(prodsearch.NO_MATCH.format(q=q)); return
"""
from __future__ import annotations

import streamlit as st

# Common search words that aren't in a product's NAME (ticker -> extra keywords).
ALIASES = {
    "CLA Comdty": "oil crude wti", "COA Comdty": "oil crude brent",
    "QSA Comdty": "oil diesel gasoil", "HOA Comdty": "oil diesel distillate",
    "XBA Comdty": "oil gasoline petrol rbob", "NGA Comdty": "natural gas henry hub",
    "FJSA Comdty": "natural gas ttf", "MOA Comdty": "carbon emissions eua",
    "CUAA Comdty": "ethanol biofuel",
}

PLACEHOLDER = "name, ticker or sector — e.g. oil, CLA, metals"
NO_MATCH = ("No products match “{q}”. Try a name (Brent), a ticker (CLA), "
            "a sector (Energy) or a region (EMEA).")
NO_MATCH_STOCK = ("No stocks match “{q}”. Try a company (Apple), a ticker (AAPL) "
                  "or a sector (Financials).")


def matches(ticker: str, name: str, asset: str, region: str, terms) -> bool:
    """True if every search term is a substring of the product's searchable blob."""
    blob = f"{ticker} {name} {asset} {region} {ALIASES.get(ticker, '')}".lower()
    return all(t in blob for t in terms)


def filter_tickers(tickers, instruments, query: str):
    """Filter a ticker list by a raw query string (no UI). `instruments` maps
    ticker -> (name, price, asset, region). Blank query returns the list unchanged."""
    q = (query or "").strip()
    if not q:
        return list(tickers)
    terms = q.lower().split()
    return [t for t in tickers if t in instruments
            and matches(t, instruments[t][0], instruments[t][2], instruments[t][3], terms)]


def search_box(tickers, instruments, key: str, container=None, label: str = "Find a product"):
    """Render a search input and return (filtered_tickers, raw_query). Place `container`
    (an st.columns cell) to position it; defaults to the main flow."""
    c = container if container is not None else st
    q = (c.text_input(label, key=key, placeholder=PLACEHOLDER) or "").strip()
    return filter_tickers(tickers, instruments, q), q


def filter_frame(df, instruments, query: str, ticker_col: str | None = None,
                 name_col: str | None = None):
    """Filter a DataFrame of product rows by a raw query. If `ticker_col` is present it is
    matched directly (best — supports ticker/alias/region); otherwise rows are matched by
    their `name_col` (product name) against the set of matching product names, and by a raw
    substring so partial names still work. Blank query / empty df returns df unchanged."""
    q = (query or "").strip()
    if not q or df is None or getattr(df, "empty", True):
        return df
    terms = q.lower().split()
    if ticker_col and ticker_col in df.columns:
        keep = df[ticker_col].map(
            lambda t: t in instruments
            and matches(t, instruments[t][0], instruments[t][2], instruments[t][3], terms))
        return df[keep]
    names = {i[0].lower() for t, i in instruments.items()
             if matches(t, i[0], i[2], i[3], terms)}
    col = name_col if (name_col and name_col in df.columns) else df.columns[0]
    keep = df[col].astype(str).str.lower().map(
        lambda s: s in names or all(term in s for term in terms))
    return df[keep]


def filter_rows(df, cols, query: str):
    """Domain-agnostic finder: keep rows where EVERY query term appears somewhere across the
    given text `cols` (ticker / company name / sector …). No futures aliases — use this for
    equities and other universes. Blank query / empty df returns df unchanged."""
    q = (query or "").strip()
    if not q or df is None or getattr(df, "empty", True):
        return df
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    terms = q.lower().split()
    blob = df[cols[0]].astype(str)
    for c in cols[1:]:
        blob = blob.str.cat(df[c].astype(str), sep=" ")
    blob = blob.str.lower()
    keep = blob.map(lambda s: all(t in s for t in terms))
    return df[keep]


def search_row_box(df, cols, key: str, container=None, label: str = "Find a stock"):
    """Render a search input and return (filtered_df, raw_query) using filter_rows."""
    c = container if container is not None else st
    q = (c.text_input(label, key=key,
                      placeholder="name, ticker or sector — e.g. Apple, AAPL, Financials") or "").strip()
    return filter_rows(df, cols, q), q
