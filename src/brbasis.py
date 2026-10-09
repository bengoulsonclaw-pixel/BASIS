"""brbasis.py — the Brazil-vs-US ag basis book (Curve / RV → 🇧🇷 Brazil basis).

Each line is a Brazilian price set against its US benchmark in the US contract's own unit,
scored the way the Curve/RV Monitor scores a spread: a rolling z-score (stretch vs the
recent regime), a full-history percentile, an OU half-life.

    coffee   B3 ICF arabica      - ICE KC         c/lb   same contract month
    corn     B3 CCM (Campinas)   - CBOT corn      c/bu   B3 most-active vs CBOT front
    soy      CEPEA Paranaguá     - CBOT soybeans  c/bu   physical vs CBOT front
    cattle   B3 BGI / CME LC     ratio                   carcass vs live weight
    sugar    ICE SB - ethanol parity c/lb   CEPEA SP hydrous (mill-net) as sugar-equivalent

Legs come from b3ag.py (B3 Price Report archive) and cepea.py (CEPEA spreadsheets); US
legs are the RAW front settles from the deep store — never panama-adjusted levels.

ROLL BREAKS. Either leg switching contract makes the basis jump for reasons that are not
basis moving (Dec→Mar carry, a different B3 month). Those sessions are flagged `brk`: the
day change is blanked and the half-life regression skips them, as curvemon does. The
level and z-score keep them — the new pair IS today's basis.

SEASONALITY. Corn and soy basis carry a harvest cycle (Brazil's safrinha corn lands
Jun–Aug, soy Feb–Apr). A full-history percentile ignores that, so those lines also carry
`spctl` — today against the same ±2 weeks of the calendar in prior years — and the Hot
Sheet flags them on THAT, never on the raw percentile.

The pull writes the book to disk (precompute rule: the page opens in milliseconds); the
page falls back to building it live if the store is missing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .curvemon import _half_life

_ROOT = Path(__file__).resolve().parents[1]
HIST = _ROOT / "data" / "signals" / "brazil_basis.parquet"
META = _ROOT / "data" / "signals" / "brazil_basis_meta.json"

WINDOW = 252
Z_THRESHOLD = 2.0

BOOK = [
    dict(key="coffee", name="Coffee — Brazil arabica vs ICE C", src="b3", root="ICF",
         col="basis", unit="c/lb", dp=1, seasonal=False,
         b3_label="B3 ICF", us_label="ICE KC",
         desc="B3 arabica (Brazilian naturals, type 4/5, São Paulo/Minas warehouses) minus "
              "ICE \"C\" (washed arabica) in the SAME contract month, both in US cents/lb. "
              "Normally negative: the screen proxy for the Brazil physical differential."),
    dict(key="corn", name="Corn — Campinas vs CBOT", src="b3", root="CCM",
         col="basis", unit="c/bu", dp=1, seasonal=True,
         b3_label="B3 CCM", us_label="CBOT C",
         desc="B3 corn (Campinas, BRL/60 kg sack, at PTAX) minus CBOT corn front, in US "
              "cents/bushel. B3's most-active month vs the CBOT front — the two exchanges list "
              "different months. Interior Brazil price: domestic feed and corn-ethanol demand "
              "can hold it ABOVE Chicago."),
    dict(key="soy", name="Soy — Paranaguá vs CBOT", src="cepea", root="soy_paranagua",
         col="basis", unit="c/bu", dp=1, seasonal=True,
         b3_label="CEPEA Paranaguá", us_label="CBOT S",
         desc="CEPEA/ESALQ Paranaguá port soy (US$/60 kg sack, CEPEA's own FX) minus CBOT "
              "soybean front, in US cents/bushel — the Brazil export premium/discount. B3's "
              "soy future is cash-settled on CME, so the physical indicator is the Brazil leg."),
    dict(key="cattle", name="Cattle — Brazil boi gordo / CME", src="b3", root="BGI",
         col="ratio", unit="×", dp=3, seasonal=False,
         b3_label="B3 BGI", us_label="CME LC",
         desc="B3 live cattle (CEPEA-indexed, BRL per 15 kg arroba of CARCASS, at PTAX, as "
              "USD/cwt) divided by CME live cattle front (USD/cwt LIVE weight). A ratio, never "
              "a difference: carcass vs live weight (~52–55% dressing) sits between them."),
]
# ── sugar–ethanol parity ─────────────────────────────────────────────────────
# CONSECANA ATR (total recoverable sugar) conversion factors: kg of ATR consumed per unit
# of product. A mill choosing between the two outputs compares revenue per kg of ATR.
ATR_PER_KG_VHP = 1.0453         # 1 kg VHP sugar (CONSECANA: 50 kg sack = 52.265 kg ATR)
ATR_PER_L_HYDROUS = 1.6913      # 1 L hydrous ethanol (CONSECANA manual, 2021 revision;
                                # the 2014/15 SP circular used 1.6761 — 0.9% apart)
LB_PER_TONNE = 2204.62262


def ethanol_parity_clb(brl_per_litre, ptax):
    """Mill-net hydrous ethanol (R$/L) -> the raw-sugar price that pays the SAME per kg of
    cane ATR, in US cents/lb. Ex-mill: it carries no mill-to-Santos logistics, which ICE
    No.11 (FOB Santos) does — a roughly constant gap the z-score/percentile absorb."""
    brl_per_kg_sugar = brl_per_litre / ATR_PER_L_HYDROUS * ATR_PER_KG_VHP
    return brl_per_kg_sugar * 1000 / ptax / LB_PER_TONNE * 100


BOOK.append(dict(
    key="sugar", name="Sugar − ethanol parity", src="parity", root="ethanol_hydrous_sp",
    col="basis", unit="c/lb", dp=2, seasonal=True,
    b3_label="Ethanol parity", us_label="ICE SB",
    sig_hi="Sugar rich vs ethanol", sig_lo="Sugar cheap vs ethanol",
    desc="ICE raw sugar No.11 front minus hydrous-ethanol parity: the sugar price that would "
         "pay a Brazilian mill the same per kg of cane sugar (ATR) as selling hydrous "
         "ethanol at CEPEA's São Paulo mill price (net of ICMS and PIS/Cofins), at PTAX. "
         "Positive = sugar pays more than ethanol, so mills tilt the cane mix to sugar; "
         "negative = ethanol pays more. Parity is ex-mill (no freight to Santos), so the "
         "level runs above the true switching point by a roughly constant logistics cost — "
         "read the z-score and percentile."))

BOOK_BY_KEY = {b["key"]: b for b in BOOK}


def _legs(spec: dict) -> pd.DataFrame:
    """date, value, b3/us legs and contracts for one line of the book."""
    if spec["src"] == "b3":
        from . import b3ag
        df = b3ag.basis(spec["root"])
        if df.empty:
            return df
        df = df.rename(columns={"b3_px": "br_px", "b3_contract": "br_contract"})
    elif spec["src"] == "parity":
        df = _parity_frame()
        if df.empty:
            return df
    else:
        from . import cepea
        df = cepea.basis(spec["root"])
        if df.empty:
            return df
        df["br_contract"] = "spot"
    df["value"] = df[spec["col"]]
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    sw = (df["br_contract"] != df["br_contract"].shift()) | \
         (df["us_contract"] != df["us_contract"].shift())
    sw.iloc[0] = False
    df["brk"] = sw.to_numpy()
    return df[["date", "value", "br_px", "us_px", "br_contract", "us_contract", "brk"]]


def _parity_frame() -> pd.DataFrame:
    """Daily on ICE sugar's dates: the latest weekly CEPEA ethanol print (as-of, never
    ahead) at the day's PTAX, converted to sugar parity."""
    from . import b3ag, cepea, deepstore
    eth = cepea.load("ethanol_hydrous_sp")[["date", "brl"]].sort_values("date")
    px = deepstore.get_raw(["SBA Comdty"])
    if eth.empty or px.empty:
        return pd.DataFrame()
    ct = deepstore.get_contracts(["SBA Comdty"])
    us = pd.DataFrame({"us_px": px["SBA Comdty"]})
    us["us_contract"] = ct["SBA Comdty"] if not ct.empty else None
    us.index = pd.to_datetime(us.index)
    us = us.dropna(subset=["us_px"]).rename_axis("date").reset_index().sort_values("date")
    fx = b3ag.ptax(start="2016-01-01").sort_values("date")      # SB deep store starts 2016
    # merge_asof refuses mixed datetime resolutions, and these three stores come back as
    # ms / us / ns depending on how each was written — pin them all to ns.
    for f in (us, eth, fx):
        f["date"] = f["date"].astype("datetime64[ns]")
    df = pd.merge_asof(us, eth, on="date", direction="backward",
                       tolerance=pd.Timedelta(days=10))
    df = pd.merge_asof(df, fx, on="date", direction="backward",
                       tolerance=pd.Timedelta(days=5))
    df = df.dropna(subset=["brl", "ptax"])
    df["br_px"] = ethanol_parity_clb(df["brl"], df["ptax"])
    df["basis"] = df["us_px"] - df["br_px"]
    df["br_contract"] = "CEPEA wk"
    return df


def history() -> pd.DataFrame:
    """Long frame: key + the _legs columns, every line of the book."""
    parts = []
    for spec in BOOK:
        try:
            d = _legs(spec)
        except Exception as e:                 # one dead source never blanks the book
            print(f"  (Brazil basis {spec['key']} skipped: {e})")
            continue
        if not d.empty:
            parts.append(d.assign(key=spec["key"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


SEAS_HALF_WINDOW = 14     # calendar days either side of today's day-of-year
SEAS_MIN_YEARS = 3        # prior seasons a seasonal percentile needs


def seasonal_pctl(s: pd.Series, asof: pd.Timestamp | None = None) -> tuple[float, int]:
    """Today's level ranked against the SAME time of year in PRIOR years only: every
    session within ±SEAS_HALF_WINDOW calendar days of today's day-of-year, before this
    season began. Returns (percentile, prior seasons used); NaN below SEAS_MIN_YEARS.

    Why: corn and soy basis carry a harvest cycle, so a full-history percentile calls
    every harvest-time low "cheap". This asks "low for October?" instead."""
    s = s.dropna()
    if s.empty:
        return float("nan"), 0
    asof = asof or s.index[-1]
    level = float(s.loc[:asof].iloc[-1])
    hist = s[s.index < asof - pd.Timedelta(days=SEAS_HALF_WINDOW + 1)]
    doy = asof.dayofyear
    dd = (hist.index.dayofyear - doy) % 365
    dd = np.minimum(dd, 365 - dd)
    same = hist[dd <= SEAS_HALF_WINDOW]
    yrs = same.index.year.nunique() if len(same) else 0
    if yrs < SEAS_MIN_YEARS:
        return float("nan"), int(yrs)
    return float((same <= level).mean() * 100), int(yrs)


def _row(spec: dict, d: pd.DataFrame, window: int, threshold: float) -> dict:
    s = d.set_index("date")["value"]
    brk = d.set_index("date")["brk"]
    roll = s.rolling(window, min_periods=max(60, window // 3))
    mean, sd = roll.mean().iloc[-1], roll.std().iloc[-1]
    level = float(s.iloc[-1])
    z = (level - mean) / sd if sd and not np.isnan(sd) else float("nan")

    def _chg(n):
        if len(s) <= n or brk.iloc[-n:].any():
            return float("nan")                # a roll inside the window is not a move
        return float(level - s.iloc[-1 - n])

    sig = "—"
    if not np.isnan(z) and abs(z) >= threshold:
        # Brazil leg rich / cheap vs the US benchmark
        sig = (spec.get("sig_hi", "Brazil rich") if z > 0
               else spec.get("sig_lo", "Brazil cheap"))
    last = d.iloc[-1]
    return {
        "key": spec["key"], "name": spec["name"], "unit": spec["unit"], "dp": spec["dp"],
        "desc": spec["desc"], "seasonal": spec["seasonal"],
        "level": level, "chg1d": _chg(1), "chg5d": _chg(5), "chg21d": _chg(21),
        "mean": float(mean), "sigma": float(sd), "z": float(z),
        "pctl": float((s <= level).mean() * 100),
        "spctl": seasonal_pctl(s)[0] if spec["seasonal"] else float("nan"),
        "seas_years": seasonal_pctl(s)[1] if spec["seasonal"] else 0,
        "half_life": _half_life(s, brk), "signal": sig,
        "br_px": float(last["br_px"]), "us_px": float(last["us_px"]),
        "br_contract": last["br_contract"], "us_contract": last["us_contract"],
        "b3_label": spec["b3_label"], "us_label": spec["us_label"],
        "asof": str(pd.Timestamp(last["date"]).date()),
        "first": str(pd.Timestamp(d["date"].iloc[0]).date()), "days": len(d),
    }


def monitor(hist: pd.DataFrame | None = None, window: int = WINDOW,
            threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    hist = load() if hist is None else hist
    if hist is None or hist.empty:
        return pd.DataFrame()
    rows = []
    for spec in BOOK:
        d = hist[hist["key"] == spec["key"]].sort_values("date")
        if len(d) >= 60:
            rows.append(_row(spec, d, window, threshold))
    return pd.DataFrame(rows)


def chart_data(key: str, hist: pd.DataFrame | None = None, window: int = WINDOW,
               years: float | None = None) -> pd.DataFrame:
    hist = load() if hist is None else hist
    d = hist[hist["key"] == key].sort_values("date").reset_index(drop=True)
    if d.empty:
        return d
    s = d["value"]
    roll = s.rolling(window, min_periods=max(60, window // 3))
    d["mean"], sd = roll.mean(), roll.std()
    d["upper"], d["lower"] = d["mean"] + 2 * sd, d["mean"] - 2 * sd
    d["z"] = (s - d["mean"]) / sd
    if years:
        d = d[d["date"] >= d["date"].max() - pd.Timedelta(days=int(365.25 * years))]
    return d


def build() -> dict:
    """Pull-time: rebuild the book's history store. Keeps the old store if nothing came back."""
    h = history()
    if h.empty:
        return {"status": "empty — kept previous store"}
    HIST.parent.mkdir(parents=True, exist_ok=True)
    tmp = HIST.with_suffix(".tmp")
    h.to_parquet(tmp, index=False)
    tmp.replace(HIST)
    info = {k: int(n) for k, n in h.groupby("key").size().items()}
    META.write_text(json.dumps({"built": datetime.now().isoformat(timespec="seconds"),
                                "rows": info}, indent=1))
    return info


def load() -> pd.DataFrame:
    if HIST.exists():
        return pd.read_parquet(HIST)
    return history()


RADAR_MAX = 2          # the book is four lines; at most two make the sheet
RADAR_PCTL_HI = 95.0
RADAR_PCTL_LO = 5.0
RADAR_SPARK = 250


def radar_items() -> list:
    """Hot Sheet reads, two kinds: a z-flag (|z| ≥ Z_THRESHOLD on the 1y band), or an
    extreme of the RIGHT percentile — the seasonal one (same weeks, prior years) for the
    harvest-seasonal lines, the full-history one otherwise. A corn basis at its usual
    harvest low is never flagged as an extreme. Reads the precomputed store only."""
    from src import hotsheet
    from .reportkit import ordinal

    hist = load()
    m = monitor(hist)
    if m.empty:
        return []
    m["ref_pctl"] = [r.spctl if r.seasonal else r.pctl for r in m.itertuples()]
    m["ref_what"] = ["for the time of year" if r.seasonal else "of its history"
                     for r in m.itertuples()]
    m["zflag"] = m["signal"] != "—"
    m["pflag"] = m["ref_pctl"].notna() & ((m["ref_pctl"] >= RADAR_PCTL_HI)
                                          | (m["ref_pctl"] <= RADAR_PCTL_LO))
    m = m[m["zflag"] | m["pflag"]].copy()
    if m.empty:
        return []
    m["heat"] = [max(hotsheet.heat_from_z(r.z) if r.z == r.z else 0.0,
                     hotsheet.heat_from_pctl(r.ref_pctl) if r.ref_pctl == r.ref_pctl else 0.0)
                 for r in m.itertuples()]
    out = []
    for r in m.sort_values("heat", ascending=False).head(RADAR_MAX).itertuples():
        lvl = (f"{r.level:.{r.dp}f}×" if r.unit == "×"
               else f"{r.level:+,.{r.dp}f} {r.unit}")
        pct = ("" if r.ref_pctl != r.ref_pctl
               else f", the {ordinal(int(round(r.ref_pctl)))} percentile {r.ref_what}")
        if r.zflag:
            text = (f"**{r.name}** screens **{r.signal}** at {lvl} "
                    f"({r.z:+.1f}σ on its 1y band{pct}).")
            metric, sub = f"z {r.z:+.1f}", "vs its 1y band"
        else:
            side = "high" if r.ref_pctl >= RADAR_PCTL_HI else "low"
            text = f"**{r.name}** sits at {lvl}{pct} — a **{side}**."
            metric, sub = f"{ordinal(int(round(r.ref_pctl)))} pctl", r.ref_what
        spark = hist[hist["key"] == r.key].sort_values("date")["value"].tail(RADAR_SPARK)
        out.append(hotsheet.item(
            tag="CURVE", key=f"brbasis:{r.key}:{'z' if r.zflag else 'pctl'}",
            section="Curve / RV", text=text, heat=r.heat, metric=metric, sub=sub,
            value=float(r.level), spark=spark.tolist(), ticker="",
            page="Curve Monitor", book="ficc"))
    return out
