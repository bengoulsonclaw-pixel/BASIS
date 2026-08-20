"""Equities technical-analysis pipeline.

Replicates the FICC Technical Analysis engine on the equity universe, off free yfinance data:
a 2-year OHLCV backfill (close feeds every strategy; volume feeds OBV / MFI and the flag /
breakout volume checks), then the SAME technical strategies + the SAME cross-strategy scoring
(tascore) as the futures side. The Equities TA page reads the cached opportunities file here.

Everything in tascore.TA_STRATEGIES runs EXCEPT the pair-based Mean Reversion (no single-name
price line to run it on — it's excluded on the FICC chart for the same reason). The strategies
were made universe-agnostic: passed a `history` frame they iterate its columns, so feeding them
the equity close/volume runs them over equities with no other change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import yfin, equities
from src.strategies import (trend, ma_crossover, ma_crossover_swing,
                            support_resistance, fibonacci, flag_breakout, breakout_retest, momentum,
                            bollinger, elliott_wave, ichimoku, obv, mfi, donchian, aroon)
# flag_breakout now runs on equities too: compute_table()/find_opportunities() take an injected
# history+volume and skip the FICC visual-report cache (persist=False), so they don't clobber futures.

_DIR = Path(__file__).resolve().parents[1] / "data" / "signals"
CLOSE_FILE = _DIR / "eqta_close.parquet"
VOL_FILE = _DIR / "eqta_volume.parquet"
SIGNALS_FILE = _DIR / "eqta_opportunities.parquet"
META_FILE = _DIR / "eqta_meta.json"

# Close-only strategies, then the volume-aware ones (OBV/MFI need volume; flag/breakout use it
# for the dry-up / breakout-volume check when present).
_CLOSE_STRATS = [trend, ma_crossover, ma_crossover_swing, support_resistance, fibonacci,
                 momentum, bollinger, elliott_wave, ichimoku, donchian, aroon]  # both price-only
_VOL_STRATS = [flag_breakout, breakout_retest, obv, mfi]   # flag/breakout use volume for the dry-up check


def _members():
    """Flat iterable of member dicts {ticker, name, sector, region, index} across every cached
    index (S&P 500 … Russell 2000). cached_universe() is {index_name: [member, …]}."""
    uni = equities.cached_universe() or {}
    for members in (uni.values() if isinstance(uni, dict) else []):
        for m in (members or []):
            if isinstance(m, dict) and m.get("ticker"):
                yield m


def universe_tickers() -> list:
    """Every equity ticker across all cached indices (incl. Russell 2000), deduped, in the
    Bloomberg-style form yfin.to_yahoo understands. Empty if no membership is cached yet."""
    return list(dict.fromkeys(m["ticker"] for m in _members()))


def member_meta() -> dict:
    """{ticker: {'name','sector','region'}} from the membership cache — for labelling the page."""
    out: dict = {}
    for m in _members():
        out.setdefault(m["ticker"], {"name": m.get("name") or m["ticker"],
                                     "sector": m.get("sector") or "—", "region": m.get("region") or ""})
    return out


def _merge_history(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Union `new` onto the parquet at `path` — new data wins on overlapping cells, older
    rows and delisted columns are KEPT. Depth-preserving: a routine 504-session refresh
    can never truncate the deep multi-year history the signal cache / ledger sit on."""
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if old.empty:
        return new
    merged = new.combine_first(old)
    return merged.sort_index()


def backfill(tickers, sessions: int = 504):
    """Pull ~`sessions` sessions of close + volume for `tickers` from yfinance and merge
    into the parquet cache (depth-preserving — see _merge_history). Returns the merged
    (close, volume) frames, each date x Bloomberg ticker."""
    _DIR.mkdir(parents=True, exist_ok=True)
    close, vol = yfin.get_ohlcv(list(dict.fromkeys(tickers)), sessions=sessions)
    if not close.empty:
        close = _merge_history(CLOSE_FILE, close)
        close.rename_axis("date").to_parquet(CLOSE_FILE)
    if not vol.empty:
        vol = _merge_history(VOL_FILE, vol)
        vol.rename_axis("date").to_parquet(VOL_FILE)
    return close, vol


def load_history():
    """(close, volume) frames from the cache — the engine + page read these."""
    close = pd.read_parquet(CLOSE_FILE) if CLOSE_FILE.exists() else pd.DataFrame()
    vol = pd.read_parquet(VOL_FILE) if VOL_FILE.exists() else pd.DataFrame()
    return close, vol


def run() -> pd.DataFrame:
    """Run every technical strategy over the cached equity OHLCV → eqta_opportunities.parquet.
    Market names are remapped to company names from the equity membership cache."""
    close, vol = load_history()
    if close.empty:
        return pd.DataFrame()
    frames = [m.find_opportunities(history=close) for m in _CLOSE_STRATS]
    frames += [m.find_opportunities(history=close, volume=vol) for m in _VOL_STRATS]
    df = pd.concat([f for f in frames if f is not None and not getattr(f, "empty", True)],
                   ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        names = _name_map()
        if names:                                     # strategies default market to the raw ticker
            df["market"] = df["instruments"].map(names).fillna(df["market"])
    _DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SIGNALS_FILE, index=False)
    META_FILE.write_text(json.dumps({
        "as_of": pd.Timestamp.today().strftime("%Y-%m-%d %H:%M"),
        "rows": int(len(df)), "names": int(close.shape[1]),
    }))
    return df


def _name_map() -> dict:
    """{ticker: company name} from the equity membership cache, for labelling the signals."""
    return {t: v["name"] for t, v in member_meta().items()}


def load_signals():
    """(opportunities DataFrame, meta dict) from the cache — what the Equities TA page reads."""
    df = pd.read_parquet(SIGNALS_FILE) if SIGNALS_FILE.exists() else pd.DataFrame()
    try:
        meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}
    except Exception:
        meta = {}
    return df, meta


# Hot Sheet provider — the equities twin of tascore.radar_items: the equity TA report's own saved
# threshold mode (the SAME picks the client PDF would write up today), scored on THIS book's
# confluence set and exclude list. The saved min_conviction / min_score bar does the real gating
# (threshold mode may legitimately return nothing); RADAR_TOP_N is just the safety cap, matched
# to the sheet's per-provider cap.
RADAR_TOP_N = 5


def radar_items() -> list:
    """Hot Sheet provider — today's top-conviction equity technical picks, one item per pick on
    the constructive and cautious sides. Reads the cached opportunities frame only (the equities
    snapshot's signal store); never triggers a pull."""
    import sys
    _src = str(Path(__file__).resolve().parent)
    if _src not in sys.path:
        sys.path.insert(0, _src)                   # convreport resolves reportkit via src/
    from src import convreport, hotsheet, tascore, universe
    from src.specs import ta_report_defaults
    sig, _meta = load_signals()
    if sig is None or sig.empty:                   # no equity signals cached yet — quiet, not broken
        return []
    rd = ta_report_defaults("equities")
    sel = convreport.select(sig, top_n=RADAR_TOP_N, mode="threshold",
                            strategies=tascore.confluence_set("equities"),
                            exclude=sorted(universe.report_excluded("equities")),
                            min_conviction=float(rd["min_conviction"]),
                            min_score=float(rd["min_score"]))
    out = []
    for side in ("constructive", "cautious"):
        for p in sel[side]:
            out.append(hotsheet.item(
                tag="EQ-TECH", key=f"{p['market']}:{side}", section="Equities · Technical",
                text=(f"**{p['market']}** clears the desk's technical quality bar on the "
                      f"**{side}** side — **{p['n']}** strategies align, "
                      f"score {p['score']:+.0f}."),
                metric=f"conv {p['conviction']:.0f}", sub="of 100",
                heat=float(p["conviction"]), value=float(p["score"]),
                ticker=p.get("instruments", ""),
                page="eq:Technical Analysis", book="equities"))
    return out
