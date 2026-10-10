"""Per-page user manual — one source, two editions.

A new joiner should be able to open any page, press one button, and come away knowing what
they are looking at, how the numbers are made, where the data came from, and what question
the page exists to answer (Ben, 2026-10-09).

TWO EDITIONS FROM ONE SOURCE
    DESK   — the default. Everything: thresholds, trigger wording, known limits, costs.
    CLIENT — occasionally shown to clients (Ben, 2026-10-10), so it must obey the same rules
             as every other client document: neutral observation and never advice
             ([[client-commentary-not-advice]]), the shared XP disclaimer, and NO product
             name — the client reports carry zero instances of "BASIS" and are XP-branded.

Each section declares its own audience. That is deliberate and it is the heart of this
module: a scrubber can catch the words "sell the rally", but it cannot know that "the OI
chain has been a frozen fixture since June" or the Bloomberg cost budget are things we may
not want in a client's hands. Omission is a judgement, made once, in the tag. Where a fact
belongs in both editions but needs different framing — "coverage: 22 of 89 products meet the
data-quality floor" versus the desk's account of the bad print that caused the floor — the
section carries a `client_body` rewording rather than being dropped, so the client still sees
the limitation.

reportkit.client_safe() then runs over the whole client render as a BACKSTOP and raises
rather than shipping advice. Belt and braces, the same contract the FICC tearsheet uses.

NUMBERS COME FROM THE CODE, NOT FROM PROSE
    Every threshold, window and count in the text is a $placeholder filled at build time from
    the live module constants (see each page's `facts`). A manual that says "flags at 1.5" when
    the page now defaults to 2.0 is worse than no manual, because the reader cannot tell. This
    is the same discipline src/strategy_docs.py already applies to the trigger reference, and
    tests/test_manual.py holds it: every page in the nav must have an entry, and the client
    edition must survive client_safe() with no desk-only section and no product name.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DESK, CLIENT = "desk", "client"
BOTH = "both"


@dataclass(frozen=True)
class Section:
    """One block of the manual.

    audience=BOTH  -> rendered in both editions (through client_safe on the client side)
    audience=DESK  -> desk edition only; the client render never sees it
    client_body    -> same section, reworded for a client (implies the section is BOTH)
    """
    id: str
    heading: str
    body: str
    audience: str = BOTH
    client_body: str | None = None

    def text_for(self, edition: str) -> str | None:
        if edition == CLIENT:
            if self.audience != BOTH:
                return None
            return self.client_body if self.client_body is not None else self.body
        return self.body


@dataclass(frozen=True)
class Page:
    key: str                       # the app's `active` value, so the button needs no mapping
    title: str                     # desk title
    client_title: str              # XP-branded, no product name
    kicker: str
    sections: list
    facts: object = None           # callable -> dict of live values for $substitution


def _facts_for(page: Page) -> dict:
    """Live constants, read from the modules the page actually runs on.

    Guarded: a manual must still render when a store is missing or a module moves, so a
    failed lookup becomes a visible "(unavailable)" rather than an exception on a page the
    reader opened precisely because they did not understand it."""
    try:
        return dict(page.facts() or {}) if callable(page.facts) else {}
    except Exception:
        return {}


class _Missing(dict):
    def __missing__(self, k):            # an un-filled placeholder must be obvious, not blank
        return "(unavailable)"


def render_sections(key: str, edition: str = DESK) -> list:
    """[(heading, body), ...] for one page's edition, with the live numbers substituted."""
    page = PAGES[key]
    facts = _Missing(_facts_for(page))
    out = []
    for sec in page.sections:
        body = sec.text_for(edition)
        if body is None:
            continue
        body = Template(body).safe_substitute(facts)
        if edition == CLIENT:
            from reportkit import client_safe
            # the backstop: raises rather than letting desk phrasing reach a client page
            body = "\n\n".join(client_safe(p) for p in body.split("\n\n"))
        out.append((sec.heading, body))
    return out


# ── the live-fact readers ────────────────────────────────────────────────────
def _hotsheet_facts() -> dict:
    from src import hotsheet, universe
    import inspect
    full = inspect.signature(hotsheet.heat_from_z).parameters["full"].default
    return {"n_providers": len(hotsheet.discover()),
            "n_products": len(universe.INSTRUMENTS),
            "heat_full": f"{float(full):g}"}


def _curve_facts() -> dict:
    from src import curvemon
    return {"n_spreads": len(curvemon.SPREADS),
            "z_window": getattr(curvemon, "WINDOW", "(unavailable)"),
            "z_flag": getattr(curvemon, "Z_THRESHOLD", "(unavailable)"),
            "n_groups": len(getattr(curvemon, "GROUPS", []) or [])}


def _strategy_facts() -> dict:
    from src import tascore, universe
    return {"n_strategies": len(tascore.TA_STRATEGIES),
            "n_products": len(universe.INSTRUMENTS)}


# ── the pages ────────────────────────────────────────────────────────────────
# Three exemplars first (an aggregator, a per-strategy page, a complex multi-section
# module) so the depth and the desk/client split can be judged before the other ~37.
PAGES: dict = {}


def _add(p: Page) -> None:
    PAGES[p.key] = p


_add(Page(
    key="Hot Sheet",
    title="🔥 Hot Sheet",
    client_title="Daily Market Highlights — Methodology",
    kicker="what the whole book flagged this morning",
    facts=_hotsheet_facts,
    sections=[
        Section("purpose", "What this page is for", """
The Hot Sheet answers one question: **of everything the desk measures this morning, what is
worth a second look?** Every other module watches one dimension — volatility, positioning,
the curve, seasonality. Each would happily show you its ten most extreme readings every day,
whether or not today is unusual. The Hot Sheet is the layer above: $n_providers separate
books each nominate what they consider notable across $n_products products, and the page
ranks the nominations against each other so you read ten lines instead of twenty pages.

It is a **triage tool, not a trade list**. A line on the sheet means "this is unusual for
this product"; it does not mean the move is over, or that it is actionable, or that anyone
has checked whether it is tradeable in size.
"""),
        Section("reading", "How to read it", """
Each row carries four things:

- **The claim** — plain English, naming the product and what is unusual about it.
- **A tag** — which book nominated it (VOL, SKEW-Z, COT, SEAS, CURVE, ROLL, FLOW and so on).
- **A metric** — the number behind the claim, almost always a z-score or a percentile.
- **Heat** — 0 to 100, how strongly that book feels about it.

The top strip is capped at two rows per tag, so one noisy book cannot take the whole sheet.
Rows carry NEW and streak badges from the stamped history, so you can tell a fresh
dislocation from one that has been sitting there for a week.
"""),
        Section("maths", "The maths behind it", """
Almost everything reduces to **"how unusual is this for this product, against its own
history?"** — never against other products. A put/call ratio or a skew that is ordinary for
one contract is extreme for another, so cross-product comparison of raw levels is meaningless.

Two conversions are used:

- **z-score** — (today − the mean of its own window) ÷ the standard deviation of that window.
  A z of +2 means today sits two standard deviations above that product's normal.
- **percentile** — where today ranks within its own window, 0 to 100.

**Heat** maps those onto a common 0–100 so books can be ranked against each other.
`heat_from_z` saturates at $heat_full sigma by default; `heat_from_pctl` measures distance
from the 50th percentile, so the 0th and 100th are equally hot.
"""),
        Section("sources", "Where the data comes from", """
Nothing on this page is fetched when you open it. Every provider reads a store written by
the morning pull, which is why the sheet draws instantly and why it is identical for everyone
who opens it that day.

Prices and option settlements come from Bloomberg at the morning pull. Positioning is the
CFTC's weekly Commitments of Traders. Seasonality and curve work run off a ten-year
settlement store held locally. The economic calendar and several fundamental feeds are free
public sources.
"""),
        Section("limits", "What it cannot tell you", """
- **Heat is not comparable across books in the way it looks.** Each provider chooses its own
  saturation point, so a seasonal row at 100 and a vol row at 100 are not making equally
  strong claims. Rank within a tag confidently; across tags, read the metric.
- **It does not know what is tradeable.** Nothing here checks liquidity, bid/offer, or
  whether the dislocation is a data artefact.
- **"Unusual" is not "wrong".** A spread can sit at three sigma because the world changed.
- **Fixed income reads in yield space.** A "long" signal on a bond means long *yield* —
  short the future.
""", client_body="""
- Readings are ranked within each discipline. The intensity score is calibrated separately
  by each underlying book, so comparisons are most reliable within a category.
- The page identifies statistical unusualness only. It does not assess liquidity, execution
  cost, or whether a reading reflects a change in fundamentals rather than a dislocation.
- Fixed income measures are expressed in **yield** terms throughout.
"""),
        Section("desk-gotchas", "Desk notes", """
- `discover()` globs the **top level** of `src/` only — a provider in `src/strategies/` is
  invisible to it. That is a known bug, not a design.
- The sheet is stamped once per morning inside `run_daily.run()`, at the end of the core
  compute phase. Re-running signals re-stamps the same day rather than appending.
- Heat saturation is the live complaint: several books tie at exactly 100 on an ordinary day,
  which is why a rare vol dislocation can rank below three seasonal windows.
""", audience=DESK),
    ],
))

_add(Page(
    key="__strategy__",
    title="Technical strategy pages",
    client_title="Technical Analysis — Methodology",
    kicker="one method, one product set, one trigger",
    facts=_strategy_facts,
    sections=[
        Section("purpose", "What these pages are for", """
There are $n_strategies technical methods, each with its own page, each run over the same
$n_products products every morning. A strategy page shows **one method's view of the whole
book**: which products it currently flags, how strongly, and the chart behind each flag.

The hub ranks products by agreement *across* methods. These pages are the opposite view —
one method, every product — and they exist so you can see what a single method is saying
before it is blended into a score.
"""),
        Section("reading", "How to read it", """
The **trigger control** at the top sets how extreme a reading has to be before the method
flags it. Lower it and you get more flags of lower quality; raise it and you get fewer,
stronger ones. It changes only what is *shown* — it never changes the underlying maths.

Each flagged product shows a direction (long / short / neutral), the metric that produced
it, and a short context line naming the specific evidence.
"""),
        Section("maths", "The maths behind it", """
Each method is a rule over price history, stated precisely in its own entry in the
**buy/sell reference** on the page. That reference is not a textbook description: every
sentence was distilled from the strategy module's actual code during a signal audit, and
each rule is locked by a fixture test — a clean textbook long must flag +1 and its mirror −1.
If the code changes, the entry and the test change together.

Two conventions apply across all of them:

- **Fixed income runs on yields**, not futures prices (short rates as 100 − price, bonds on
  the generic yield). A "long" read means *rising yields* — which is selling the bond future.
- Signals are computed on a **roll-adjusted** price history, so a contract roll does not
  register as a gap. Published levels are shown unadjusted.
""", client_body="""
Each method is a precisely defined rule over price history, and the page states that
definition in full alongside the results. The definitions are not textbook summaries: each
was transcribed from the implementing code during a formal audit, and each is locked by a
regression test in which a textbook example of the pattern must register +1 and its mirror
−1. Where the implementation changes, the stated definition and its test change together.

Two conventions apply across all of them:

- **Fixed income is measured on yields**, not futures prices (short rates as 100 − price,
  bonds on the generic yield). A "long" read there refers to *rising yields*, which
  corresponds to a lower price on the bond future.
- Measures are computed on a **roll-adjusted** price history, so a contract roll does not
  register as a gap. Levels shown on screen are the unadjusted published prices.
"""),
        Section("sources", "Where the data comes from", """
Settlement prices from Bloomberg, stored locally to roughly ten years so a method can be
measured over more than one cycle. Volume-based methods use exchange volume and are not run
on FX futures, where the real market is OTC and the futures volume is unrepresentative.
"""),
        Section("limits", "What it cannot tell you", """
- A flag is **a condition being met, not a forecast**. Methods disagree routinely; that is
  the point of scoring agreement rather than following any one.
- Every method is fitted to history and history is a biased sample.
- The trigger control is a *display* threshold. Moving it does not make a signal better.
""", client_body="""
- A flag indicates that a defined technical condition is currently met. It is an observation
  of the present state, not a projection.
- Methods frequently disagree with one another; readings are most informative in aggregate.
- All measures are derived from historical price behaviour.
"""),
    ],
))

_add(Page(
    key="Curve Monitor",
    title="📐 Curve / RV",
    client_title="Curve & Relative Value — Methodology",
    kicker="the spread book, monitored and ticketed",
    facts=_curve_facts,
    sections=[
        Section("purpose", "What this page is for", """
A fixed book of $n_spreads curve and relative-value spreads — rate curves, cross-market
pairs and calendars — scored every morning so you can see **which relationships are
stretched against their own history** rather than eyeballing them one at a time.

Three sections: the **Monitor** scores the book; **Trade Tickets** turns the stretched ones
into sized, backtested fade ideas with entry, target and stop; **Track record** is the
append-only forward ledger of how those tickets actually did.
"""),
        Section("reading", "How to read it", """
Each spread is scored two ways, because they answer different questions:

- a **rolling z-score** over $z_window — is this stretched versus the *recent regime*?
- a **percentile of the full ten-year store** — is this stretched versus *everything the
  last decade has shown*?

A spread can be two sigma rich on the first and perfectly ordinary on the second; that
disagreement is information, not noise. The flag threshold is currently **$z_flag sigma**.
Levels shown are real market observables — actual front prices and benchmark yields.
"""),
        Section("maths", "The maths behind it", """
**Units are chosen per asset class, because one normalisation does not fit all.**

- **Rate curves and short-rate calendars** are quoted in **basis points**. A short-rate
  future is priced 100 − rate, so the price difference *is* the rate differential; dividing
  by the price would answer no question anyone asks.
- **Outright commodity and equity spreads** are measured as a **percentage of the front
  price**. Raw points flatter whatever has re-rated: a spread measured in points on a market
  that doubled looks extreme purely because the price doubled.
- **Bond spreads** stay in price points; the honest divisor is the contract's DV01.

**Calendars are handled specially.** A raw front-minus-second series switches contract
*pair* at every roll, so it is not a like-for-like history. Levels are therefore ranked on
**annualised carry** — the spread over the front price, scaled by the gap between the two
contracts — and the change series has roll days zeroed so a roll cannot masquerade as a move.

Both legs of every spread are taken from **unadjusted** prices. Differencing a roll-adjusted
series against a raw one produces a spread that drifts without limit.
"""),
        Section("sources", "Where the data comes from", """
Settlement prices from Bloomberg, held locally for about ten years so a percentile has a
decade to rank against. Benchmark yields are the standard published series for each market.
Everything is struck on one common session: taking each product's own last print would
compare an Asian close against a European one.
"""),
        Section("limits", "What it cannot tell you", """
- **A stretched spread is not a mean-reversion signal.** Spreads stay stretched, and
  structural breaks look exactly like extremes until they don't.
- **The book is fixed.** It scores the relationships it was given; it does not search for
  new ones.
- **Ten years is one or two cycles**, not a large sample for a percentile.
- The ledger's hit rate is by design well under half — the tickets target the full mean, and
  most do not reach it.
""", client_body="""
- An extended reading describes the current state of a relationship against its own history.
  Spreads can remain extended, and a structural change in a market is indistinguishable from
  a temporary dislocation at the time it occurs.
- The book covers a fixed, defined set of relationships rather than a search across all
  possible pairs.
- Percentile measures are ranked against approximately ten years of history.
"""),
        Section("desk-gotchas", "Desk notes", """
- Copper is quoted in cents per pound in the deep store — the only product whose store unit
  differs from its screen unit.
- The ticket stop is one sigma *beyond* entry, not the monitor's fixed three sigma; at |z|>3
  the fixed version lands on the profit side.
- Calendars became roll-neutral on 2026-10-06 (carry basis plus roll breaks). The ledger
  restart that implies is still pending.
""", audience=DESK),
    ],
))


# ── rendering ────────────────────────────────────────────────────────────────
def _md(text: str) -> str:
    """The small markdown subset the manual bodies use: **bold**, `code`, - bullets.

    Deliberately not a markdown library — the bodies are ours, the subset is fixed, and a
    dependency that silently renders raw HTML from a body would undo the escaping below."""
    import html
    import re

    def inline(s: str) -> str:
        """Markup is applied to the JOINED text, never line by line.

        Doing it per line silently breaks any **bold** or *italic* span that happens to wrap,
        which is most of them in a hand-written paragraph — the asterisks then print literally
        in the PDF. Bold is matched before italic so `**x**` is not eaten as two italics."""
        s = html.escape(" ".join(s.split()))
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![*\w])\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    out = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith("- "):
            items, cur = [], ""
            for ln in lines:
                if ln.startswith("- "):
                    if cur:
                        items.append(cur)
                    cur = ln[2:]
                else:
                    cur += " " + ln            # a wrapped bullet continues the current item
            if cur:
                items.append(cur)
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
        else:
            out.append("<p>" + inline(" ".join(lines)) + "</p>")
    return "\n".join(out)


def render_html(key: str, edition: str = DESK) -> str:
    from datetime import date

    from jinja2 import Environment, FileSystemLoader

    from reportkit import data_uri

    page = PAGES[key]
    secs = [(h, _md(b)) for h, b in render_sections(key, edition)]
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=True)
    return env.get_template("manual.html").render(
        edition=edition,
        title=page.title if edition == DESK else page.client_title,
        kicker=page.kicker, sections=secs,
        verified=f"{date.today():%d %B %Y}", asof="",
        logo=data_uri(ROOT / "templates" / "assets" / "logo.png"),
    )


def build_pdf(key: str, out_path, edition: str = DESK) -> str:
    from reportkit import render_pdf
    return render_pdf(render_html(key, edition), out_path)


def page_for(active: str) -> str | None:
    """The manual key for the app's current page, or None if it has none yet.

    Every technical strategy page shares one entry — they differ by method, and the method's
    own rule is already on the page in the audited buy/sell reference."""
    if active in PAGES:
        return active
    try:
        from src import tascore
        if active in tascore.TA_STRATEGIES or active.replace("eq:", "") in tascore.TA_STRATEGIES:
            return "__strategy__"
    except Exception:
        pass
    return None


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build a page's user manual")
    ap.add_argument("page")
    ap.add_argument("out_pdf")
    ap.add_argument("--client", action="store_true", help="client edition (XP-branded, no product name)")
    a = ap.parse_args()
    build_pdf(a.page, a.out_pdf, CLIENT if a.client else DESK)
    print(f"Wrote {a.out_pdf}")


if __name__ == "__main__":
    main()
