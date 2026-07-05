"""Carry / roll-yield monitor  (placeholder).

Carry needs at least two points on each futures curve (front vs deferred) to
measure the slope / roll yield. The mock feed only has a single front-month
series, so this is intentionally a stub until the Bloomberg strip (e.g. the
generic 1st vs 2nd/3rd contracts) is wired in.
"""
from __future__ import annotations

from .base import empty

STRATEGY = "Carry"
NOTE = (
    "Carry needs multiple contract maturities (front vs deferred). Wire in the "
    "Bloomberg strip (e.g. CL1/CL2..., or generic 2nd/3rd) then compute the "
    "annualised roll yield here."
)


def find_opportunities(history=None):
    return empty()
