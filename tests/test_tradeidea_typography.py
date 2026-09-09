"""Trade Idea: the FILLABLE copy must set type exactly like the BLANK template.

Both artefacts come off one template (templates/tradeidea.html), but the writable slots are
rendered two ways — a static div in the blank PDF, a <textarea> in the copy the writer types
into — and the fillable one carries a `font: inherit` reset so a slot inside a banner or the
title band picks up the furniture around it.

That reset is a loaded gun. It sits at single-class specificity, so ANY per-slot size declared
at the same weight loses to it on source order and the slot silently falls back to whatever its
parent uses. It happened: the prose box took the body's 16px default and printed a quarter
larger than the rest of the page (Ben: "the rationale writing is massive compared to the rest of
the page ... totally out of sync") — while every other slot looked right, which is exactly why
eyeballing one render didn't catch it. A form control also drops `text-transform`, which would
have left the instrument as the one banner in the deck that isn't upper case.

So this compares the two variants slot by slot in a real browser: same size, same style, same
weight, same transform. A slot whose type must differ has to be added to EXPECTED_DIFFS on
purpose — that deliberate edit is the review gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tradeidea as ti                                            # noqa: E402

# Every writable slot, across all three section kinds.
PAYLOAD = {
    **ti.default_payload("FICC", "Bonds"),
    "topic": "US 10Y — seasonal window",
    "sections": [
        {"id": "p1", "kind": "prose", "title": "Prose", "weight": 1.0, "prompt": "x"},
        {"id": "t1", "kind": "table", "title": "Table", "rows": 2, "refline": True,
         "cols": ["Instrument", "Direction", "Two-way price"]},
        {"id": "c1", "kind": "chart", "title": "Chart", "weight": 2.2, "prompt": "y",
         "caption": "Source: Bloomberg"},
    ],
}
SLOTS = ["f_topic", "f_date", "f_oneline", "f_market", "f_side", "f_name", "f_mail1",
         "f_mail2", "f_p1", "f_t1_ref", "f_t1_0_0", "f_c1_cap"]
EXPECTED_DIFFS: dict[str, str] = {}      # none: every slot should match its blank counterpart

_PROBE = """(ids) => Object.fromEntries(ids.map(id => {
  const el = document.getElementById(id);
  if (!el) return [id, "MISSING"];
  const c = getComputedStyle(el);
  return [id, [c.fontSize, c.fontStyle, c.fontWeight, c.textTransform].join(" ")];
}))"""


@pytest.fixture(scope="module")
def computed():
    """{variant: {slot: 'size style weight transform'}} measured in headless Chromium."""
    playwright = pytest.importorskip("playwright.sync_api")
    from reportkit import launch_chromium
    out = {}
    with playwright.sync_playwright() as pw:
        browser = launch_chromium(pw)
        try:
            page = browser.new_page(viewport={"width": 794, "height": 1123})
            for variant, editable in (("blank", False), ("fillable", True)):
                page.set_content(ti.render_html(PAYLOAD, editable=editable), wait_until="load")
                out[variant] = page.evaluate(_PROBE, SLOTS)
        finally:
            browser.close()
    return out


def test_every_slot_is_rendered(computed):
    missing = {v: [s for s in SLOTS if computed[v][s] == "MISSING"] for v in computed}
    assert not any(missing.values()), f"slots absent from a variant: {missing}"


@pytest.mark.parametrize("slot", SLOTS)
def test_fillable_type_matches_blank(computed, slot):
    blank, fillable = computed["blank"][slot], computed["fillable"][slot]
    if slot in EXPECTED_DIFFS:
        pytest.skip(f"{slot}: {EXPECTED_DIFFS[slot]}")
    assert fillable == blank, (
        f"{slot} renders differently in the copy the writer types into.\n"
        f"  blank template : {blank}\n"
        f"  fillable copy  : {fillable}\n"
        "Usually the `font: inherit` reset on .fld winning on source order — give the slot a "
        "compound selector (e.g. `textarea.field.fld`) so it outranks the reset."
    )


def test_prose_box_is_the_house_body_size(computed):
    """The size that broke: 9.5pt = 12.667px at 96dpi, NOT the browser's 16px default."""
    assert computed["fillable"]["f_p1"].startswith("12.6"), computed["fillable"]["f_p1"]


@pytest.mark.parametrize("desk,address", sorted(ti.DESK_EMAIL.items()))
def test_desk_address_is_content_not_a_prompt(desk, address):
    """The desk's shared address must PRINT whether or not anyone touched that box.

    It was a placeholder, and placeholders are stripped when the copy prints — so a note went
    out with the second contact line simply gone (Ben, 2026-09-09), which is how every other
    report in the deck renders it as fixed text. It stays editable; it just isn't a prompt.
    """
    blank = ti.render_html({**ti.default_payload(desk, ""), "topic": "x"}, editable=False)
    fillable = ti.render_html({**ti.default_payload(desk, ""), "topic": "x"}, editable=True)
    assert address in blank, f"{desk}: missing from the blank template"
    # in the fillable copy it must be the field's VALUE (between the tags), not only its
    # placeholder — a placeholder would vanish on print exactly as it did before.
    body = fillable.split('id="f_mail2"', 1)[1].split("</textarea>", 1)[0]
    assert body.rstrip().endswith(address), f"{desk}: desk address is not the field's value"


def test_instrument_slot_stays_upper_case(computed):
    """Every banner in the deck is upper-cased by the shared sheet; a form control resets it."""
    assert "uppercase" in computed["fillable"]["f_market"]
