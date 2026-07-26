"""AI polish for the report write-ups — conversational desk voice.

Run by the Morning Coffee interpreter (which has `anthropic`):
    <mc_python> ai_polish.py in.json out.json [system.txt]
Reads a JSON list of deterministic notes and writes a JSON list rewritten in a natural,
conversational desk-analyst voice, SAME length and order. The optional third argument
is a file holding an alternate system prompt (e.g. the Volatility Report's swaps
comment); without it the Technical Analysis per-chart brief below is used. It preserves every number /
level / percentage / product name and the neutral, client-safe tone; on ANY failure it
echoes the input unchanged, so the report never depends on the model being reachable.

API key: ANTHROPIC_API_KEY from the environment, else from the Morning Coffee project's
.env (AI/Futures_Movements/.env) — the app/scheduled runs don't export the variable.

Model: AI_POLISH_MODEL (env) if set, else the chain below — the first model that returns
a valid rewrite wins, so a key without access to the newest model degrades gracefully.
"""
import json
import os
import re
import sys
from pathlib import Path

# Preference order: Fable 5 first (the strongest natural-prose writer of the family —
# this is a writing task), then Sonnet 5, then Haiku 4.5 as the always-there backstop.
MODELS = ["claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5"]

_MC_ENV = Path(__file__).resolve().parent.parent / "Futures_Movements" / ".env"

SYSTEM = (
    "You are a senior futures strategist writing the per-chart notes in a desk's weekly "
    "technical report for professional clients. Rewrite each terse, stat-packed note so it "
    "reads like a real person talked a client through the chart — flowing sentences with "
    "natural connective tissue and varied phrasing, an engaged plain-English desk voice "
    "(think a seasoned analyst's morning note): warm and readable, but professional.\n"
    "HARD RULES — never break these:\n"
    "(1) Keep EVERY number, price, level, percentage and ratio EXACTLY as given — never "
    "invent, drop, round or alter a figure — and keep each wrapped in the same **bold** "
    "markers; keep every product and indicator name.\n"
    "(2) Stay neutral and observational: this is client-safe commentary, NOT advice. Never "
    "say buy, sell, long, short, recommend, 'we like', or imply the reader should act. "
    "Describe what the chart is doing and what each level would mean — e.g. 'screens "
    "constructive/cautious', 'has been grinding higher', 'a daily close above **X** would "
    "extend the move', 'a slip back through **Y** would undercut it', 'the measured "
    "objective sits near **Z**'. Preserve the direction and every scenario (the confirmation "
    "trigger, the invalidation with its distance, the objective, the reward:risk) and keep "
    "each on the correct side of price.\n"
    "(3) Vary how each note opens — don't start them all the same way — and keep each to "
    "3-6 flowing sentences.\n"
    "Return ONLY a JSON array of strings, the same length and order as the input — nothing else.")


def _api_key() -> str:
    """ANTHROPIC_API_KEY from the environment, else parsed out of the Morning Coffee .env."""
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        for line in _MC_ENV.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*(.+?)\s*$", line)
            if m:
                return m.group(1).strip().strip("'\"")
    except Exception:
        pass
    return ""


def _rewrite(client, model: str, texts: list, system: str = SYSTEM) -> list:
    msg = client.messages.create(                      # no temperature: deprecated on Claude 5 models
        model=model, max_tokens=8000, system=system,
        messages=[{"role": "user", "content": json.dumps(texts, ensure_ascii=False)}])
    # Claude 5 models may lead with a thinking block — take the text blocks, wherever they sit.
    raw = "".join(getattr(b, "text", "") or "" for b in msg.content).strip()
    if raw.startswith("```"):                          # tolerate a ```json fence
        raw = raw.strip("`")
        raw = raw[4:].strip() if raw.lower().startswith("json") else raw
    got = json.loads(raw)
    if isinstance(got, list) and len(got) == len(texts):
        return [g if isinstance(g, str) and g.strip() else t for g, t in zip(got, texts)]
    raise ValueError(f"unexpected shape from {model}")


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    system = SYSTEM
    if len(sys.argv) > 3:                              # optional per-report system prompt
        with open(sys.argv[3], encoding="utf-8-sig") as f:
            system = f.read().strip() or SYSTEM
    with open(inp, encoding="utf-8-sig") as f:         # -sig: tolerate a BOM from Windows tools
        texts = json.load(f)
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=_api_key())
        pref = os.getenv("AI_POLISH_MODEL", "").strip()
        chain = ([pref] if pref else []) + [m for m in MODELS if m != pref]
        for model in chain:
            try:
                texts = _rewrite(client, model, texts, system)
                sys.stderr.write(f"ai_polish used {model}\n")
                break
            except Exception as e:                     # no access / bad output → next model
                sys.stderr.write(f"ai_polish {model} failed ({e}); trying next\n")
        else:
            sys.stderr.write("ai_polish: all models failed — template text kept\n")
    except Exception as e:                             # offline / no key / no sdk -> echo input
        sys.stderr.write(f"ai_polish fallback ({e})\n")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
