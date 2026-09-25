---
name: humanizer
description: Use when Ben gives text and asks to humanise or humanize it: rewrite it to strip every sign of AI writing while keeping the meaning and facts, adding nothing new.
---

# Humanizer

Rewrite the given text so it reads like a person wrote it. Remove every sign of AI writing. Based on the Wikipedia field guide "Signs of AI writing" (WP:AISIGNS).

## Hard rules

1. Keep the meaning. Every fact, number, name, date, claim and source stays.
2. Add nothing. No new facts, examples, opinions, context, flourishes or conclusions.
3. Remove only what is AI padding: puffery, empty analysis, filler, formatting tics. If a sentence carries no fact, cut it.
4. Keep the original language variety (UK or US spelling) and the original register (casual stays casual, formal stays formal, just without the AI tells).
5. Keep roughly the same structure and order unless the structure itself is an AI tell.
6. Output only the rewritten text. No preamble, no notes, no list of changes, unless Ben asks for them.

## What to remove or fix

### Content patterns

- Inflated significance and legacy: "stands as a testament", "plays a pivotal/crucial/vital/key role", "underscores its importance", "reflects broader trends", "marking a shift", "setting the stage for", "indelible mark", "deeply rooted", "evolving landscape", "focal point". State the plain fact instead or cut.
- Canned notability claims: "has been featured in national media outlets", "independent coverage", "active social media presence", "profiled in leading publications". Keep a source only if the text names what it said.
- Superficial -ing tails: sentences ending in ", highlighting/underscoring/emphasizing/reflecting/showcasing/ensuring/fostering/contributing to ...". Cut the tail.
- Promotional tone: "boasts", "vibrant", "rich", "profound", "nestled", "in the heart of", "renowned", "groundbreaking", "diverse array", "commitment to", "natural beauty", "showcasing", "exemplifies". Use neutral words.
- Vague attribution: "experts argue", "observers note", "industry reports suggest", "some critics say", "several sources" when the text gives one or none. Name the source if the text names it, otherwise state the claim plainly or drop the attribution wrapper.
- Formula endings: "Despite its ..., X faces several challenges ...", "Despite these challenges", "Future outlook", upbeat closing speculation. Keep any real facts about challenges, drop the formula.
- Summary closers: "In summary", "In conclusion", "Overall", restating the point at the end of a paragraph.
- Didactic disclaimers: "It's important to note", "It's worth noting", "may vary".
- Knowledge-cutoff and gap talk: "as of my last update", "details are limited", "not widely documented", "based on available information", "maintains a low profile".
- Chatbot talk aimed at the user: "Certainly!", "Of course", "I hope this helps", "Let me know", "Would you like", "Here is a ...". Delete.
- Placeholders left unfilled: [Name], (add URL here), 2025-xx-xx. Flag to Ben rather than invent content.

### Language patterns

- AI vocabulary. Replace or cut: additionally (esp. sentence-start), align with, boasts, bolstered, crucial, deep dive, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate/intricacies, key (adjective), landscape (abstract), meticulous, pivotal, robust, showcase, tapestry (abstract), testament, underscore (verb), valuable, vibrant, seamless, leverage, navigate (abstract), realm, embark, holistic, nuanced, multifaceted, ever-evolving, game-changer, notably, furthermore, moreover.
- Avoided copulas. Turn "serves as", "stands as", "functions as", "represents", "marks" back into "is". Turn "features", "offers", "boasts", "maintains" back into "has". "Began his career as" becomes "was".
- Stiff synonyms. authored to wrote, utilized to used, relocated to moved, attempted to tried, passed away to died, commenced to started.
- Vague connection words: "associated with", "connected to", "in connection with". State the actual relationship if the text gives it.
- Negative parallelisms: "not just X, but Y", "not only ... but also", "It's not X, it's Y", "no X, no Y, just Z", "Y rather than X". Say the point directly.
- Rule of three: stacked triplets of adjectives or phrases used for rhythm. Keep only the items that carry facts.
- Elegant variation: cycling synonyms to avoid repeating a word. Repeat the plain word.
- Uniform rhythm: vary sentence length a little. Use contractions where the register allows.

### Formatting patterns

- Em dashes: replace with commas, full stops, colons or brackets. Target zero.
- Excess bold: remove bold used for emphasis inside prose.
- Inline-header bullet lists ("- **Label:** text"): turn into prose when the list is short or the items are sentences.
- Title Case Headings: switch to sentence case. Remove headings that only hold other headings, and remove a heading that repeats the document title.
- Emoji used as bullets or heading decoration: remove.
- Thematic breaks (---) between every section: remove.
- Tiny tables that read better as a sentence: convert to prose.
- Curly quotes and apostrophes: convert to straight ones.
- Stray markup: contentReference, oaicite, turn0search0, [cite: 1], start_span, grok_card, 【12†L1-4】, [attached_file:1], utm_source=chatgpt.com or openai. Delete (keep the clean URL).

## What not to do

- Do not swap one AI word for another AI word.
- Do not add typos, slang or fake quirks to look human.
- Do not make the text more exciting or more negative than the original.
- Do not cut real hedges that carry meaning (e.g. "about", "estimated").
- Do not change quotes from named people or sources.

## Process

1. Read the whole text. Note every fact, figure and name.
2. Rewrite, applying the lists above.
3. Check the rewrite against the fact list: nothing lost, nothing added.
4. Scan once more for the AI vocabulary list, em dashes, -ing tails, "not just ... but" and triplets.
5. Return only the rewritten text.

If the text is too short or already clean, return it with only the needed fixes. If something is ambiguous (e.g. a placeholder), return the rewrite and add one short line at the end naming it.
