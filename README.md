# Veris

**Veris** (from *verus*, "true") is an evidence-grounded CLI research assistant.
Give it a free-text request and it plans searches, retrieves and deduplicates
sources, ranks evidence by relevance, verifies that claims actually reference
what was asked, and synthesizes a cited Markdown report — for either a general
research question or an adverse-media background check, auto-detected from
your query.

```bash
uv run veris research "state of solid-state batteries in 2026"
uv run veris research "Need a background check on the board of directors of XYZ company"
```

Progress prints as one line per event as it happens (no live-redraw terminal
UI — see "Why no live table" below), and every event is durably checkpointed
to `runs/<run_id>/events.jsonl`, replayable with `veris events <run_id>` even
after the run finished, failed, or the terminal that started it is gone.

## 1. Setup

```bash
uv sync
uv run python -m spacy download en_core_web_sm
cp .env.example .env
```

Edit `.env` and fill in at least one search provider key plus (optionally) Groq:

```bash
TAVILY_API_KEY=tvly-...
FIRECRAWL_API_KEY=fc-...          # optional fallback
SEARXNG_BASE_URL=http://localhost:8080   # optional, self-hosted
GROQ_API_KEY=gsk_...              # optional — without it, query understanding,
                                   # planning, and synthesis all fall back to
                                   # deterministic non-LLM behavior
```

`uv sync` reads `.env` automatically via `pydantic-settings` — no manual export needed.

## 2. Run

```bash
# General research — auto-detected mode, no adverse-media framing
uv run veris research "state of solid-state batteries in 2026"
uv run veris research "compare the EU and US approaches to AI regulation"

# Background check — auto-detected from phrasing ("background check", "due
# diligence", "screen", "vet", "red flags", etc.), or force it with --mode
uv run veris research "Need a background check on the board of directors of XYZ company"
uv run veris research "The CEO and CFO of ABC company" --mode background_check

# Fast path — you already know the exact name/topic, skip understanding/resolution
uv run veris research --subject "Jane Doe" --context "CEO of Acme Corp" --mode background_check
```

- `QUERY` (positional, optional): free-text request.
- `--mode` / `-m`: `general` | `background_check` — overrides auto-detection.
- `--subject` / `-s`: skip understanding/resolution, research this exact name/topic.
- `--type` / `-t`: `person` | `company` | `group` | `topic` (only with `--subject`).
- `--context` / `-c`: disambiguating context (role/employer/location/timeframe).
- `--output` / `-o`: where to copy the final report (default `report.md`); it's
  always also saved under `runs/<run_id>/report.md`.
- `--run-id`, `--verbose`: pin a run id / enable debug logging.

Inspect a past run:

```bash
uv run veris inspect <run_id>              # run metadata + counts
uv run veris events <run_id>                # replay the full event checkpoint
uv run veris sources <run_id>                # documents retrieved (all subjects)
uv run veris verify <run_id>                  # per-claim VERIFIED/CONFLICTED
uv run veris verify <run_id> --subject "Jane Doe"   # filtered to one subject
```

## Layout

Organized by pipeline responsibility, not by file count — each package owns one
concern end-to-end:

```
src/veris/
├── pipeline.py            Async-generator event stream, orchestrates every package below
├── cli.py                  Typer CLI: sequential event-line output + events.jsonl replay
├── core/                  Foundational contracts used by everything else
│   ├── config.py            Settings (.env) + logging
│   ├── models.py            All Pydantic contracts + enums (ResearchMode, etc.)
│   └── storage.py           Run artifact persistence (subject-scoped) + document dedup
├── intake/                Raw query -> concrete research subjects
│   ├── ingestion.py         spaCy NER only — no intent classification (see below)
│   ├── understanding.py     Query decomposition + mode detection: LLM + regex fallback
│   └── resolution.py        Bounded, single-hop role/group -> named subjects
├── retrieval/              Research subject -> raw web content
│   ├── planning.py           Mode-aware query generation (general angles vs. risk categories)
│   ├── search.py              Tavily (news-biased) -> Firecrawl -> SearXNG fallback
│   └── extraction.py          Crawl4AI extraction + URL canonicalization/dedup
├── analysis/                Raw content -> verified, traceable claims
│   ├── evidence.py            TF-IDF relevance ranking, mode-aware candidate gating
│   └── verification.py        Subject-match gate + validity + source quality
└── synthesis/               Verified claims -> cited report
    └── report.py               Mode-aware: general narrative vs. background-check report
```

## What "mode" changes

`ResearchMode` (`general` | `background_check`) is detected once during
understanding and threaded through the rest of the pipeline — nothing branches
on it inside `pipeline.py` itself, each stage just adapts:

| Stage | `general` | `background_check` |
|---|---|---|
| **Planning** | Balanced angle queries: overview, recent developments, analysis, counterpoints | Risk-category queries: legal, regulatory, criminal, financial, reputational |
| **Evidence** | Every sentence is a TF-IDF candidate (min relevance floor); no risk category | Sentence must mention the subject or an adverse keyword first, *then* TF-IDF ranks; tagged by risk category |
| **Verification** | Same subject/topic-match gate, generic "off-topic" reason text | Same gate, explicit "misattribution/namesake risk" reason text |
| **Synthesis** | Plain narrative report per subject, sources cited | Grouped by risk category per subject, "reported by X" phrasing, screening disclaimer |

Subject *resolution* (role/group → named individuals) and the provenance chain
(Report → Claim → Evidence → Document → Source URL) are identical in both modes
— resolving "the founding team of OpenAI" for curiosity uses the same bounded,
single-hop search as resolving "the board of directors of XYZ" for a screen.

## Design notes

- **Subject-match is a hard gate, not a soft signal**, in both modes: a claim
  must actually reference the subject/topic to be VERIFIED. In background-check
  mode this is the primary defense against misattribution — verified with a
  synthetic test where a real fraud allegation about one board member correctly
  attaches only to him, not to a colleague merely mentioned in the same article.
- **TF-IDF relevance ranking**, not a vector DB — computed fresh per document
  with scikit-learn, discarded after the run. Small-scale alternative to
  embeddings; misses paraphrase, documented as a known limitation.
- **Resolution is transparent**: the report's "Subjects Covered" section shows
  which names were given directly (confidence 1.0) vs. found via search (0.6).
- **Provenance chain enforced**: Report → Claim → Evidence → Document → Source URL.
- **No confidence-as-probability**: verification emits VERIFIED/CONFLICTED plus
  explainable heuristic scores, never a probability of truth.

## Known limitations

- TF-IDF misses paraphrased/related content that doesn't share query vocabulary
  (proven in testing: a same-article sentence about a related event was
  correctly excluded because it shared no terms with the topic query).
- Role attribution on resolved subjects is best-effort (a resolution snippet
  naming two roles together can mislabel which role belongs to which name).
- Subjects run sequentially in the event stream; each subject's own searches
  are internally concurrent.
- No cross-source contradiction detection.
- Crawl4AI only; Scrapy multi-page crawling isn't wired in.
