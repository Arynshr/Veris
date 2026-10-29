# Veris

Veris is a command-line research and screening assistant. Give it a topic, or a person or company to screen, and it plans searches, retrieves pages, extracts evidence, verifies claims and writes a cited Markdown report. Every statement in the report can be traced back to a source URL.

```
Query → Understand → Resolve subject → Plan → Search → Retrieve → Registry
      → Evidence → Classify → Verify → Synthesize → Report
```

Provenance chain (never broken): `Report → Claim → Evidence → Document → Source URL`

> **Screening aid only.** Output is not a legal, FCRA or GDPR determination. Identity is matched on name and context terms, so namesakes can still slip through. Review findings before acting on them.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Services and dependencies](#services-and-dependencies)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [CLI reference](#cli-reference)
7. [Run artifacts](#run-artifacts)
8. [Report anatomy](#report-anatomy)
9. [Verification model](#verification-model)
10. [Safety and compliance](#safety-and-compliance)
11. [Testing and evaluation](#testing-and-evaluation)
12. [Troubleshooting](#troubleshooting)
13. [Project layout](#project-layout)
14. [Roadmap](#roadmap)

---

## Features

- **Two modes.** `general` researches a topic. `background_check` screens a person or company for adverse news. The mode is detected automatically unless you set `--mode`.
- **Subject resolution.** Handles a single subject, a named list, or a role group such as "the board of directors of X".
- **Resilient retrieval.** Search providers sit behind one interface with a fallback chain. A failing provider or page never ends the run.
- **Traceable evidence.** Passages are cleaned of crawler markup and boilerplate, bounded, and linked back to their document and URL.
- **Identity gating.** Exact full-name matching, namesake rejection and anchor terms reduce misattribution.
- **Rule-based classification.** Categories, legal status and severity come from a YAML taxonomy, not a model.
- **Three-state verification.** `VERIFIED`, `SINGLE_SOURCE` or `CONFLICTED`. There are no probability scores.
- **Prompt-injection defence.** Instruction-like passages are flagged at evidence time and kept out of synthesis.
- **Deterministic reports.** `report.json` is canonical and `report.md` is rendered from it with Jinja2. The same inputs give byte-identical output.
- **Persisted runs.** Every stage writes artifacts under `runs/<run_id>/` so a run can be inspected afterwards.

**LLM budget.** Only three call sites use a model: query understanding, singular-role resolution, and one executive-summary slot per run. Everything else is deterministic.

---

## Architecture

A linear, asynchronous pipeline with Pydantic contracts between stages.

| Stage | Module | Responsibility |
|---|---|---|
| Intake | `veris/intake/` | spaCy NER, query understanding, subject resolution |
| Planning | `veris/retrieval/planning.py` | Template-based query plan |
| Search | `veris/retrieval/search.py` | Tavily → Firecrawl → SearXNG provider chain |
| Retrieval | `veris/retrieval/extraction.py`, `crawl_policy.py` | Crawl4AI extraction with robots.txt, paywall and denial handling |
| Registry | `veris/core/documents.py`, `storage.py` | Canonical-URL and content-hash dedup, persistence |
| Evidence | `veris/analysis/evidence.py`, `cleaning.py` | Passage selection by TF-IDF, cleaning, noise and identity flags |
| Classification | `veris/analysis/classification.py`, `taxonomy.py`, `temporal.py` | Category, legal status, severity, dates |
| Verification | `veris/analysis/verification.py`, `sources.py` | Claim clustering, source tiers, three-state verdict |
| Safety | `veris/analysis/safety.py` | Injection-pattern flagging |
| Synthesis | `veris/synthesis/` | Slot contract, deterministic validator, summary, report build and render |
| CLI | `veris/cli.py`, `veris/pipeline.py` | Commands and live progress |

Business logic depends only on the narrow interfaces in `veris/core/ports.py`, so providers can be swapped or faked in tests.

---

## Services and dependencies

Veris has no server component. It is a CLI that talks to external services you configure.

| Service | Role | Required? | Credential |
|---|---|---|---|
| **LLM provider** | Query understanding, role resolution, one summary slot | **Yes** | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` |
| **Tavily** | Primary web search | One search provider is required | `TAVILY_API_KEY` |
| **Firecrawl** | Fallback web search | One search provider is required | `FIRECRAWL_API_KEY` |
| **SearXNG** | Self-hosted metadata/search | One search provider is required; opt-in | `SEARXNG_BASE_URL` |
| **Crawl4AI + Playwright Chromium** | Page extraction (JS-capable) | **Yes**, local | none |
| **spaCy `en_core_web_sm`** | Entity recognition | **Yes**, local | none |

Supported LLM providers are listed by `veris providers`. They are defined in `config/llm_providers.yaml` and include OpenAI-compatible and Anthropic endpoints. The `custom` provider takes any OpenAI-compatible URL through `LLM_BASE_URL`.

### Optional: self-hosted SearXNG

```bash
docker run -d --name searxng -p 8080:8080 searxng/searxng
```

Veris requests JSON results, so enable the JSON format in the instance's `settings.yml` (`search.formats: [html, json]`) and restart the container. Then set `SEARXNG_BASE_URL=http://localhost:8080` or pass `--searxng-url`. SearXNG is skipped when no URL is configured.

---

## Quick start

### Prerequisites

- Python 3.11 or newer (development uses 3.13)
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- At least one search key, and one LLM key

### 1. Install

```bash
git clone <your-repo-url> veris && cd veris
uv sync                                   # or: python -m venv .venv && pip install -e .
uv run python -m spacy download en_core_web_sm
uv run crawl4ai-setup                     # installs the headless browser
```

On a bare Linux host or container, also run `uv run python -m playwright install-deps chromium`.

### 2. Configure

Interactive, once:

```bash
uv run veris init
```

This asks for a provider, model and key, checks them, and saves them to `~/.veris/.env` with owner-only permissions.

Non-interactive (CI, containers):

```bash
export LLM_PROVIDER=groq
export LLM_MODEL=openai/gpt-oss-20b
export LLM_API_KEY=...
export TAVILY_API_KEY=...
```

### 3. Run

```bash
uv run veris research "state of solid-state batteries in 2026"
```

Open `report.md`, then inspect the run:

```bash
uv run veris inspect <run_id>
uv run veris sources <run_id>
```

---

## Configuration

### Precedence

Highest first: CLI flags, process environment, `./.env`, `<project>/.env`, `~/.veris/.env`.

Never commit secrets. Keep keys in the environment or in `~/.veris/.env`.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` | LLM connection | none |
| `LLM_BASE_URL` | Endpoint for `custom` provider or a proxy | provider default |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_TOKENS` | LLM limits | 60, 2048 |
| `TAVILY_API_KEY` | Tavily search | none |
| `FIRECRAWL_API_KEY` | Firecrawl search | none |
| `SEARXNG_BASE_URL` | SearXNG instance | none |
| `GROQ_API_KEY`, `GROQ_MODEL` | Deprecated; mapped to `LLM_*` with a warning | none |

Every field of `Settings` in `veris/core/config.py` can also be set from the environment using its upper-case name. Common tuning knobs:

| Setting | Meaning | Default |
|---|---|---|
| `MAX_SEARCH_RESULTS_PER_QUERY` | Results kept per query (1-10) | 2 |
| `MAX_QUERIES_PER_PLAN` | Queries per subject | 8 |
| `MAX_DOCUMENTS_PER_SUBJECT` | Pages read per subject | 10 |
| `LOOKBACK_YEARS` | Window for adverse findings | 2 |
| `AS_OF_DATE` | Reference date for the window (for reproducible runs) | today |
| `MAX_FINDINGS_PER_SUBJECT` | Findings shown in the main report body; the rest go to Appendix A | 10 |
| `MIN_FINDING_RELEVANCE` | Below this a finding moves to Appendix A | 0.05 |
| `REJECT_NAMESAKE_DOCUMENTS` | Reject documents naming only a namesake | true |
| `TEMPLATE_VERSION` | Report template under `templates/` | v2 |

### Policy files

All thresholds, taxonomies and patterns live in YAML. They are validated before any search spend and hashed into every run (`config_hash`).

| File | Controls |
|---|---|
| `config/taxonomy.yaml` | Categories, status cues, severity matrix |
| `config/source_tiers.yaml` | Domain tiers and quality scores |
| `config/safety.yaml` | Instruction-like (injection) patterns |
| `config/cleaning.yaml` | Boilerplate patterns, link-density, length and repetition thresholds, snippet size |
| `config/llm_providers.yaml` | Supported LLM providers and endpoints |
| `templates/v2/` | Jinja2 report templates |

Changing any of these changes the config hash, so two reports are comparable only when their hashes match.

---

## CLI reference

```bash
veris --help
```

### `veris research [QUERY]`

| Option | Purpose |
|---|---|
| `-m, --mode general\|background_check` | Override mode detection |
| `-s, --subject NAME` | Research exactly this subject and skip request parsing |
| `-t, --type` | Subject type, used with `--subject` |
| `-c, --context TEXT` | Disambiguating detail: role, employer, city, years |
| `-o, --output PATH` | Report path (default `report.md`) |
| `-n, --max-results`, `--max-queries`, `-d, --max-documents`, `--max-subjects` | Per-run limits |
| `--provider`, `--model`, `--llm-base-url` | LLM overrides |
| `--searxng-url` | Use a SearXNG instance for this run |
| `--run-id` | Name the run |
| `-v, --verbose` | Debug logs |

Examples:

```bash
# General research
veris research "state of solid-state batteries in 2026"

# Screen one person, with context to separate namesakes
veris research --mode background_check --subject "Mr. XX" --context "ABC founder" -o xx.md

# Group request, capped
veris research "background check the board of directors of Acme Corp" --max-subjects 6
```

### Other commands

| Command | Purpose |
|---|---|
| `veris init` | Interactive LLM setup (needs a TTY) |
| `veris providers` | List LLM providers and how to configure them |
| `veris inspect <run_id>` | Run summary |
| `veris sources <run_id>` | Pages used |
| `veris verify <run_id> [-s SUBJECT]` | Verification results |
| `veris events <run_id>` | Replay the pipeline event stream |
| `veris eval [--golden-dir] [-o] [--baseline]` | Score the deterministic stages against golden cases |

---

## Run artifacts

Every run is saved under `runs/<run_id>/`.

```text
runs/<run_id>/
├── subjects/<subject>/
│   ├── plan.json
│   ├── search_results.json
│   ├── evidence.json
│   ├── claims.json
│   └── verification.json
├── documents/            # one JSON per unique retrieved document
├── report.json           # canonical report
└── report.md             # rendered from report.json only
```

Artifacts carry a `schema_version`. Rejected and flagged items (noise, identity mismatch, instruction-like) stay in these files for audit and are excluded from the report. Crawl memory (robots.txt and denial results) is cached in `.veris/crawl_memory.json`.

---

## Report anatomy

`report.md` has a fixed layout:

1. Header: query, mode and lookback window
2. Executive summary (the single model-written slot, validated against its cited claims)
3. Findings per subject: short title, one or two sentence snippet, verification and severity badges, and `[R#]` reference markers
4. Category summary: counts only
5. **References**: one row per canonical URL, with title, domain, source tier, publication date and link
6. Appendix A: secondary findings, grouped by reason
7. Appendix B: evidence rows (`claim → passage → ref → URL → retrieved_at → status`)
8. Coverage: queries, providers, fetched/failed/paywalled/snippet-only counts, date range, out-of-window, undated and excluded counts
9. Methodology and limitations
10. Footer: run ID, config hash, template version

Rules the template enforces:

- Links appear only in References and Appendix B.
- A clean result is never reported bare. The wording depends on coverage, for example "No adverse findings in retrieved sources; N sources could not be retrieved."
- Findings are ordered by severity, then event date (undated last), then relevance, then claim ID.
- Conflicted claims appear under "Disputed", never as established facts.

---

## Verification model

| State | Meaning |
|---|---|
| `VERIFIED` | At least two independent domains, or one primary/top-tier source |
| `SINGLE_SOURCE` | Identity holds, but support is uncorroborated |
| `CONFLICTED` | Identity mismatch, or terminal legal statuses disagree (convicted vs dismissed/acquitted) |

Legal status resolves by precedence: dismissed_acquitted > convicted > settled > charged > alleged. Negations such as "not convicted" are handled, and "denied" does not change status. Severity comes from the status × category matrix in the taxonomy.

---

## Safety and compliance

- **Crawling.** robots.txt is honoured. Pages that refuse automated access (401/403, paywalls) are skipped and remembered, never bypassed. The crawler identifies itself with a configurable user agent.
- **Prompt injection.** Passages that look like instructions are flagged and excluded from synthesis. Evidence is passed to the model as a delimited data block, with temperature 0.
- **Validated synthesis.** Each model-written slot is checked without another model call. Citations must exist and not be flagged. Dates, numbers and amounts must appear in the cited passages. Legal-status wording must match the cited claims. On failure Veris retries once, then falls back to templated bullets.
- **Secrets.** Read only from the environment or `~/.veris/.env` (0600). Logs are redacted of keys.
- **Data handling.** Reports and artifacts contain information about named people. Treat `runs/` as sensitive and apply your own retention policy.

---

## Testing and evaluation

```bash
uv run pytest                                              # unit tests, no network
uv run veris eval --output eval/metrics.json               # score golden cases
uv run veris eval --baseline eval/metrics.json             # exit 1 on any regression
```

Tests use fixtures and fake providers (`veris/testing.py`) and make no network calls. Run `veris eval` after every change to analysis code and keep headline metrics (`findings.f1`, `category.f1`, `state_accuracy`, and the status and severity accuracies) from regressing.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No search providers configured" | No search key or URL set | Set `TAVILY_API_KEY`, `FIRECRAWL_API_KEY` or `SEARXNG_BASE_URL` |
| "The browser used for page extraction is not installed" | Chromium missing | `crawl4ai-setup` or `python -m playwright install chromium` |
| "The browser is missing system libraries" | Linux dependencies missing | `python -m playwright install-deps chromium` |
| spaCy model error | `en_core_web_sm` not installed | `python -m spacy download en_core_web_sm` |
| `veris init` exits with code 1 | No interactive terminal | Set `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` instead |
| LLM config error about key or base URL | Provider needs a key or `LLM_BASE_URL` | Run `veris providers` for the required fields |
| Report says sources could not be retrieved | Paywalls, robots.txt or failures | See Coverage in the report and `veris sources <run_id>` |
| Report missing expected findings | Cleaning or identity gate excluded them | Check the Coverage exclusion counts, then inspect `evidence.json` and `claims.json` flags |
| SearXNG returns nothing | JSON format disabled | Enable `json` under `search.formats` and restart |
| Template not found | `TEMPLATE_VERSION` points to a missing folder | Use `v2`, or restore `templates/<version>/` |

Use `-v` on `research` for debug logs.

---

## Project layout

```text
.
├── config/                 # taxonomy, source tiers, safety, cleaning, LLM providers
├── templates/v2/           # Jinja2 report templates
├── eval/golden/            # golden cases for `veris eval`
├── tests/
├── runs/                   # generated run artifacts (git-ignored)
└── src/veris/
    ├── cli.py              # Typer CLI
    ├── pipeline.py         # stage orchestration and event stream
    ├── intake/             # NER, understanding, resolution
    ├── retrieval/          # planning, search providers, extraction, crawl policy
    ├── analysis/           # evidence, cleaning, classification, verification, sources, safety
    ├── synthesis/          # slots, validator, summary, report build/render
    ├── evaluation/         # golden set and harness
    └── core/               # models, config, storage, LLM client, ports
```

---

## Roadmap

The execution order and acceptance criteria are tracked in `task.md`; cleanup work is in `prune.md`. Planned CLI work not yet reflected above: `list`, `show` and `doctor` commands, run flags such as `--lookback`, `--min-severity`, `--dry-run` and `--subjects-file`, and exit codes `0` ok, `2` partial failure, `3` adverse findings at or above a threshold.

Out of scope: LangGraph or multi-agent orchestration, a frontend, a vector database, PostgreSQL, confidence-as-probability, monitoring mode, sanctions lists, multilingual support and PDF/HTML rendering.

---

## License

Add your license here.
