"""
CLI entrypoint (spec section 13).

    veris research "state of solid-state batteries in 2026"
    veris research "the board of directors of XYZ company" --mode background_check
    veris research --subject "Jane Doe" --context "CEO of Acme Corp" --mode background_check
    veris providers
    veris inspect <run_id>
    veris sources <run_id>
    veris verify <run_id> [--subject "Jane Doe"]
    veris events <run_id>
    veris eval [--baseline eval/baseline.json]
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from veris.core.config import (
    MAX_SEARCH_RESULTS_CEILING,
    configure_logging,
    get_settings,
)
from veris.core.llm import LLMError, load_providers, missing_llm_fields
from veris.core.models import (
    EventStatus,
    PipelineStage,
    ResearchMode,
    ResolvedSubject,
    SubjectType,
)
from veris.core.storage import RunPersistence
from veris.evaluation import EvalReport, compare_reports, load_golden_cases, run_eval
from veris.pipeline import stream_research

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="markdown",
    help="""Veris researches a topic (or screens a person/company) and writes a cited Markdown report.

**Quick start**

1. Put your keys in a `.env` file: `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY` (run `veris providers` to see the choices) and one search key: `TAVILY_API_KEY`, `FIRECRAWL_API_KEY` or `SEARXNG_BASE_URL`.
2. Run `veris research "your topic"`.
3. Open `report.md`. Use `veris inspect RUN_ID` to see what was found and `veris sources RUN_ID` for the pages used.

Every run is saved under `runs/RUN_ID/`. Pages that refuse automated access (robots.txt, 401/403, paywalls) are skipped and remembered, never bypassed.""",
)
console = Console()

_VERIFICATION_STYLE = {
    "VERIFIED": "green",
    "SINGLE_SOURCE": "yellow",
    "CONFLICTED": "red",
}

_STATUS_STYLE = {
    EventStatus.STARTED: ("◐", "yellow"),
    EventStatus.COMPLETED: ("●", "green"),
    EventStatus.FAILED: ("✗", "red"),
}


def _print_event(event) -> None:
    icon, style = _STATUS_STYLE[event.status]
    subject = event.data.get("subject", "")
    subject_part = f"[dim]{subject:<18}[/dim]" if subject else " " * 18
    console.print(
        f"[{style}]{icon}[/{style}] [bold]{event.stage.value:<13}[/bold] {subject_part} {event.message}"
    )


def _ensure_llm(settings):
    try:
        missing = missing_llm_fields(settings)
        providers = sorted(load_providers(settings.llm_providers_path).providers)
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if not missing:
        return settings
    if not sys.stdin.isatty():
        console.print(
            "[red]LLM not configured.[/red] Set "
            + ", ".join(m.upper() for m in missing)
            + " (env/.env) or pass --provider/--model."
        )
        raise typer.Exit(code=1)
    rebuild = lambda u: type(settings)(**{**settings.model_dump(), **u})  # noqa: E731
    updates: dict = {}
    if "llm_provider" in missing:
        updates["llm_provider"] = typer.prompt("LLM provider", type=click.Choice(providers))
    if "llm_model" in missing:
        updates["llm_model"] = typer.prompt("Model id")
    try:
        for name in missing_llm_fields(rebuild(updates)):
            updates[name] = typer.prompt(
                "API key" if name == "llm_api_key" else "Base URL", hide_input=name == "llm_api_key"
            )
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    return rebuild(updates)


@app.command()
def research(
    query: str = typer.Argument(
        None,
        help='What to research, in plain English, e.g. "state of solid-state batteries in 2026". Omit it when using --subject.',
    ),
    mode: ResearchMode = typer.Option(
        None,
        "--mode",
        "-m",
        help="`general` researches a topic; `background_check` screens a person or company for adverse news. Detected automatically if omitted.",
        rich_help_panel="What to research",
    ),
    subject: str = typer.Option(
        None,
        "--subject",
        "-s",
        help="Research exactly this name or topic and skip request parsing.",
        rich_help_panel="What to research",
    ),
    subject_type: SubjectType = typer.Option(
        SubjectType.TOPIC,
        "--type",
        "-t",
        help="Kind of subject (person, topic, ...). Only used with --subject.",
        rich_help_panel="What to research",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Detail that tells similar names apart: role, employer, city, years.",
        rich_help_panel="What to research",
    ),
    output: str = typer.Option(
        "report.md", "--output", "-o", help="Where to write the Markdown report."
    ),
    max_results: int = typer.Option(
        None,
        "--max-results",
        "-n",
        help=f"Search results kept per query (1-{MAX_SEARCH_RESULTS_CEILING}). This run only.",
        rich_help_panel="Limits",
    ),
    max_queries: int = typer.Option(
        None,
        "--max-queries",
        help="Most search queries planned per subject. This run only.",
        rich_help_panel="Limits",
    ),
    max_documents: int = typer.Option(
        None,
        "--max-documents",
        "-d",
        help="Most pages read per subject, best-ranked first. Lower it for faster, cheaper runs.",
        rich_help_panel="Limits",
    ),
    max_subjects: int = typer.Option(
        None,
        "--max-subjects",
        help="Most people a group request can expand to, e.g. 'the board of directors of X'.",
        rich_help_panel="Limits",
    ),
    provider: str = typer.Option(
        None,
        "--provider",
        help="LLM provider, e.g. openai, anthropic, groq. See `veris providers`. Env: LLM_PROVIDER.",
        rich_help_panel="LLM",
    ),
    model: str = typer.Option(
        None, "--model", help="LLM model id. Env: LLM_MODEL.", rich_help_panel="LLM"
    ),
    llm_base_url: str = typer.Option(
        None,
        "--llm-base-url",
        help="Endpoint for the `custom` provider or a proxy. Env: LLM_BASE_URL. The API key is read from LLM_API_KEY or asked for interactively.",
        rich_help_panel="LLM",
    ),
    searxng_url: str = typer.Option(
        None,
        "--searxng-url",
        help="Use your own SearXNG instance, e.g. http://localhost:8080. Off unless set.",
        rich_help_panel="Search",
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Name for this run (default: generated).", rich_help_panel="Other"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show debug logs.", rich_help_panel="Other"
    ),
) -> None:
    """Research a topic or screen a person/company and write a cited report.

    **Examples**

    `veris research "state of solid-state batteries in 2026"`

    `veris research "board of directors of Acme Corp" --mode background_check`

    `veris research --subject "Jane Doe" --context "CEO of Acme Corp" -m background_check`

    Needs an LLM (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`) and a search key. If any are missing in a terminal, you are asked for them.
    """
    if not query and not subject:
        console.print("[red]Provide either a free-text QUERY or --subject.[/red]")
        raise typer.Exit(code=1)

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    base_settings = get_settings()

    overrides = {
        k: v
        for k, v in {
            "max_search_results_per_query": max_results,
            "max_queries_per_plan": max_queries,
            "max_documents_per_subject": max_documents,
            "max_resolved_subjects": max_subjects,
            "searxng_base_url": searxng_url,
            "llm_provider": provider,
            "llm_model": model,
            "llm_base_url": llm_base_url,
        }.items()
        if v is not None
    }
    try:
        # Re-validate (not plain model_copy, which skips field validators)
        # so --max-results etc. still go through the same bounds
        # (e.g. le=MAX_SEARCH_RESULTS_CEILING) as config/env values, and
        # construct a fresh Settings so CLI flags apply only to this run,
        # never mutating the process-wide cached settings from
        # get_settings() (which other concurrent/subsequent runs read).
        settings = (
            type(base_settings)(**{**base_settings.model_dump(), **overrides})
            if overrides
            else base_settings
        )
    except ValidationError as exc:
        console.print(f"[red]Invalid option:[/red] {exc}")
        raise typer.Exit(code=1)

    settings = _ensure_llm(settings)

    override = None
    if subject:
        override = [
            ResolvedSubject(
                name=subject, subject_type=subject_type, resolution_confidence=1.0
            )
        ]
        effective_query = f"Research on {subject}" + (
            f" - {context}" if context else ""
        )
    else:
        effective_query = query
        console.print(
            '[dim]Parsing request - role/group queries ("the board of directors of X") trigger '
            "an extra subject-resolution search before research begins.[/dim]\n"
        )

    degraded: list[str] = []

    async def _run() -> str | None:
        final_run_id = run_id
        async for event in stream_research(
            query=effective_query,
            output_path=output,
            settings=settings,
            run_id=run_id,
            resolved_subjects_override=override,
            research_mode_override=mode,
        ):
            _print_event(event)
            if (
                event.stage == PipelineStage.OUTPUT
                and event.status == EventStatus.COMPLETED
            ):
                final_run_id = event.data.get("run_id", final_run_id)
                degraded.extend(event.data.get("degraded", []))
        return final_run_id

    try:
        completed_run_id = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface pipeline failure clearly; artifacts persisted up to that point
        console.print(f"\n[red]Run failed:[/red] {exc}")
        console.print(
            "[dim]Progress up to the failure is in runs/<run_id>/events.jsonl and other stage artifacts.[/dim]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"\n[green]Research completed.[/green] [dim](veris inspect {completed_run_id})[/dim]"
    )
    if degraded:
        console.print("[yellow]Completed with warnings:[/yellow] " + ", ".join(degraded))


@app.command()
def providers() -> None:
    """List supported LLM providers and how to configure them."""
    try:
        registry = load_providers(get_settings().llm_providers_path)
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    table = Table("Provider", "API style", "Endpoint", "Key needed")
    for name, spec in sorted(registry.providers.items()):
        table.add_row(
            name, spec.kind, spec.base_url or "set LLM_BASE_URL", "yes" if spec.key_required else "no"
        )
    console.print(table)
    console.print("Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY (env or .env), then run `veris research ...`.")


@app.command()
def events(run_id: str = typer.Argument(..., help="Run id to replay.")) -> None:
    """Replay the durable event checkpoint for a run - works even after the
    terminal that started it is gone, or the run failed partway through."""
    store = RunPersistence(get_settings(), run_id=run_id)
    records = store.load_events()
    if not records:
        console.print(f"[yellow]No events.jsonl found for run '{run_id}'.[/yellow]")
        raise typer.Exit(code=1)

    for record in records:
        icon, style = _STATUS_STYLE[EventStatus(record["status"])]
        subject = record.get("data", {}).get("subject", "")
        subject_part = f"[dim]{subject:<18}[/dim]" if subject else " " * 18
        console.print(
            f"[{style}]{icon}[/{style}] [bold]{record['stage']:<13}[/bold] {subject_part} {record['message']}"
        )


@app.command()
def inspect(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
    """Show a summary of a persisted run."""
    store = RunPersistence(get_settings(), run_id=run_id)
    run_json_path = store.run_dir / "run.json"
    if not run_json_path.exists():
        console.print(f"[red]No run.json found for run '{run_id}'.[/red]")
        raise typer.Exit(code=1)

    data = json.loads(run_json_path.read_text(encoding="utf-8"))
    table = Table(title=f"Run {run_id}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def sources(run_id: str = typer.Argument(..., help="Run id to inspect.")) -> None:
    """List retrieved documents for a run."""
    store = RunPersistence(get_settings(), run_id=run_id)
    documents = store.load_documents()
    if not documents:
        console.print(f"[yellow]No documents found for run '{run_id}'.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Sources for run {run_id}")
    table.add_column("Title")
    table.add_column("URL")
    for doc in documents:
        table.add_row(doc.get("title", ""), doc.get("url", ""))
    console.print(table)


@app.command()
def verify(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
    subject: str = typer.Option(
        None,
        "--subject",
        "-s",
        help="Filter to one subject (default: all subjects in the run).",
    ),
) -> None:
    """Show claim verification results for a run, including subject-match confidence."""
    store = RunPersistence(get_settings(), run_id=run_id)
    try:
        results = store.load_verification(subject=subject)
    except FileNotFoundError:
        console.print(
            f"[red]No verification results found for run '{run_id}'"
            + (f", subject '{subject}'" if subject else "")
            + ".[/red]"
        )
        raise typer.Exit(code=1)

    if not results:
        console.print(
            f"[yellow]No verification results found for run '{run_id}'.[/yellow]"
        )
        raise typer.Exit(code=1)

    table = Table(
        title=f"Claim verification for run {run_id}"
        + (f" — {subject}" if subject else "")
    )
    table.add_column("Claim ID")
    table.add_column("Status")
    table.add_column("Subject Match")
    table.add_column("Source Quality")
    table.add_column("Domains")
    table.add_column("Reasons")
    for r in results:
        style = _VERIFICATION_STYLE[r["status"]]
        table.add_row(
            r["claim_id"],
            f"[{style}]{r['status']}[/{style}]",
            str(r["subject_match_score"]),
            str(r["source_quality_score"]),
            str(r.get("independent_domains", "")),
            "; ".join(r.get("reasons", [])),
        )
    console.print(table)


@app.command(name="eval")
def evaluate(
    golden_dir: Path = typer.Option(
        Path("eval/golden"), "--golden-dir", help="Directory of golden case JSON files."
    ),
    output: Path = typer.Option(
        Path("eval/metrics.json"), "--output", "-o", help="Where to write metrics JSON."
    ),
    baseline: Path = typer.Option(
        None,
        "--baseline",
        help="Earlier metrics JSON; exit 1 if any headline metric regresses.",
    ),
) -> None:
    """Score evidence mapping + verification on the golden set, fully offline."""
    try:
        report = run_eval(load_golden_cases(golden_dir), get_settings())
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"Eval: {len(report.cases)} cases")
    table.add_column("Metric")
    table.add_column("Value")
    for name, value in report.headline().items():
        table.add_row(name, f"{value:.4f}")
    table.add_row("findings.precision", f"{report.findings.precision:.4f}")
    table.add_row("findings.recall", f"{report.findings.recall:.4f}")
    console.print(table)
    console.print(f"[dim]Metrics written to {output}[/dim]")

    if baseline:
        regressions = compare_reports(
            report, EvalReport.model_validate_json(baseline.read_text(encoding="utf-8"))
        )
        if regressions:
            console.print("[red]Regression vs baseline:[/red]")
            for line in regressions:
                console.print(f"  {line}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
