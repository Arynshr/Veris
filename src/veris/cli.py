"""
CLI entrypoint (spec section 13).

    veris research "state of solid-state batteries in 2026"
    veris research "the board of directors of XYZ company" --mode background_check
    veris research --subject "Jane Doe" --context "CEO of Acme Corp" --mode background_check
    veris inspect <run_id>
    veris sources <run_id>
    veris verify <run_id> [--subject "Jane Doe"]
    veris events <run_id>
"""
import asyncio
import json
import logging

import typer 
from rich.console import Console 
from rich.table import Table

from veris.core.config import configure_logging, get_settings
from veris.core.models import EventStatus, PipelineStage, ResearchMode, ResolvedSubject, SubjectType
from veris.core.storage import RunPersistence
from veris.pipeline import stream_research

app = typer.Typer(add_completion=False, help="Veris - evidence-grounded research assistant.")
console = Console()

_STATUS_STYLE = {EventStatus.STARTED: ("◐", "yellow"), EventStatus.COMPLETED: ("●", "green"), EventStatus.FAILED: ("✗", "red")}


def _print_event(event) -> None:
    icon, style = _STATUS_STYLE[event.status]
    subject = event.data.get("subject", "")
    subject_part = f"[dim]{subject:<18}[/dim]" if subject else " " * 18
    console.print(f"[{style}]{icon}[/{style}] [bold]{event.stage.value:<13}[/bold] {subject_part} {event.message}")


@app.command()
def research(
    query: str = typer.Argument(None, help='Free-text request, e.g. "state of solid-state batteries in 2026".'),
    mode: ResearchMode = typer.Option(None, "--mode", "-m", help="Override auto-detected mode (general | background_check)."),
    subject: str = typer.Option(None, "--subject", "-s", help="Skip query understanding: research this exact name/topic directly."),
    subject_type: SubjectType = typer.Option(SubjectType.TOPIC, "--type", "-t", help="Subject type (only used with --subject)."),
    context: str = typer.Option("", "--context", "-c", help="Disambiguating context, e.g. role, employer, location, timeframe."),
    output: str = typer.Option("report.md", "--output", "-o", help="Path to write the final Markdown report."),
    run_id: str = typer.Option(None, "--run-id", help="Optional explicit run id (default: generated)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug-level logging."),
) -> None:
    """Run the full research pipeline end-to-end, printing progress as it happens."""
    if not query and not subject:
        console.print("[red]Provide either a free-text QUERY or --subject.[/red]")
        raise typer.Exit(code=1)

    configure_logging(logging.DEBUG if verbose else logging.INFO)
    settings = get_settings()

    override = None
    if subject:
        override = [ResolvedSubject(name=subject, subject_type=subject_type, resolution_confidence=1.0)]
        effective_query = f"Research on {subject}" + (f" - {context}" if context else "")
    else:
        effective_query = query
        console.print(
            "[dim]Parsing request - role/group queries (\"the board of directors of X\") trigger "
            "an extra subject-resolution search before research begins.[/dim]\n"
        )

    async def _run() -> str | None:
        final_run_id = run_id
        async for event in stream_research(
            query=effective_query, output_path=output, settings=settings, run_id=run_id,
            resolved_subjects_override=override, research_mode_override=mode,
        ):
            _print_event(event)
            if event.stage == PipelineStage.OUTPUT and event.status == EventStatus.COMPLETED:
                final_run_id = event.data.get("run_id", final_run_id)
        return final_run_id

    try:
        completed_run_id = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - surface pipeline failure clearly; artifacts persisted up to that point
        console.print(f"\n[red]Run failed:[/red] {exc}")
        console.print("[dim]Progress up to the failure is in runs/<run_id>/events.jsonl and other stage artifacts.[/dim]")
        raise typer.Exit(code=1)

    console.print(f"\n[green]Research completed.[/green] [dim](veris inspect {completed_run_id})[/dim]")


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
        console.print(f"[{style}]{icon}[/{style}] [bold]{record['stage']:<13}[/bold] {subject_part} {record['message']}")


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
    subject: str = typer.Option(None, "--subject", "-s", help="Filter to one subject (default: all subjects in the run)."),
) -> None:
    """Show claim verification results for a run, including subject-match confidence."""
    store = RunPersistence(get_settings(), run_id=run_id)
    try:
        results = store.load_verification(subject=subject)
    except FileNotFoundError:
        console.print(f"[red]No verification results found for run '{run_id}'" + (f", subject '{subject}'" if subject else "") + ".[/red]")
        raise typer.Exit(code=1)

    if not results:
        console.print(f"[yellow]No verification results found for run '{run_id}'.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title=f"Claim verification for run {run_id}" + (f" — {subject}" if subject else ""))
    table.add_column("Claim ID")
    table.add_column("Status")
    table.add_column("Subject Match")
    table.add_column("Source Quality")
    table.add_column("Reasons")
    for r in results:
        style = "green" if r["status"] == "VERIFIED" else "red"
        table.add_row(
            r["claim_id"], f"[{style}]{r['status']}[/{style}]",
            str(r["subject_match_score"]), str(r["source_quality_score"]),
            "; ".join(r.get("reasons", [])),
        )
    console.print(table)


if __name__ == "__main__":
    app()
