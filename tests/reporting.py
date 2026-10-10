"""Structured test logging and JSONL results, written after the session ends."""

import json
import logging
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(test)s] %(message)s"
_MARKERS = {"unit", "integration", "regression"}


class _TestFilter(logging.Filter):
    current = "session"

    def filter(self, record: logging.LogRecord) -> bool:
        record.test = _TestFilter.current
        return True


class Recorder:
    def __init__(self, results_dir: Path) -> None:
        self.dir = results_dir
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        self.records: list[dict] = []
        self.started = time.monotonic()
        self.started_at = datetime.now(UTC).isoformat()
        self._handler: logging.Handler | None = None
        self.log = logging.getLogger("veris.tests")

    def pytest_sessionstart(self, session) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.dir / f"{self.run_id}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler.addFilter(_TestFilter())
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        self._handler = handler
        self.log.info("session start run_id=%s python=%s", self.run_id, platform.python_version())

    def pytest_runtest_logstart(self, nodeid, location) -> None:
        _TestFilter.current = nodeid
        self.log.info("START")

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" or (report.when != "call" and not report.passed):
            error = None
            if report.failed:
                error = str(report.longreprtext).strip().splitlines()[-1][:500]
            elif report.skipped and isinstance(report.longrepr, tuple):
                error = str(report.longrepr[-1])[:200]
            self.records.append(
                {
                    "type": "test",
                    "run_id": self.run_id,
                    "nodeid": report.nodeid,
                    "when": report.when,
                    "outcome": report.outcome,
                    "duration_s": round(report.duration, 4),
                    "markers": sorted(m for m in report.keywords if m in _MARKERS),
                    "error": error,
                }
            )
            self.log.log(
                logging.ERROR if report.failed else logging.INFO,
                "%s when=%s %.3fs%s",
                report.outcome.upper(),
                report.when,
                report.duration,
                f" :: {error}" if error else "",
            )

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        counts: dict[str, int] = {}
        for rec in self.records:
            counts[rec["outcome"]] = counts.get(rec["outcome"], 0) + 1
        summary = {
            "type": "session",
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "duration_s": round(time.monotonic() - self.started, 3),
            "exit_status": int(exitstatus),
            "counts": counts,
            "python": platform.python_version(),
            "platform": sys.platform,
        }
        path = self.dir / f"{self.run_id}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in [*self.records, summary]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(path)
        shutil.copyfile(path, self.dir / "latest.jsonl")
        self.log.info("session end exit=%s counts=%s results=%s", exitstatus, counts, path)
        if self._handler:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
        _TestFilter.current = "session"


def results_dir() -> Path:
    return Path(os.environ.get("VERIS_TEST_RESULTS_DIR", "test_results"))
