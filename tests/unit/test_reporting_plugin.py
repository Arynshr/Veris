import json
from types import SimpleNamespace

from tests.reporting import SCHEMA_VERSION, Recorder


def _report(nodeid, outcome, when="call", longrepr=""):
    return SimpleNamespace(
        nodeid=nodeid,
        outcome=outcome,
        when=when,
        duration=0.01234,
        keywords={"unit": 1, "other": 1},
        failed=outcome == "failed",
        passed=outcome == "passed",
        skipped=outcome == "skipped",
        longreprtext=longrepr,
        longrepr=longrepr,
    )


def test_recorder_writes_jsonl_and_log_after_session(tmp_path):
    rec = Recorder(tmp_path)
    rec.pytest_sessionstart(None)
    rec.pytest_runtest_logstart("t::a", None)
    rec.pytest_runtest_logreport(_report("t::a", "passed"))
    rec.pytest_runtest_logstart("t::b", None)
    rec.pytest_runtest_logreport(
        _report("t::b", "failed", longrepr="trace\nAssertionError: boom")
    )
    rec.pytest_runtest_logreport(_report("t::a", "passed", when="setup"))
    assert not list(tmp_path.glob("*.jsonl"))  # written only post-session
    rec.pytest_sessionfinish(None, 1)

    lines = [
        json.loads(x)
        for x in (tmp_path / f"{rec.run_id}.jsonl").read_text().splitlines()
    ]
    tests, session = lines[:-1], lines[-1]
    assert [t["nodeid"] for t in tests] == ["t::a", "t::b"]
    assert (
        tests[1]["outcome"] == "failed" and tests[1]["error"] == "AssertionError: boom"
    )
    assert tests[0]["markers"] == ["unit"] and tests[0]["duration_s"] == 0.0123
    assert session["type"] == "session" and session["schema_version"] == SCHEMA_VERSION
    assert (
        session["counts"] == {"passed": 1, "failed": 1} and session["exit_status"] == 1
    )
    assert (tmp_path / "latest.jsonl").read_text() == (
        tmp_path / f"{rec.run_id}.jsonl"
    ).read_text()
    log = (tmp_path / f"{rec.run_id}.log").read_text()
    assert "[t::b]" in log and "FAILED" in log and "session end" in log
