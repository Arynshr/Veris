from veris.evaluation import compare_reports, load_golden_cases, run_eval


def test_golden_set_has_expected_size_and_variety(golden_dir):
    cases = load_golden_cases(golden_dir)
    assert 10 <= len(cases) <= 20
    ids = {c.case_id for c in cases}
    assert {"clean_result", "namesake_different_person"} <= ids


def test_eval_is_offline_and_deterministic(golden_dir, settings):
    cases = load_golden_cases(golden_dir)
    first = run_eval(cases, settings)
    second = run_eval(cases, settings)
    assert first.model_dump_json() == second.model_dump_json()
    assert first.schema_version == 2


def test_clean_and_namesake_cases_produce_no_false_findings(golden_dir, settings):
    report = run_eval(load_golden_cases(golden_dir), settings)
    by_id = {c.case_id: c for c in report.cases}
    for case_id in ("clean_result", "subject_not_in_document"):
        assert by_id[case_id].findings.fp == 0
    assert by_id["namesake_different_person"].findings.fp == 0


def test_compare_reports_flags_regression(golden_dir, settings):
    report = run_eval(load_golden_cases(golden_dir), settings)
    assert compare_reports(report, report) == []
    worse = report.model_copy(update={"state_accuracy": report.state_accuracy - 0.1})
    assert compare_reports(worse, report) == [
        f"state_accuracy: {report.state_accuracy:.4f} -> {worse.state_accuracy:.4f}"
    ]
