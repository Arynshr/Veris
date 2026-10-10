import hashlib

from veris.core.models import Document, EventStatus, PipelineEvent, PipelineStage
from veris.core.storage import RunPersistence


def _doc(url: str, content: str = "hello world") -> Document:
    return Document(
        document_id=hashlib.sha1(url.encode()).hexdigest()[:16],
        url=url,
        canonical_url=url,
        title="t",
        content=content,
        source="a.example",
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def test_register_documents_dedupes_by_canonical_url(settings_factory):
    store = RunPersistence(settings_factory(), run_id="test-run")
    store.init_dirs()
    registered = store.register_documents(
        [_doc("https://a.example/x"), _doc("https://a.example/x")]
    )
    assert len(registered) == 1
    assert store.total_documents_registered == 1


def test_register_documents_dedupes_by_content_hash_even_if_url_differs(
    settings_factory,
):
    store = RunPersistence(settings_factory(), run_id="test-run")
    store.init_dirs()
    same_content = "identical content across two mirrors"
    registered = store.register_documents(
        [
            _doc("https://a.example/x", same_content),
            _doc("https://b.example/y", same_content),
        ]
    )
    assert len(registered) == 1


def test_events_round_trip(settings_factory):
    store = RunPersistence(settings_factory(), run_id="test-run")
    store.init_dirs()
    store.append_event(
        PipelineEvent(
            stage=PipelineStage.SEARCH,
            status=EventStatus.COMPLETED,
            message="ok",
            data={"subject": "Acme"},
        )
    )
    records = store.load_events()
    assert len(records) == 1
    assert records[0]["stage"] == "search"
    assert records[0]["data"]["subject"] == "Acme"


def test_load_events_missing_file_returns_empty_list(settings_factory):
    store = RunPersistence(settings_factory(), run_id="never-ran")
    assert store.load_events() == []


def test_save_claims_roundtrips_with_schema_version(settings_factory):
    import json

    from veris.core.models import Claim

    store = RunPersistence(settings_factory(), run_id="claims-run")
    store.init_dirs()
    store.save_claims([Claim(claim_id="c1", text="t", subject="Acme", evidence_ids=["e1"])], subject="Acme")
    path = next(store.run_dir.rglob("claims.json"))
    data = json.loads(path.read_text())
    assert data[0]["claim_id"] == "c1" and data[0]["schema_version"] == 2

