import pytest
import yaml

from tests.helpers import FakeSearchProvider
from veris.pipeline import run_research


@pytest.fixture
def provider(monkeypatch):
    fake = FakeSearchProvider()
    monkeypatch.setattr("veris.pipeline.build_provider_chain", lambda settings: [fake])
    return fake


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["taxonomy_path", "safety_path", "source_tiers_path"])
async def test_invalid_policy_file_fails_before_any_search(settings_factory, provider, tmp_path, field):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"schema_version": 99}))
    with pytest.raises(ValueError):
        await run_research(query="anything", settings=settings_factory(**{field: bad}))
    assert provider.calls == []
