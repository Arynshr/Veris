import json

import httpx
import pytest
from pydantic import SecretStr

from veris.testing import make_settings
from veris.core.llm import (
    LLMClient,
    LLMConfig,
    LLMConfigError,
    missing_llm_fields,
    resolve_llm_config,
)

pytestmark = pytest.mark.unit

PROVIDERS = "schema_version: 1\nproviders:\n  openai: {base_url: https://x.test/v1}\n  custom: {key_required: false}\n"


def client(handler, kind="openai"):
    cfg = LLMConfig(
        provider="p",
        kind=kind,
        base_url="https://x.test/v1",
        model="m1",
        api_key=SecretStr("k"),
        timeout=5,
        max_retries=0,
        max_tokens=64,
    )
    return LLMClient(cfg, transport=httpx.MockTransport(handler))


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_validate_listed_model():
    c = client(lambda r: httpx.Response(200, json={"data": [{"id": "m1"}]}))
    c.validate()
    assert c.validated


def test_validate_bad_key():
    c = client(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(LLMConfigError, match="API key rejected"):
        c.validate()


def test_validate_unknown_model_suggests():
    def handler(r):
        if r.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m2"}, {"id": "other"}]})
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    with pytest.raises(LLMConfigError, match="Did you mean: m2"):
        client(handler).validate()


def test_validate_probe_when_listing_unsupported():
    c = client(
        lambda r: httpx.Response(404) if r.url.path.endswith("/models") else ok("ok")
    )
    c.validate()


def test_schema_mode_and_temperature_fallback():
    seen = []

    def handler(r):
        body = json.loads(r.content)
        seen.append(body)
        if "temperature" in body:
            return httpx.Response(
                400, json={"error": {"message": "temperature unsupported"}}
            )
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "response_format not supported"}}
            )
        return ok('```json\n{"a": 1}\n```')

    assert json.loads(
        client(handler).complete("s", "u", schema={"type": "object"})
    ) == {"a": 1}
    assert (
        "temperature" not in seen[-1]
        and seen[-1]["response_format"]["type"] == "json_object"
    )


def test_anthropic_text_blocks():
    def handler(r):
        assert r.headers["x-api-key"] == "k"
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]})

    assert client(handler, kind="anthropic").complete("s", "u") == "hi"


def test_config_resolution(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(PROVIDERS)
    base = {"llm_providers_path": path}
    assert missing_llm_fields(make_settings(**base)) == ["llm_provider", "llm_model"]
    s = make_settings(llm_provider="openai", llm_model="m", **base)
    assert missing_llm_fields(s) == ["llm_api_key"]
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        resolve_llm_config(s)
    with pytest.raises(LLMConfigError, match="Unknown LLM provider"):
        missing_llm_fields(make_settings(llm_provider="nope", llm_model="m", **base))
    assert missing_llm_fields(
        make_settings(llm_provider="custom", llm_model="m", **base)
    ) == ["llm_base_url"]
    cfg = resolve_llm_config(
        make_settings(llm_provider="openai", llm_model="m", llm_api_key="k", **base)
    )
    assert cfg.base_url == "https://x.test/v1"
