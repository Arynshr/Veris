"""Provider-agnostic BYOK LLM client: OpenAI-compatible and Anthropic Messages APIs."""

import difflib
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, SecretStr, ValidationError

from veris.core.config import Settings, get_logger

logger = get_logger(__name__)

SUPPORTED_SCHEMA_VERSION = 1
ANTHROPIC_VERSION = "2023-06-01"
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    pass


class ProviderSpec(BaseModel):
    kind: Literal["openai", "anthropic"] = "openai"
    base_url: str | None = None
    key_required: bool = True


class ProviderRegistry(BaseModel):
    schema_version: int
    providers: dict[str, ProviderSpec]


class LLMConfig(BaseModel, frozen=True):
    provider: str
    kind: Literal["openai", "anthropic"]
    base_url: str
    model: str
    api_key: SecretStr | None
    timeout: float
    max_retries: int
    max_tokens: int


@lru_cache
def load_providers(path: Path) -> ProviderRegistry:
    try:
        registry = ProviderRegistry.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValidationError) as exc:
        raise LLMConfigError(f"Invalid LLM provider file {path}: {exc}") from exc
    if registry.schema_version != SUPPORTED_SCHEMA_VERSION:
        raise LLMConfigError(f"Unsupported provider schema {registry.schema_version}")
    return registry


def _spec(settings: Settings) -> ProviderSpec | None:
    name = settings.llm_provider
    if not name:
        return None
    providers = load_providers(settings.llm_providers_path).providers
    if name not in providers:
        raise LLMConfigError(
            f"Unknown LLM provider '{name}'. Available: {', '.join(sorted(providers))}. "
            "Use 'custom' with LLM_BASE_URL for any OpenAI-compatible endpoint."
        )
    return providers[name]


def _key(settings: Settings) -> str:
    return (
        settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    )


def missing_llm_fields(settings: Settings) -> list[str]:
    missing = [f for f in ("llm_provider", "llm_model") if not getattr(settings, f)]
    if spec := _spec(settings):
        if spec.key_required and not _key(settings):
            missing.append("llm_api_key")
        if not (settings.llm_base_url or spec.base_url):
            missing.append("llm_base_url")
    return missing


def resolve_llm_config(settings: Settings) -> LLMConfig:
    if missing := missing_llm_fields(settings):
        providers = ", ".join(
            sorted(load_providers(settings.llm_providers_path).providers)
        )
        raise LLMConfigError(
            f"LLM is not configured; set {', '.join(m.upper() for m in missing)}. "
            f"Providers: {providers}."
        )
    spec = _spec(settings)
    base_url = (settings.llm_base_url or spec.base_url or "").strip().rstrip("/")
    if not re.match(r"^https?://", base_url):
        raise LLMConfigError(f"Invalid LLM base URL '{base_url}'")
    return LLMConfig(
        provider=settings.llm_provider,
        kind=spec.kind,
        base_url=base_url,
        model=settings.llm_model.strip(),
        api_key=settings.llm_api_key if _key(settings) else None,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.max_retries,
        max_tokens=settings.llm_max_tokens,
    )


def _norm(model: str) -> str:
    return model.removeprefix("models/")


def _detail(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error", resp.text)
        return str(err.get("message", err) if isinstance(err, dict) else err)[:300]
    except (ValueError, AttributeError):
        return resp.text[:300]


def _response_format(mode: str | None, schema: dict | None) -> dict | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    return {"type": "json_object"} if mode == "json_object" else None


class LLMClient:
    def __init__(
        self, cfg: LLMConfig, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.cfg = cfg
        self.validated = False
        key = cfg.api_key.get_secret_value() if cfg.api_key else ""
        headers = {"content-type": "application/json"}
        if cfg.kind == "anthropic":
            headers |= {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        elif key:
            headers["authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=cfg.base_url + "/",
            headers=headers,
            timeout=cfg.timeout,
            transport=transport,
        )

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(self.cfg.max_retries + 1):
            last = attempt == self.cfg.max_retries
            try:
                resp = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if last:
                    raise LLMError(f"{self.cfg.provider} unreachable: {exc}") from exc
            else:
                if resp.status_code not in _RETRY_STATUS or last:
                    return resp
            time.sleep(min(2**attempt, 8))
        raise LLMError(f"{self.cfg.provider} request failed")

    def _check(self, resp: httpx.Response, what: str) -> None:
        if resp.is_success:
            return
        detail = _detail(resp)
        if resp.status_code in (401, 403) or (
            resp.status_code == 400 and "api key" in detail.lower()
        ):
            raise LLMConfigError(
                f"{self.cfg.provider}: API key rejected ({resp.status_code}): {detail}"
            )
        raise LLMError(
            f"{self.cfg.provider} {what} failed ({resp.status_code}): {detail}"
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict | None = None,
        json_mode: bool = False,
    ) -> str:
        if schema:
            system += (
                "\n\nRespond with JSON only, matching this JSON schema:\n"
                + json.dumps(schema)
            )
        elif json_mode:
            system += "\n\nRespond with JSON only."
        try:
            text = (
                self._anthropic(system, user)
                if self.cfg.kind == "anthropic"
                else self._openai(system, user, schema, json_mode)
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"{self.cfg.provider} returned an unexpected response: {exc}"
            ) from exc
        return _FENCE_RE.sub("", text).strip() if (schema or json_mode) else text

    def _openai(
        self, system: str, user: str, schema: dict | None, json_mode: bool
    ) -> str:
        modes = ["json_schema"] if schema else []
        modes += ["json_object"] if schema or json_mode else []
        modes.append(None)
        body: dict = {
            "model": self.cfg.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        for mode in modes:
            body.pop("response_format", None)
            if fmt := _response_format(mode, schema):
                body["response_format"] = fmt
            while True:
                resp = self._request("POST", "chat/completions", json=body)
                if resp.is_success:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                err, bad = _detail(resp).lower(), resp.status_code in (400, 422)
                if bad and "temperature" in body and "temperature" in err:
                    body.pop("temperature")
                    continue
                if (
                    bad
                    and mode
                    and any(k in err for k in ("response_format", "json", "schema"))
                ):
                    break
                self._check(resp, "completion")
        raise LLMError(f"{self.cfg.provider} rejected every response mode")

    def _anthropic(self, system: str, user: str) -> str:
        body: dict = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        while True:
            resp = self._request("POST", "messages", json=body)
            if resp.is_success:
                break
            if (
                resp.status_code == 400
                and "temperature" in body
                and "temperature" in _detail(resp).lower()
            ):
                body.pop("temperature")
                continue
            self._check(resp, "completion")
        return "".join(
            b.get("text", "") for b in resp.json()["content"] if b.get("type") == "text"
        )

    def list_models(self) -> set[str] | None:
        params = {"limit": 1000} if self.cfg.kind == "anthropic" else None
        resp = self._request("GET", "models", params=params)
        if resp.status_code in (401, 403):
            self._check(resp, "model listing")
        if not resp.is_success:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        items = data.get("data", data.get("models")) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return None
        ids = {
            _norm(str(i.get("id") or i.get("name") or ""))
            for i in items
            if isinstance(i, dict)
        }
        return ids - {""} or None

    def validate(self) -> None:
        """Verify endpoint, key and model; falls back to a probe call when the model is unlisted."""
        if self.validated:
            return
        models = self.list_models()
        if not (models and _norm(self.cfg.model) in models):
            try:
                self.complete("Reply with the single word: ok", "ok")
            except LLMConfigError:
                raise
            except LLMError as exc:
                hint = ""
                if models:
                    close = difflib.get_close_matches(
                        _norm(self.cfg.model), sorted(models), n=5, cutoff=0.4
                    )
                    hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise LLMConfigError(
                    f"Model '{self.cfg.model}' is not usable on {self.cfg.provider}: {exc}.{hint}"
                ) from exc
            if models:
                logger.warning(
                    "%s not in %s model list; probe succeeded",
                    self.cfg.model,
                    self.cfg.provider,
                )
        self.validated = True


_clients: dict[tuple, LLMClient] = {}


def get_llm(settings: Settings) -> LLMClient:
    cfg = resolve_llm_config(settings)
    key = (
        cfg.provider,
        cfg.base_url,
        cfg.model,
        cfg.api_key.get_secret_value() if cfg.api_key else "",
    )
    if key not in _clients:
        _clients[key] = LLMClient(cfg)
    return _clients[key]
