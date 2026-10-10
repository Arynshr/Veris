import socket

import pytest


def test_network_access_is_blocked():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("example.com", 80))


def test_settings_factory_never_inherits_credentials(settings):
    assert not any([settings.tavily_api_key, settings.firecrawl_api_key, settings.groq_api_key, settings.searxng_base_url])
    assert settings.as_of_date is not None
