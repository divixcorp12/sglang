"""The server reports ready only after the scheduler's post-warm-up work: the startup warm-up runner posts
/end_startup_warmup after its last warm-up request, before it sets the server Up (CPU)."""

from types import SimpleNamespace

import pytest
import requests

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or {}
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(str(self.status_code))


@pytest.fixture
def server(monkeypatch):
    tokenizer_manager = SimpleNamespace(server_status=ServerStatus.Starting)
    monkeypatch.setattr(http_server, "_global_state", SimpleNamespace(tokenizer_manager=tokenizer_manager))
    monkeypatch.setattr(http_server, "ssl_verify_of", lambda server_args: True)
    monkeypatch.setattr(http_server.time, "sleep", lambda seconds: None)
    posts = []

    def post(url, **kwargs):
        posts.append((url.rsplit("/", 1)[-1], kwargs, tokenizer_manager.server_status))
        if url.endswith("/end_startup_warmup"):
            return FakeResponse({"success": True, "messages": ["copy engine armed; quiet after 0.20 s"]})
        return FakeResponse()

    monkeypatch.setattr(http_server.requests, "post", post)
    monkeypatch.setattr(
        http_server.requests, "get", lambda url, **kwargs: FakeResponse({"is_generation": True})
    )
    return SimpleNamespace(
        args=SimpleNamespace(url=lambda: "http://127.0.0.1:30000"), tokenizer_manager=tokenizer_manager, posts=posts
    )


def test_the_server_warm_up_ends_the_startup_warm_up_before_it_reports_ready(server):
    assert http_server._execute_server_warmup(server.args)
    names = [name for name, _, _ in server.posts]
    assert names == ["generate", "end_startup_warmup"]
    _, kwargs, status_at_call = server.posts[1]
    assert status_at_call == ServerStatus.Starting, "ready was reported before the scheduler's post-warm-up work"
    assert isinstance(kwargs["json"]["warmup_ended_at"], float)
    assert server.tokenizer_manager.server_status == ServerStatus.Up


def test_a_failed_end_of_warm_up_warns_and_the_server_still_starts(server, monkeypatch, caplog):
    def post(url, **kwargs):
        if url.endswith("/end_startup_warmup"):
            raise requests.exceptions.ConnectionError("down")
        return FakeResponse()

    monkeypatch.setattr(http_server.requests, "post", post)
    assert http_server._execute_server_warmup(server.args)
    assert server.tokenizer_manager.server_status == ServerStatus.Up
    assert any("end_startup_warmup failed" in r.getMessage() for r in caplog.records)
