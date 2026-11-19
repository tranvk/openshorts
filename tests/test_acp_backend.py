"""Codex ACP subscription backend, exercised through a fake stdio agent."""
import json
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import acp_backend
import acp_bridge
import gemini_worker
import llm_backend


def _fake_agent(response=None, *, auth="chat-gpt", delay=0, crash=False,
                permission=False, raw_text=None, pipeline=False):
    response = response if response is not None else {"windows": []}
    detail_response = {
        "shorts": [{
            "start": 10, "end": 40, "source_window_id": "w0",
            "predicted_score": 91,
            "video_description_for_tiktok": "A clear payoff.",
            "video_description_for_instagram": "A clear payoff.",
            "video_title_for_youtube_short": "The Payoff",
            "viral_hook_text": "Watch this",
            "why": "It has a complete setup and payoff.",
        }]
    }
    # This is intentionally a real subprocess.  It pins framing, request
    # correlation, permission denial, process exit, and timeout cleanup without
    # requiring Codex or network access in CI.
    return f"""
import json, os, sys, time

def send(value):
    sys.stdout.write(json.dumps(value, separators=(\",\", \":\")) + \"\\n\")
    sys.stdout.flush()

for line in sys.stdin:
    message = json.loads(line)
    method = message.get(\"method\")
    request_id = message.get(\"id\")
    if method == \"initialize\":
        send({{"jsonrpc\": \"2.0\", \"id\": request_id, \"result\": {{
            \"protocolVersion\": 1,
            \"agentInfo\": {{\"name\": \"@agentclientprotocol/codex-acp\", \"version\": \"test\"}},
            \"agentCapabilities\": {{\"sessionCapabilities\": {{\"systemPrompt\": {{}}}}}},
        }}}})
    elif method == \"authentication/status\":
        send({{"jsonrpc\": \"2.0\", \"id\": request_id,
              \"result\": {{\"type\": {auth!r}}}}})
    elif method == \"session/new\":
        send({{"jsonrpc\": \"2.0\", \"id\": request_id,
              \"result\": {{\"sessionId\": \"session-test\"}}}})
    elif method == \"session/prompt\":
        if {permission!r}:
            send({{"jsonrpc\": \"2.0\", \"id\": \"permission-test\",
                  \"method\": \"session/request_permission\",
                  \"params\": {{\"sessionId\": \"session-test\", \"options\": []}}}})
            answer = json.loads(sys.stdin.readline())
            if answer.get(\"result\", {{}}).get(\"outcome\", {{}}).get(\"outcome\") != \"cancelled\":
                os._exit(9)
        if {crash!r}:
            os._exit(7)
        time.sleep({delay!r})
        prompt_text = message.get(\"params\", {{}}).get(\"prompt\", [{{}}])[0].get(\"text\", \"\")
        payload = {detail_response!r} if {pipeline!r} and \"source_window_id\" in prompt_text else {response!r}
        text = {raw_text!r} if {raw_text is not None!r} else json.dumps(payload)
        midpoint = max(1, len(text) // 2)
        for chunk in (text[:midpoint], text[midpoint:]):
            send({{"jsonrpc\": \"2.0\", \"method\": \"session/update\", \"params\": {{
                \"sessionId\": \"session-test\", \"update\": {{
                    \"sessionUpdate\": \"agent_message_chunk\",
                    \"content\": {{\"type\": \"text\", \"text\": chunk}},
                }}
            }}}})
        send({{"jsonrpc\": \"2.0\", \"id\": request_id,
              \"result\": {{\"stopReason\": \"end_turn\"}}}})
"""


def _configure(monkeypatch, script):
    monkeypatch.setenv("LLM_PROVIDER", "codex-acp")
    monkeypatch.setenv("CODEX_ACP_COMMAND", sys.executable)
    monkeypatch.setenv("CODEX_ACP_ARGS", json.dumps(["-u", "-c", script]))
    monkeypatch.setenv("ACP_TIMEOUT", "5")


def _valid_score():
    return {
        "windows": [{
            "id": "w0", "start": 0, "end": 90, "score": 88,
            "reason": "Clear hook",
        }]
    }


def test_codex_provider_is_explicit_and_described(monkeypatch):
    _configure(monkeypatch, _fake_agent())
    assert llm_backend.provider() == "codex-acp"
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "codex-acp",
        "model": "subscription-default",
        "transport": "stdio",
        "subscription": True,
        "available": True,
        "visualFeatures": "gemini-required",
    }


def test_valid_acp_response_is_schema_validated(monkeypatch):
    _configure(monkeypatch, _fake_agent(_valid_score(), permission=True))
    parsed, cost = llm_backend.generate_json("untrusted transcript", gemini_worker.ScoreResponse)
    assert parsed["windows"][0]["score"] == 88
    assert cost is None  # ACP does not promise usage or billing data.


def test_authenticated_host_bridge_runs_full_inference_path(monkeypatch):
    _configure(monkeypatch, _fake_agent(_valid_score(), permission=True))
    token = "bridge-test-token-" + ("x" * 32)
    server = acp_bridge.make_server("127.0.0.1", 0, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv(
            "ACP_BRIDGE_URL", f"http://127.0.0.1:{server.server_port}"
        )
        monkeypatch.setenv("ACP_BRIDGE_TOKEN", token)

        parsed, cost = llm_backend.generate_json(
            "untrusted transcript", gemini_worker.ScoreResponse
        )

        assert parsed["windows"][0]["score"] == 88
        assert cost is None
        assert llm_backend.describe() == {
            "provider": "codex-acp",
            "model": "subscription-default",
            "transport": "http-bridge",
            "subscription": True,
            "available": True,
            "visualFeatures": "gemini-required",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_host_bridge_rejects_unauthenticated_health_check(monkeypatch):
    token = "bridge-test-token-" + ("x" * 32)
    server = acp_bridge.make_server("127.0.0.1", 0, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/health"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=2)
        assert exc.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bridge_configuration_requires_strong_token_and_local_url(monkeypatch):
    monkeypatch.setenv("ACP_BRIDGE_URL", "http://host.docker.internal:8765")
    monkeypatch.setenv("ACP_BRIDGE_TOKEN", "short")
    with pytest.raises(acp_backend.ACPError, match="at least 32"):
        acp_backend._bridge_http("/health", None, 0.1)

    monkeypatch.setenv("ACP_BRIDGE_URL", "http://bridge.example.com:8765")
    with pytest.raises(acp_backend.ACPError, match="host.docker.internal"):
        acp_backend.bridge_url()
    monkeypatch.setenv("LLM_PROVIDER", "codex-acp")
    description = llm_backend.describe()
    assert description["available"] is False
    assert "host.docker.internal" in description["error"]


def test_bridge_server_rejects_short_token():
    with pytest.raises(ValueError, match="at least 32"):
        acp_bridge.make_server("127.0.0.1", 0, "short")


def test_bridge_cli_requires_explicit_non_loopback_opt_in(monkeypatch):
    monkeypatch.setenv("ACP_BRIDGE_TOKEN", "x" * 40)
    monkeypatch.setattr(acp_bridge, "_arguments", lambda: SimpleNamespace(
        print_token=False,
        host="0.0.0.0",
        port=8765,
        allow_non_loopback=False,
    ))
    monkeypatch.setattr(
        acp_bridge,
        "make_server",
        lambda *_args: pytest.fail("server must not bind without explicit opt-in"),
    )
    assert acp_bridge.main() == 2


def test_api_key_auth_is_refused_before_prompt(monkeypatch):
    _configure(monkeypatch, _fake_agent(auth="api-key"))
    with pytest.raises(acp_backend.ACPAuthenticationError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "requires a ChatGPT subscription login" in str(exc.value)


def test_missing_auth_has_clear_login_instruction(monkeypatch):
    _configure(monkeypatch, _fake_agent(auth="unauthenticated"))
    with pytest.raises(acp_backend.ACPAuthenticationError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "codex login" in str(exc.value)


def test_schema_violation_never_leaks_into_pipeline(monkeypatch):
    _configure(monkeypatch, _fake_agent({"windows": [{"id": "w0"}]}))
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_malformed_json_is_rejected(monkeypatch):
    _configure(monkeypatch, _fake_agent(raw_text="this is not JSON"))
    with pytest.raises(acp_backend.ACPProtocolError, match="ACP returned malformed JSON"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)


def test_missing_adapter_has_install_instruction(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "codex-acp")
    monkeypatch.setenv("CODEX_ACP_COMMAND", "/definitely/missing/codex-acp")
    monkeypatch.setenv("CODEX_ACP_ARGS", "[]")
    with pytest.raises(acp_backend.ACPError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "@agentclientprotocol/codex-acp" in str(exc.value)


def test_timeout_cancels_and_terminates_provider(monkeypatch):
    _configure(monkeypatch, _fake_agent(delay=5))
    monkeypatch.setenv("ACP_TIMEOUT", "0.15")
    monkeypatch.setattr(acp_backend, "_SHUTDOWN_GRACE", 0.05)
    with pytest.raises(acp_backend.ACPTimeoutError):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)


def test_provider_crash_is_reported(monkeypatch):
    _configure(monkeypatch, _fake_agent(crash=True))
    with pytest.raises(acp_backend.ACPError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "exited unexpectedly" in str(exc.value)


def test_concurrent_requests_use_independent_sessions(monkeypatch):
    _configure(monkeypatch, _fake_agent(_valid_score(), delay=0.05))
    monkeypatch.setenv("ACP_MAX_CONCURRENT", "2")

    def run(_):
        return llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(4)))
    assert [result["windows"][0]["id"] for result in results] == ["w0"] * 4


def test_end_to_end_transcript_clip_selection(monkeypatch):
    """The real two-pass pipeline uses ACP output without constructing Gemini."""
    main = pytest.importorskip("main")
    _configure(monkeypatch, _fake_agent(_valid_score(), pipeline=True))
    monkeypatch.setenv("CLIP_TARGET_MIN", "1")
    monkeypatch.setenv("CLIP_TARGET_MAX", "1")
    words = [
        {"word": f"word{i}", "start": float(i), "end": float(i) + 0.5}
        for i in range(80)
    ]
    transcript = {
        "language": "en",
        "segments": [{
            "start": 0.0, "end": 80.0,
            "text": " ".join(word["word"] for word in words),
            "words": words,
        }],
    }

    result = main.get_viral_clips(transcript, 80.0)

    assert result is not None
    assert len(result["shorts"]) == 1
    assert result["shorts"][0]["source_window_id"] == "w0"
    assert "cost_analysis" not in result  # ACP usage/billing is unavailable.


def test_child_environment_omits_application_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "paid-openai-secret")
    monkeypatch.setenv("CODEX_API_KEY", "paid-codex-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("ACP_CODEX_HOME", "/isolated/codex-home")
    env = acp_backend._child_env()
    assert "OPENAI_API_KEY" not in env
    assert "CODEX_API_KEY" not in env
    assert "GEMINI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert env["CODEX_HOME"] == "/isolated/codex-home"
    assert env["INITIAL_AGENT_MODE"] == "read-only"
    config = json.loads(env["CODEX_CONFIG"])
    assert config["features"]["shell_tool"] is False
    assert config["web_search"] == "disabled"


def test_unknown_provider_fails_closed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "codxe-acp")
    monkeypatch.setenv("LLM_BASE_URL", "http://paid-or-local.example/v1")
    assert llm_backend.active() is True
    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)


def test_claude_subscription_provider_is_explicitly_blocked(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-acp")
    with pytest.raises(RuntimeError, match="Anthropic currently forbids"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
