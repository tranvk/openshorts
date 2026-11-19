# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived. `get_viral_clips` builds ~90 s scoring
windows over the transcript, scores them in batches, shortlists the best
`shortlist_target` (`min(10, duration//90 + 2)`, floor 3) and asks the detail pass
for `clip_count_targets(len(shortlist))` clips. The floor matters: jobs that
return only 1-3 clips retain far worse than jobs that return 4-9.

Two invariants keep that chain from leaking:

- **The scoring pass ranks, it does not select.** The prompt scores **every**
  window and the shortlist is the global top N. A prompt that says "choose up to
  3 windows from this batch" caps the shortlist at `3 * n_batches` instead of the
  target and starves every source under ~30 min. Batches are near-equal
  (`score_batches`): a trailing batch of one window would return that window
  whatever its score.
- **The clip floor is enforced in code, not just in the prompt.** When the detail
  pass comes back under `min_clips`, the shortlist windows that produced nothing
  get one more call for the difference. An empty answer is accepted: padding is
  worse than a short list.

`target_clips` on `/api/process` (dashboard: advanced options) pins both ends via
`CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` when the user wants an exact number.

## Silent footage: the vision fallback (`main.get_visual_clips`)

`transcribe_video` raising `NoAudioError`, **or** `speech_is_sparse()` returning
true, both set `transcript = None` and route to `get_visual_clips`. Sparse means
under `MIN_SPEECH_WORDS` (8) in total or under `MIN_SPEECH_WORDS_PER_MIN` (5).
Music-only footage and mic-muted recordings transcribe to a handful of stray
words; without the second test the picker scores those words and cuts around
them.

The vision pass uploads the video, Gemini returns the same `{"shorts"}` shape
(`gemini_worker.VisualResponse`) in the same 15-60 s band, so every later stage
is identical. `CLIP_TARGET_MIN`/`MAX` apply directly (there are no scoring
windows). The transcript is stored as `{"language": "none", "segments": []}`, so
the clips have no subtitles, which is correct.

**This is the one stage that sends Gemini the video instead of frames, on
purpose**: frames can say what kind of video it is, not which 40 seconds to cut.
Gemini counts video at ~300 tokens/second, so an hour is ~1.08M tokens, past a 1M
window. A source longer than `VISION_SEGMENT_SECONDS` (2400) is therefore split by
`clip_selection.vision_segments` into near-equal stretches (no one-minute tail),
each re-encoded to a 360p / 1 fps proxy (`_vision_proxy`: re-encoded, not
stream-copied, or every timestamp would be off by up to a GOP), watched on its
own, offset back to source time, then merged with `dedupe_overlapping` +
`trim_to_best` to `CLIP_TARGET_MAX`. One failed stretch is logged and skipped.
`VISION_MAX_SEGMENTS` (8, ~5 h 20) bounds the spend of one job; past it only the
first stretches are watched. A source that fits one stretch is one stretch: it
goes through the same proxy, never as the raw file (a broken capture, such as an
HLS VOD whose audio will not decode, made Gemini answer 400 INVALID_ARGUMENT).
When the proxy's audio cannot be encoded it is retried without audio. Do not
port the frame trick here. Gemini-only: a
text-only `LLM_BASE_URL` server cannot see footage, and without `GEMINI_API_KEY`
the function logs one line and returns None, which fails the job.

The public `/gta-5-clips`, `/gta-6-clips`, `/wow-forever-clips` and `/platforms/twitch` pages state
these thresholds and the stretch sizes; if the behaviour changes, change
`dashboard/seo/pages.js` and `dashboard/seo/clip-tools.js` too.

## Text providers for the moment picker (`llm_backend.py`)

`LLM_BASE_URL` (+ `LLM_MODEL`, `LLM_API_KEY`) routes the two transcript passes of
`get_viral_clips` to any OpenAI-compatible `/chat/completions` instead of Gemini;
the response is validated with the same pydantic schemas Gemini enforces
server-side, so `main.py` sees one shape. `main.score_batch_size` drops to 3
windows per call there (local contexts are 4-8k; a truncated prompt scores garbage
silently). Self-host `/api/process` then accepts a request without
`X-Gemini-Key` and `/api/config.localLlm` tells the dashboard not to demand one.
Frame-based stages (`layout_picker`, `screencast_layout`, `get_visual_clips`)
stay on Gemini. Never wired in cloud mode: `BILLING_ENABLED` ignores it.

`LLM_PROVIDER=codex-acp` dispatches the same scoring/detail prompts through
`acp_backend.py` and the maintained `@agentclientprotocol/codex-acp` stdio
adapter. Each call starts a fresh adapter in an empty temporary directory,
requires `authentication/status.type == "chat-gpt"`, denies agent permission
requests, and validates the accumulated assistant text with the same Pydantic
schema as Gemini. ACP usage/cost metadata is not standardized, so the provider
returns `(parsed, None)` rather than inventing zero billing.

The child gets an allowlisted environment: runtime/profile/TLS variables only.
In particular `OPENAI_API_KEY`, `CODEX_API_KEY`, Gemini keys, AWS keys, and other
application secrets are not inherited. `INITIAL_AGENT_MODE=read-only` and a
hardened `CODEX_CONFIG` disable shell and web helpers. The client advertises no
filesystem, terminal, MCP, or elicitation capabilities and rejects every client
request except by cancelling `session/request_permission`.

These are application controls, not a complete host isolation boundary: Codex's
read-only sandbox can read files visible to the OS account if a tool becomes
available through a future adapter/config change. A host with sensitive data
must therefore use a dedicated unprivileged OS account or VM. The adapter may
use that account's Codex credential store, but OpenShorts never reads or copies
credential files. `ACP_CODEX_HOME` selects a dedicated profile that the operator
must authenticate separately.

When OpenShorts runs in Docker but the authenticated Codex profile lives on the
Windows host, `acp_bridge.py` provides the deployment boundary. The container
sends an authenticated HTTP request containing only the prompt, JSON schema,
and optional model; the host bridge invokes the same constrained stdio path and
returns only assistant text. The bridge has no generic command, filesystem, or
ACP passthrough endpoint. Its health endpoint is authenticated, bridge URLs are
limited to localhost/`host.docker.internal` by default, and neither credentials
nor the host profile are mounted into the container. Non-loopback binding
requires an explicit CLI opt-in and a Windows firewall rule scoped to Docker's
private network.

No provider failure routes to Gemini. An unknown `LLM_PROVIDER`, a missing
adapter, wrong auth type, auth expiry, process crash, timeout, malformed JSON,
or schema violation fails that job. Calls are process-isolated and bounded by
`ACP_MAX_CONCURRENT`; timeout sends `session/cancel`, closes stdin, then kills
the process group if graceful cleanup fails.

Claude subscription ACP remains unsupported because Anthropic's current legal
guidance forbids third-party apps from routing Pro/Max credentials. Selecting
`LLM_PROVIDER=claude-acp` produces an explicit error, never a Gemini fallback.
