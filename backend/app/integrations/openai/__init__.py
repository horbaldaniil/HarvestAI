"""OpenAI integration package.

Thin async httpx wrapper over the chat-completion endpoint. We use httpx
directly (not the official `openai` SDK) for two reasons:
- Consistency with the OpenMeteo client: same retry/timeout/error pattern.
- The official SDK's streaming surface uses sync iterators by default and
  bundles its own httpx — we'd be wrapping a wrapper.

Used by:
- chat_service.run_streaming → stream_completion (tokens via SSE)
- reports.builder → completion (one-shot for PDF summary paragraph)
"""
