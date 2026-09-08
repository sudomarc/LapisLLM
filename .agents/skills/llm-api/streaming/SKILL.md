---
name: streaming-resilience
description: Build reliable LLM streaming protocols with typed events, ordering, cancellation, reconnect-safe semantics, partial failure handling, and terminal states.
version: 1.0.0
status: stable
category: LLM API
---
# Streaming Resilience
## Purpose
Treat streaming as a stateful protocol rather than a sequence of arbitrary text chunks.
## When to use
Use for SSE, HTTP streaming, websocket generation, tool events, or partial responses.
## Workflow
Define event types and IDs; define ordering and terminal states; handle cancellation, timeout, disconnect, backpressure, and partial output; preserve request/response correlation; avoid assuming chunk boundaries.
## Verification
Test fragmented events, empty chunks, duplicate/reordered delivery where relevant, mid-stream tool calls, disconnects, cancellation, and finalization.
## Safety
Never leak hidden prompts, credentials, internal errors, or data from another request through a stream.
