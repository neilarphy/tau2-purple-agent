# tau2-purple-agent

Purple agent for the [tau2-bench](https://github.com/RDI-Foundation/tau2-agentbeats) benchmark on [AgentBeats](https://agentbeats.dev).

## Approach

This agent implements a **two-stage reasoning pipeline** for policy-compliant customer service:

1. **Plan stage** — hidden chain-of-thought call: the model reasons over the current conversation state, retrieved data, and applicable policy rules before deciding on an action.
2. **Execute stage** — the model produces a single structured JSON action (tool call or user response) based on its reasoning.

Key design decisions:
- **Full conversation history** maintained per task context with sliding-window truncation
- **Policy-first system prompt** derived from iterative failure analysis across all 50 airline tasks, covering: social engineering resistance, cancellation eligibility, same-cabin enforcement, one-certificate-per-reservation limits, bulk vs. targeted reservation lookups, and multi-step request sequencing
- **Robust JSON parsing** with automatic retry on malformed output

## Model

Uses **Gemini** via an OpenAI-compatible API endpoint. The agent communicates through the OpenAI Python SDK, making it easy to swap underlying models.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | Yes | API key for the LLM provider |
| `LLM_BASE_URL` | No | OpenAI-compatible base URL of the LLM provider |
| `AGENT_MODEL` | No | Model name (e.g. `gemini-2.0-flash`) |

## Local development

```bash
uv sync
LLM_API_KEY=your-key LLM_BASE_URL=https://your-provider/v1 uv run src/server.py --host 127.0.0.1 --port 9009
```

## Docker

```bash
docker build -t tau2-purple-agent .
docker run -p 9009:9009 \
  -e LLM_API_KEY=your-key \
  -e LLM_BASE_URL=https://your-provider/v1 \
  -e AGENT_MODEL=gemini-2.0-flash \
  tau2-purple-agent
```
