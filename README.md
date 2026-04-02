# tau2-purple-agent

Purple agent for the [tau2-bench](https://github.com/RDI-Foundation/tau2-agentbeats) benchmark on [AgentBeats](https://agentbeats.dev).

## How it works

This agent acts as a customer service AI that:
- Receives domain policies and available tools from the green agent
- Uses an LLM (via OpenAI-compatible API) to reason about customer requests
- Returns tool calls or direct responses in JSON format
- Maintains conversation history per task context

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BOTHUB_API_KEY` | Yes | API key for the LLM provider |
| `BOTHUB_BASE_URL` | No | API base URL (default: `https://bothub.chat/api/v1/`) |
| `AGENT_MODEL` | No | Model to use (default: `gpt-4o`) |

## Local development

```bash
uv sync
BOTHUB_API_KEY=your-key uv run src/server.py --host 127.0.0.1 --port 9009
```

## Docker

```bash
docker build -t tau2-purple-agent .
docker run -p 9009:9009 -e BOTHUB_API_KEY=your-key tau2-purple-agent
```
