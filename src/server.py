import argparse
import logging
import os

import uvicorn

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", default=None)
    args = parser.parse_args()

    url = args.card_url or f"http://{args.host}:{args.port}/"

    skill = AgentSkill(
        id="tau2-customer-service",
        name="Customer Service Agent",
        description="Handles customer service tasks across airline, retail, and telecom domains using tool calls and policy-aware reasoning.",
        tags=["customer-service", "tool-use", "tau2"],
        examples=["Help me cancel my flight booking", "I want to return a product"],
    )

    agent_card = AgentCard(
        name="tau2-purple-agent",
        description="Purple agent for tau2-bench: an LLM-powered customer service agent that follows domain policies and uses available tools to resolve customer requests.",
        url=url,
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities={"streaming": False},
        skills=[skill],
    )

    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(), task_store=task_store
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
