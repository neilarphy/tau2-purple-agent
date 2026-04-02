import json
import logging
import os
import re
from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger

logger = logging.getLogger(__name__)

BOTHUB_BASE_URL = os.environ.get("BOTHUB_BASE_URL", "https://bothub.chat/api/v1/")
BOTHUB_API_KEY = os.environ.get("BOTHUB_API_KEY", "")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o")


def extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    # Try JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return text.strip()


def parse_action(text: str) -> dict:
    """Parse LLM response into action dict. Always returns valid action."""
    raw = extract_json(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    logger.warning(f"Could not parse JSON, falling back to respond: {text[:200]}")
    return {"name": "respond", "arguments": {"content": text}}


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.conversations: dict[str, list[dict]] = {}
        self.client = AsyncOpenAI(
            base_url=BOTHUB_BASE_URL,
            api_key=BOTHUB_API_KEY,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        context_id = message.contextId or "default"

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )

        # Build conversation history
        if context_id not in self.conversations:
            self.conversations[context_id] = []

            # First message contains the full prompt with policy + tools + user messages
            self.conversations[context_id].append({
                "role": "system",
                "content": (
                    "You are a customer service agent. You MUST respond with ONLY valid JSON, "
                    "no markdown, no explanation, no extra text. Just the JSON object.\n\n"
                    "Format: {\"name\": \"tool_or_respond\", \"arguments\": {...}}\n\n"
                    "Use tools when needed, use \"respond\" to talk to the user."
                ),
            })
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })
        else:
            # Subsequent messages: tool results or user messages
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })

        # Call LLM
        try:
            response = await self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=self.conversations[context_id],
                temperature=0,
                max_tokens=2048,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            reply = json.dumps({"name": "respond", "arguments": {"content": "I apologize, I encountered an error. Could you please repeat your request?"}})

        # Save assistant reply to history
        self.conversations[context_id].append({
            "role": "assistant",
            "content": reply,
        })

        # Parse and return clean JSON
        action = parse_action(reply)
        result = json.dumps(action)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="response",
        )
