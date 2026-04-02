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

BOTHUB_BASE_URL = os.environ.get("BOTHUB_BASE_URL", "https://bothub.chat/api/v2/openai/v1")
BOTHUB_API_KEY = os.environ.get("BOTHUB_API_KEY", "")
AGENT_MODEL = os.environ.get("AGENT_MODEL", os.environ.get("AGENT_LLM", "gpt-4o"))


def extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()

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
        context_id = getattr(message, 'context_id', None) or getattr(message, 'contextId', None) or "default"

        await updater.update_status(
            TaskState.working, new_agent_text_message("Processing...")
        )

        if context_id not in self.conversations:
            self.conversations[context_id] = []

            self.conversations[context_id].append({
                "role": "system",
                "content": (
                    "You are a precise customer service agent. You will receive domain policies, "
                    "available tools, and user requests.\n\n"
                    "CRITICAL RULES:\n"
                    "1. ALWAYS follow the domain policy EXACTLY. Never make exceptions to policy rules.\n"
                    "2. Before taking any action, verify you have all required information by using tools.\n"
                    "3. If the user asks for something that violates policy, politely DENY the request "
                    "and explain why, citing the specific policy.\n"
                    "4. Use tools to look up information before making decisions. NEVER guess or assume data.\n"
                    "5. When a policy says you need to check something, ALWAYS check it with the appropriate tool.\n"
                    "6. Complete all required steps in the correct order as specified by the policy.\n\n"
                    "RESPONSE FORMAT - You MUST respond with ONLY a valid JSON object, no other text:\n"
                    "- To call a tool: {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                    "- To respond to user: {\"name\": \"respond\", \"arguments\": {\"content\": \"your message\"}}\n\n"
                    "IMPORTANT: Output ONLY the JSON object. No markdown, no code blocks, no explanation."
                ),
            })
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })
        else:
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })

        try:
            logger.info(f"Calling LLM: model={AGENT_MODEL}, base_url={BOTHUB_BASE_URL}, msgs={len(self.conversations[context_id])}")
            response = await self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=self.conversations[context_id],
                temperature=0,
                max_tokens=4096,
            )
            reply = response.choices[0].message.content or ""
            logger.info(f"LLM reply (first 300): {reply[:300]}")
        except Exception as e:
            logger.error(f"LLM call FAILED: {type(e).__name__}: {e}")
            reply = json.dumps({"name": "respond", "arguments": {"content": "I apologize, I encountered an error. Could you please repeat your request?"}})

        self.conversations[context_id].append({
            "role": "assistant",
            "content": reply,
        })

        action = parse_action(reply)
        result = json.dumps(action)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="response",
        )
