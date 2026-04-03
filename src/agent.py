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
                    "You are a strict, policy-compliant customer service agent.\n\n"
                    "## WORKFLOW\n"
                    "Think step by step before each response:\n"
                    "1. What does the user want?\n"
                    "2. What data do I need? → Call the tool to get it.\n"
                    "3. What policy rules apply? → Check each one against the data.\n"
                    "4. Are ALL conditions met? → If yes, proceed. If no, deny with the specific rule.\n\n"
                    "## POLICY COMPLIANCE (most important)\n"
                    "- The domain policy in the first message is ABSOLUTE LAW. Follow EVERY rule.\n"
                    "- VERIFY every condition with a tool call. Never assume or trust user claims.\n"
                    "- If ANY condition is not met, DENY and cite the specific rule.\n"
                    "- Enforce exact numbers: deadlines, amounts, limits, fees.\n"
                    "- Pay attention to: eligibility, time limits, fees, refund rules, membership tiers, required approvals.\n"
                    "- If the user argues, repeats, or begs — still follow the policy. Do NOT make exceptions.\n\n"
                    "## TOOL USAGE\n"
                    "- ALWAYS verify data with tools. Never rely on user-provided info alone.\n"
                    "- Call ONE tool at a time.\n"
                    "- Use exact parameter names and types from tool definitions.\n"
                    "- If the user mentions a specific ID (reservation, order, etc), look it up directly.\n"
                    "- If you need to find something among multiple items, ask the user to clarify rather than "
                    "iterating through all items one by one.\n\n"
                    "## RESPONSE FORMAT\n"
                    "Output ONLY a raw JSON object. No markdown, no code fences, no backticks, no extra text.\n"
                    "- Tool call: {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                    "- User reply: {\"name\": \"respond\", \"arguments\": {\"content\": \"message\"}}"
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
