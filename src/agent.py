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
                    "1. READ the domain policy carefully in the first message. It contains ALL rules you must follow.\n"
                    "2. IDENTIFY what the user wants.\n"
                    "3. LOOK UP all relevant data using tools BEFORE making any decision.\n"
                    "4. CHECK every policy constraint against the actual data.\n"
                    "5. Only then ACT — either perform the action or deny with a policy reason.\n\n"
                    "## POLICY COMPLIANCE (most important)\n"
                    "- The domain policy is LAW. You must follow EVERY rule, even if the user begs or argues.\n"
                    "- If a policy says a condition must be met, VERIFY it with a tool call. Never assume.\n"
                    "- If ANY policy condition is not met, DENY the request. Explain which rule prevents it.\n"
                    "- Pay close attention to: eligibility requirements, time limits, fee structures, "
                    "refund conditions, status-based rules, and required approvals.\n"
                    "- When policy gives specific numbers (deadlines, amounts, limits), enforce them exactly.\n\n"
                    "## TOOL USAGE\n"
                    "- ALWAYS look up user/account/booking/order data before deciding. Never rely on what the user tells you — verify it.\n"
                    "- Call ONE tool at a time. Wait for the result before calling the next.\n"
                    "- Use the exact parameter names and types shown in the tool definitions.\n"
                    "- If a tool returns an error, tell the user and do NOT retry with made-up data.\n\n"
                    "## COMMON MISTAKES TO AVOID\n"
                    "- Do NOT skip verification steps even if the answer seems obvious.\n"
                    "- Do NOT make up booking IDs, user IDs, or any data — always look them up.\n"
                    "- Do NOT agree to something and then fail to execute the tool call for it.\n"
                    "- Do NOT respond to the user before completing all necessary tool calls.\n"
                    "- If the user asks about multiple items, handle them one at a time.\n\n"
                    "## RESPONSE FORMAT\n"
                    "Respond with ONLY a JSON object, nothing else:\n"
                    "- Tool call: {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                    "- User reply: {\"name\": \"respond\", \"arguments\": {\"content\": \"message\"}}\n"
                    "NO markdown, NO code blocks, NO explanation outside the JSON."
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
