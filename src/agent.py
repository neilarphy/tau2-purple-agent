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


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = strip_thinking(text)
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


def parse_action(text: str) -> tuple[dict, bool]:
    """Parse LLM response into action dict. Returns (action, was_fallback)."""
    raw = extract_json(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed, False
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    logger.warning(f"Could not parse JSON, falling back to respond: {text[:200]}")
    return {"name": "respond", "arguments": {"content": text}}, True


MAX_CONTEXT_MESSAGES = 30  # system + first_user + up to 28 subsequent messages


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.conversations: dict[str, list[dict]] = {}
        self.known_tools: dict[str, set[str]] = {}  # context_id -> set of tool names
        self.client = AsyncOpenAI(
            base_url=BOTHUB_BASE_URL,
            api_key=BOTHUB_API_KEY,
        )

    def _extract_tool_names(self, first_message: str) -> set[str]:
        """Extract tool names from the first message's tool definitions."""
        tools = {"respond"}
        try:
            # Look for tool definitions in JSON format
            for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', first_message):
                tools.add(match.group(1))
        except Exception:
            pass
        return tools

    def _get_trimmed_messages(self, context_id: str) -> list[dict]:
        """Keep system + first user message + last N turns to prevent context overflow."""
        msgs = self.conversations[context_id]
        if len(msgs) <= MAX_CONTEXT_MESSAGES:
            return msgs
        # Always keep: system (0) + first user message (1), then most recent
        preserved = msgs[:2]
        preserved.append({
            "role": "user",
            "content": "[Earlier conversation messages omitted. Continue from here.]"
        })
        recent = msgs[-(MAX_CONTEXT_MESSAGES - 3):]
        return preserved + recent

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
                    "- If the user argues, repeats, or begs — still follow the policy. Do NOT make exceptions.\n"
                    "- Do NOT invent rules that aren't in the policy. Only enforce what is explicitly stated.\n"
                    "- If the policy allows an action, DO it. Don't refuse or add extra conditions not in policy.\n\n"
                    "## TRANSFER TO HUMAN RULES\n"
                    "- ONLY use transfer_to_human_agents when the policy EXPLICITLY requires it for this situation.\n"
                    "- If you CAN resolve it with available tools, you MUST do so — do NOT transfer.\n"
                    "- Never transfer because you're unsure — check the policy and data first.\n"
                    "- Common mistake: transferring for cancellations/refunds/changes you can handle with tools.\n\n"
                    "## NEVER ASK THE USER FOR\n"
                    "- Reservation IDs, booking codes, confirmation numbers, order numbers\n"
                    "- Airport codes (IATA codes) — map city names yourself\n"
                    "- Internal system identifiers, account IDs\n"
                    "- Technical details they wouldn't know\n"
                    "Instead: look up their account by name/email, then find the relevant item yourself.\n\n"
                    "## AIRPORT CODE MAPPING (use these, never ask the user)\n"
                    "New York: JFK, Los Angeles: LAX, Chicago: ORD, San Francisco: SFO, "
                    "Miami: MIA, Dallas: DFW, Atlanta: ATL, Seattle: SEA, Boston: BOS, "
                    "Denver: DEN, Houston: IAH, Washington DC: DCA, Philadelphia: PHL, "
                    "Phoenix: PHX, Minneapolis: MSP, Detroit: DTW, Orlando: MCO, "
                    "Portland: PDX, Las Vegas: LAS, Salt Lake City: SLC, Tampa: TPA, "
                    "San Diego: SAN, Nashville: BNA, Austin: AUS, Charlotte: CLT, "
                    "London: LHR, Paris: CDG, Tokyo: NRT\n\n"
                    "## TOOL USAGE\n"
                    "- ALWAYS verify data with tools. Never rely on user-provided info alone.\n"
                    "- Call ONE tool at a time.\n"
                    "- Use exact parameter names and types from tool definitions.\n"
                    "- If the user mentions a specific ID, look it up directly.\n"
                    "- If the user describes something without an ID (e.g. 'my canceled flight'), "
                    "look up their account and find the matching item yourself.\n"
                    "- Only ask the user for info they would reasonably know (name, email, what they want).\n\n"
                    "## EFFICIENCY\n"
                    "- If a user describes something (e.g. 'my flight to Chicago'), look up their info "
                    "and find the matching item. Don't list all items and ask which one.\n"
                    "- If there's only one matching item, act on it directly.\n"
                    "- Minimize the number of tool calls by deducing from context.\n"
                    "- Don't ask clarifying questions if you can figure it out from the data.\n\n"
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
            # Extract tool names from first message for validation
            self.known_tools[context_id] = self._extract_tool_names(input_text)
            logger.info(f"Extracted tools: {self.known_tools[context_id]}")
        else:
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })

        try:
            messages = self._get_trimmed_messages(context_id)
            logger.info(f"Calling LLM: model={AGENT_MODEL}, base_url={BOTHUB_BASE_URL}, msgs={len(messages)} (full={len(self.conversations[context_id])})")
            response = await self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
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

        action, was_fallback = parse_action(reply)

        # Validate tool name if we know the available tools
        known = self.known_tools.get(context_id, set())
        if not was_fallback and known and action["name"] not in known:
            logger.warning(f"Unknown tool '{action['name']}', known: {known}. Treating as fallback.")
            was_fallback = True

        # Retry once if JSON parsing failed or tool name invalid
        if was_fallback:
            logger.warning("Retrying with correction prompt")
            self.conversations[context_id].append({
                "role": "user",
                "content": (
                    "Your previous response was not valid. "
                    "Respond with ONLY a raw JSON object, no other text:\n"
                    '{"name": "tool_name", "arguments": {...}} or '
                    '{"name": "respond", "arguments": {"content": "..."}}'
                ),
            })
            try:
                messages = self._get_trimmed_messages(context_id)
                response = await self.client.chat.completions.create(
                    model=AGENT_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=4096,
                )
                retry_reply = response.choices[0].message.content or ""
                logger.info(f"Retry LLM reply (first 300): {retry_reply[:300]}")
                self.conversations[context_id].append({
                    "role": "assistant",
                    "content": retry_reply,
                })
                retry_action, retry_fallback = parse_action(retry_reply)
                if not retry_fallback:
                    action = retry_action
            except Exception as e:
                logger.error(f"Retry LLM call failed: {e}")

        result = json.dumps(action)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result))],
            name="response",
        )
