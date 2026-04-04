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


MAX_CONTEXT_MESSAGES = 30  


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.conversations: dict[str, list[dict]] = {}
        self.known_tools: dict[str, set[str]] = {}  
        self.known_facts: dict[str, dict] = {}  
        self.call_history: dict[str, list[str]] = {}  
        self.client = AsyncOpenAI(
            base_url=BOTHUB_BASE_URL,
            api_key=BOTHUB_API_KEY,
        )

    def _extract_facts(self, context_id: str, text: str) -> None:
        """Extract key facts from tool results to prevent agent from forgetting them."""
        facts = self.known_facts.setdefault(context_id, {})
        try:
            data = json.loads(text) if text.strip().startswith(("{", "[")) else None
        except (json.JSONDecodeError, ValueError):
            data = None
        if not isinstance(data, dict):
            return

        if "name" in data and "reservations" in data:
            name = data.get("name", {})
            facts["user_name"] = f"{name.get('first', '')} {name.get('last', '')}".strip()
            facts["user_id"] = data.get("id", "")
            facts["membership"] = data.get("membership", "")
            facts["reservation_ids"] = data.get("reservations", [])
            facts["payment_methods"] = data.get("payment_methods", {})

        if "reservation_id" in data and "passengers" in data:
            rid = data["reservation_id"]
            passengers = data.get("passengers", [])
            facts[f"res_{rid}_cabin"] = data.get("cabin", "")
            facts[f"res_{rid}_passengers"] = [
                {
                    "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                    "dob": p.get("dob", ""),
                    "passenger_id": p.get("passenger_id", ""),
                }
                for p in passengers
            ]
            facts[f"res_{rid}_flights"] = data.get("flights", [])
            facts[f"res_{rid}_insurance"] = data.get("insurance", "none")
            facts[f"res_{rid}_status"] = data.get("status", "")

    def _get_facts_summary(self, context_id: str) -> str:
        """Build a concise summary of key facts (passengers, membership) for injection."""
        facts = self.known_facts.get(context_id, {})
        if not facts:
            return ""
        lines = ["KNOWN DATA (use this, do NOT ask user for any of it):"]
        # Only include essential data that LLM tends to forget
        for k, v in facts.items():
            if "_passengers" in k or k in ("user_name", "user_id", "membership"):
                if isinstance(v, (list, dict)):
                    lines.append(f"  {k}: {json.dumps(v, default=str)}")
                else:
                    lines.append(f"  {k}: {v}")
        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _check_duplicate_call(self, context_id: str, action: dict) -> bool:
        """Check if this exact tool call was already made. Returns True if duplicate."""
        history = self.call_history.setdefault(context_id, [])
        name = action.get("name", "")
        args = action.get("arguments", {})
        if name == "respond":
            return False  
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        if sig in history:
            return True
        history.append(sig)
        return False

    def _extract_tool_names(self, first_message: str) -> set[str]:
        """Extract tool names from the first message's tool definitions."""
        tools = {"respond"}
        try:
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
                    "Before each response, think step by step:\n"
                    "1. What does the user want?\n"
                    "2. What data do I need? → Call the tool to get it.\n"
                    "3. What policy rules apply? → Re-read the EXACT policy text for this situation.\n"
                    "4. Are ALL conditions met? → If yes, proceed. If no, deny citing the specific rule.\n\n"
                    "## POLICY COMPLIANCE (most important)\n"
                    "- The domain policy in the first message is ABSOLUTE LAW. Follow EVERY rule EXACTLY.\n"
                    "- VERIFY every condition with a tool call. Never assume or trust user claims.\n"
                    "- If ANY condition is not met, DENY and cite the specific rule.\n"
                    "- Enforce exact numbers: deadlines, amounts, limits, fees.\n"
                    "- If the user argues, repeats, or begs — still follow the policy. No exceptions.\n"
                    "- Do NOT invent rules. Only enforce what is explicitly stated in the policy.\n"
                    "- If the policy allows an action, DO it. Don't add extra conditions not in policy.\n"
                    "- Before every action, re-read the relevant policy section carefully.\n\n"
                    "## COMMON POLICY MISTAKES TO AVOID\n"
                    "Double-check these rules — they are tricky:\n\n"
                    "Compensation:\n"
                    "- NEVER proactively offer compensation. Only offer when user EXPLICITLY asks.\n"
                    "- Eligible: ONLY silver/gold members OR insured users OR business class.\n"
                    "- Regular members with no insurance in (basic) economy → NO compensation ever.\n"
                    "- Cancelled flights: $100 × number of passengers in the reservation.\n"
                    "- Delayed flights: $50 × number of passengers, BUT ONLY if user wants to "
                    "change/cancel the reservation. Offer AFTER completing the change/cancellation.\n"
                    "- No compensation for any other reason.\n\n"
                    "Cancellation:\n"
                    "- Allowed if: booked within 24hrs, OR airline cancelled, OR business class, "
                    "OR has insurance with covered reason (health/weather).\n"
                    "- If any flight already flown → cannot cancel → transfer to human.\n"
                    "- The API does NOT check rules — YOU must verify before calling cancel.\n\n"
                    "Modification:\n"
                    "- Basic economy flights CANNOT be modified (but cabin CAN be changed).\n"
                    "- IMPORTANT: After upgrading cabin from basic_economy to economy or business, "
                    "the NEW cabin's rules apply. So: upgrade cabin FIRST, then modify flights.\n"
                    "- Cannot change cabin if any flight already flown.\n"
                    "- Origin, destination, trip type CANNOT change. If user wants different destination, "
                    "the ONLY option is cancel + rebook (if eligible for cancellation).\n"
                    "- Cabin must be same across all flights in reservation.\n"
                    "- When user asks about cabin upgrade, ALWAYS search flights to get actual prices "
                    "for the new cabin and present the cost difference with real numbers.\n\n"
                    "Booking:\n"
                    "- Max 5 passengers. Payment: max 1 certificate, 1 credit card, 3 gift cards.\n"
                    "- All payment methods must be in user profile.\n"
                    "- Ask about travel insurance ($30/passenger).\n\n"
                    "Baggage (free checked bags — membership × cabin):\n"
                    "- Regular: basic_economy=0, economy=1, business=2\n"
                    "- Silver: basic_economy=1, economy=2, business=3\n"
                    "- Gold: basic_economy=2, economy=3, business=4\n"
                    "- Extra bags: $50 each. Don't add bags user doesn't need.\n\n"
                    "## TRANSFER TO HUMAN\n"
                    "- ONLY when the request CANNOT be handled with available tools.\n"
                    "- If any flight in reservation is ALREADY FLOWN (departure in the past) "
                    "and user wants to cancel → you MUST transfer to human agent.\n"
                    "- Do NOT transfer for things you can handle yourself.\n\n"
                    "## DESTINATION CHANGE → CANCEL + REBOOK\n"
                    "- If user wants to change destination (which is not allowed as modification), "
                    "and they are eligible for cancellation (insurance+covered reason, or business, etc.), "
                    "PROACTIVELY offer to cancel the current reservation and book a new one to the new destination. "
                    "Drive this workflow yourself — don't just list options passively.\n\n"
                    "## NEVER ASK THE USER FOR\n"
                    "- Reservation IDs, booking codes, confirmation numbers\n"
                    "- Airport codes — map city names yourself\n"
                    "- Internal system identifiers\n"
                    "- Date of birth, passenger IDs, or ANY data you already retrieved from the system\n"
                    "Instead: look up their account, then find the relevant item yourself.\n"
                    "IMPORTANT: When updating passenger info (name change etc.), USE the DOB and passenger_id "
                    "from the reservation details you already retrieved. Never ask the user for these.\n\n"
                    "## AIRPORT CODES (use these, never ask)\n"
                    "New York: JFK, Los Angeles: LAX, Chicago: ORD, San Francisco: SFO, "
                    "Miami: MIA, Dallas: DFW, Atlanta: ATL, Seattle: SEA, Boston: BOS, "
                    "Denver: DEN, Houston: IAH, Washington DC: DCA, Philadelphia: PHL, "
                    "Phoenix: PHX, Minneapolis: MSP, Detroit: DTW, Orlando: MCO, "
                    "Portland: PDX, Las Vegas: LAS, Salt Lake City: SLC, Tampa: TPA\n\n"
                    "## TOOL USAGE\n"
                    "- ALWAYS verify data with tools. Never rely on user-provided info alone.\n"
                    "- Call ONE tool at a time.\n"
                    "- Use exact parameter names and types from tool definitions.\n"
                    "- If user describes something without an ID, look up their account and find it.\n"
                    "- Only ask for info users would reasonably know (name, email, what they want).\n\n"
                    "## EFFICIENCY (critical — wasting steps can cause task failure)\n"
                    "- Once user confirms an action (says 'yes'), IMMEDIATELY call the API. No extra steps.\n"
                    "- Do NOT iterate through all reservations unless necessary. Ask user which one, "
                    "or deduce from their description (e.g. 'flight to Chicago' → find ORD reservation).\n"
                    "- If user wants to BOOK a new flight, don't look up existing reservations first — "
                    "ask for trip details directly.\n"
                    "- If user describes something (e.g. 'my flight to Chicago'), find the match yourself.\n"
                    "- If there's only one match, act on it directly.\n"
                    "- Minimize tool calls. Every extra step risks running out of turns.\n\n"
                    "## RESPONSE FORMAT\n"
                    "Output ONLY a raw JSON object. No markdown, no code fences, no extra text.\n"
                    "- Tool call: {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
                    "- User reply: {\"name\": \"respond\", \"arguments\": {\"content\": \"message\"}}"
                ),
            })
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })
            self.known_tools[context_id] = self._extract_tool_names(input_text)
            logger.info(f"Extracted tools: {self.known_tools[context_id]}")
            logger.info(f"First message (first 500): {input_text[:500]}")
        else:
            self._extract_facts(context_id, input_text)
            self.conversations[context_id].append({
                "role": "user",
                "content": input_text,
            })

        agent_turns = sum(1 for m in self.conversations[context_id] if m["role"] == "assistant")

        turn_count = len(self.conversations[context_id])
        if turn_count > 6 and turn_count % 10 == 0:
            self.conversations[context_id].append({
                "role": "system",
                "content": (
                    "REMINDER: Re-read the policy before responding. Key rules:\n"
                    "- Never proactively offer compensation. Only if user asks.\n"
                    "- Compensation: cancelled=$100×passengers, delayed=$50×passengers (only after change/cancel).\n"
                    "- Eligible for compensation: silver/gold OR insured OR business only.\n"
                    "- Cancel allowed if: within 24hrs OR airline cancelled OR business OR insured+covered reason.\n"
                    "- Basic economy: cannot modify flights, CAN change cabin.\n"
                    "- Before any booking action, list details and get user confirmation.\n"
                    "- Once user confirms, IMMEDIATELY call the API. Do not delay.\n"
                    "- Be efficient: don't iterate all reservations if you can deduce the right one."
                ),
            })

        if agent_turns >= 25:
            self.conversations[context_id].append({
                "role": "system",
                "content": (
                    f"⚠️ URGENCY: You have used {agent_turns} steps. You are running low on steps. "
                    "IMMEDIATELY execute pending actions. No more information gathering — act NOW. "
                    "Skip confirmations if you already have the data needed."
                ),
            })
        elif agent_turns >= 18:
            self.conversations[context_id].append({
                "role": "system",
                "content": (
                    f"EFFICIENCY WARNING: You have used {agent_turns} steps. Be very efficient. "
                    "Do not iterate through items — ask or deduce. "
                    "Combine steps where possible. Act quickly."
                ),
            })

        facts_summary = self._get_facts_summary(context_id)
        if facts_summary and agent_turns >= 3 and agent_turns % 5 == 0:
            self.conversations[context_id].append({
                "role": "system",
                "content": facts_summary,
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

        known = self.known_tools.get(context_id, set())
        if not was_fallback and known and action["name"] not in known:
            logger.warning(f"Unknown tool '{action['name']}', known: {known}. Treating as fallback.")
            was_fallback = True

        # Log duplicate calls but don't block them (LLM may have valid reasons)
        if not was_fallback:
            self._check_duplicate_call(context_id, action)

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
