"""
og_tools_plugin.py — OG (Inf Super) architecture as an agent 2.1 plugin.

Ports the three-component inf_super architecture from sari-agent-1.0:
  1. Semantic Associative Learner  — runs before each new user turn, reads the
     current view + state, extracts a new semantic memory fact, and injects a
     context-specific "recall" string into the system prompt.
  2. Episodic Associative Learner  — runs after each final agent response (no
     tool call), synthesises a structured episodic reflection and persists it.
  3. create_plan tool              — lets the agent explicitly record its
     planning notes (main_goal / sub_goal / key_info / status / checklist),
     mirroring the old 'notes' field in the inf_super JSON output.

All navigation, manipulation, and perception tools are forwarded to
agent_tools3._dispatch, identical to agent_tools3_plugin.
"""

import json
import re

from agent_tools3 import (
    NAVIGATION_TOOLS,
    MANIPULATION_TOOLS,
    PERCEPTION_TOOLS,
    MEMORY_TOOLS,
    _dispatch,
    load_semantic_memory,
    _save_semantic_memory,
    load_episodic_memory,
    save_episodic_memory,
)
from reference import (
    BASE_SEMANTIC_MEMORY,
    SYSTEM_INSTRUCTION_INF_SUPER_ASSOCIATIVE_SEMANTIC,
    SYSTEM_INSTRUCTION_INF_SUPER_ASSOCIATIVE_EPISODIC,
)
from utils.utils import AgentPlugin, AgentContext

# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

_ALL_TOOLS = NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS + MEMORY_TOOLS
_ALL_TOOL_NAMES = {t["name"] for t in _ALL_TOOLS}

_PLAN_TOOL = {
    "type": "function",
    "name": "create_plan",
    "description": (
        "Record your planning state for the current step. Call this at the start "
        "of each reasoning cycle to explicitly track goals, observations, progress, "
        "and what remains to be done. Mirrors the 'notes' field from the original "
        "inf_super architecture."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "main_goal": {
                "type": "string",
                "description": "The overall task objective.",
            },
            "sub_goal": {
                "type": "string",
                "description": "The immediate sub-goal for the current step.",
            },
            "key_info": {
                "type": "string",
                "description": "Key observations from the current view or state.",
            },
            "status": {
                "type": "string",
                "description": "Progress assessment towards the goals.",
            },
            "checklist": {
                "type": "string",
                "description": "Items to find or actions to complete, e.g. '[X] item1, [ ] item2'.",
            },
        },
        "required": ["main_goal", "sub_goal", "key_info", "status", "checklist"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_OG_BASE_PROMPT = (
    "You are an Embodied AI Agent operating within a 3D convenience store simulation. "
    "Your task is to navigate, locate, and manipulate (grab, pick) specific target "
    "item(s) specified by the User.\n\n"
    "**Capabilities:**\n"
    "1. **Perceive:** Call get_current_view(), get_agent_state(), and get_hand_state() "
    "to observe the environment.\n"
    "2. **Navigate:** Use movement and camera tools to explore the store.\n"
    "3. **Manipulate:** Use hand extension and grip tools to pick up items.\n"
    "4. **Memory:** Semantic Memory (facts about the environment) and Episodic Memory "
    "(past session experience) are injected into your context automatically each turn. "
    "You may also store new facts with store_semantic_memory().\n"
    "5. **Planning:** Call create_plan() at the start of each reasoning cycle to record "
    "your main goal, sub-goal, key observations, status, and checklist — mirroring the "
    "structured planning loop from the original inf_super architecture.\n\n"
    "**Required reasoning process at each step:**\n"
    "1. Call create_plan() — re-state the main goal, define the sub-goal, record key "
    "info from the current observation, assess progress, update the checklist.\n"
    "2. Observe — call get_current_view() to see what is in front of you.\n"
    "3. Consult memory — use the Semantic Recall and Episodic Memory injected into "
    "your context to inform your next action.\n"
    "4. Act — execute the appropriate navigation, manipulation, or perception tools.\n\n"
    "**Critical procedures (follow strictly):**\n"
    "1. **Centering:** When a target item is visible, keep it centered in your view "
    "as you approach. Pan left/right to correct drift.\n"
    "2. **Approach:** Move forward while the item stays centered and gets larger.\n"
    "3. **Grabbing:** When the item dominates the central view, extend your right hand "
    "forward then call grip_right().\n"
    "4. **Obstacle avoidance:** Visually check for walls and shelves before moving "
    "forward; use move_backward() to create space in tight spots.\n"
    "5. **Completion:** Stop issuing tool calls only when you have successfully grabbed "
    "the target item and confirmed via get_hand_state().\n"
)


def _build_system_prompt(recall: str = "") -> str:
    """Compose the full system prompt with current memories injected."""
    facts = load_semantic_memory()
    episodic = load_episodic_memory()

    parts = [_OG_BASE_PROMPT, "\n\n", BASE_SEMANTIC_MEMORY]

    if facts:
        facts_str = "\n".join(f"  - {f}" for f in facts)
        parts.append(f"\n\n## LEARNED SEMANTIC MEMORY\n{facts_str}")

    if episodic:
        parts.append(
            f"\n\n## EPISODIC MEMORY (last session)\n"
            f"- Summary: {episodic.get('dense_summary', 'N/A')}\n"
            f"- Surprise: {episodic.get('surprise', 'N/A')}\n"
            f"- What worked: {episodic.get('what_worked', 'N/A')}\n"
            f"- What to avoid: {episodic.get('what_to_avoid', 'N/A')}"
        )

    if recall:
        parts.append(f"\n\n## SEMANTIC RECALL (current context)\n{recall}")

    return "".join(parts)


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, tolerating single-quoted keys."""
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    raw = match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Plugin class
# ---------------------------------------------------------------------------

class OGToolsPlugin(AgentPlugin):
    PLUGIN_NAME = "OG Tools (Inf Super)"
    AGENT_TOOLS = []  # unused; overridden by assemble_tool_dicts

    def __init__(self, context: AgentContext):
        super().__init__(context)
        self._dynamic_system_prompt = _build_system_prompt()
        self._turn_count = 0

    @property
    def SYSTEM_PROMPT(self) -> str:
        return self._dynamic_system_prompt

    def assemble_tool_dicts(self) -> list[dict]:
        return list(_ALL_TOOLS) + [_PLAN_TOOL]

    def __getattr__(self, name: str):
        if name in _ALL_TOOL_NAMES:
            async def handler(args: dict) -> dict:
                return await _dispatch(name, args)
            return handler
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    async def create_plan(self, args: dict) -> dict:
        self.ctx.log(
            f"[Plan t={self._turn_count}] "
            f"goal={args['main_goal']!r} | sub={args['sub_goal']!r} | "
            f"status={args['status']!r}"
        )
        return {"recorded": True, **args}

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def on_turn_start(self, context: AgentContext) -> None:
        """Run the Semantic Associative Learner before each new user turn."""
        self._turn_count += 1
        recall = await self._run_semantic_learner(context)
        self._dynamic_system_prompt = _build_system_prompt(recall=recall)
        context.reload_system_prompt()

    async def on_turn_end(self, context: AgentContext) -> None:
        """Run the Episodic Associative Learner after the agent's final response."""
        await self._run_episodic_learner(context)
        self._dynamic_system_prompt = _build_system_prompt()
        context.reload_system_prompt()

    # ------------------------------------------------------------------
    # Semantic learner
    # ------------------------------------------------------------------

    async def _run_semantic_learner(self, context: AgentContext) -> str:
        """
        Parallel semantic LLM call (inf_super architecture).
        Gets the current screenshot + state, calls the semantic learner model,
        stores a new memory fact, and returns the recall string for injection.
        """
        try:
            screenshot = await _dispatch("get_current_view", {})
            agent_state = await _dispatch("get_agent_state", {})
            hand_state = await _dispatch("get_hand_state", {})

            facts = load_semantic_memory()
            semantic_log = BASE_SEMANTIC_MEMORY
            if facts:
                semantic_log += "\n\nLEARNED FACTS:\n" + "\n".join(f"- {f}" for f in facts)

            user_content = [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot['image_base64']}",
                },
                {
                    "type": "input_text",
                    "text": (
                        f"Timestep: {self._turn_count}\n"
                        f"Agent State: {json.dumps(agent_state)}\n"
                        f"Hand State: {json.dumps(hand_state)}\n"
                        f"Semantic Memory Log:\n{semantic_log}"
                    ),
                },
            ]

            response = await context.client.responses.create(
                model=context.model_name,
                instructions=SYSTEM_INSTRUCTION_INF_SUPER_ASSOCIATIVE_SEMANTIC,
                input=[{"role": "user", "content": user_content}],
                max_output_tokens=512,
            )

            text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for part in item.content:
                        if hasattr(part, "text"):
                            text += part.text

            data = _extract_json(text)
            if data:
                new_fact = data.get("new_semantic_memory", "").strip()
                recall = data.get("recall", "").strip()
                if new_fact:
                    facts = load_semantic_memory()
                    facts.append(f"[t={self._turn_count}] {new_fact}")
                    _save_semantic_memory(facts)
                    context.log(f"[Semantic Learner] +fact: {new_fact[:100]}")
                if recall:
                    context.log(f"[Semantic Learner] recall: {recall[:100]}")
                return recall

        except Exception as e:
            context.log(f"[Semantic Learner] skipped ({e})")

        return ""

    # ------------------------------------------------------------------
    # Episodic learner
    # ------------------------------------------------------------------

    async def _run_episodic_learner(self, context: AgentContext) -> None:
        """
        Post-turn episodic LLM call (inf_super architecture).
        Synthesises a structured reflection from the last exchange and persists it.
        """
        try:
            exchange_parts = []
            for msg in context.messages[-12:]:
                role = msg.get("role") or msg.get("type", "")
                content = (
                    msg.get("content")
                    or msg.get("output")
                    or msg.get("arguments", "")
                )
                if isinstance(content, str) and content.strip():
                    exchange_parts.append(f"{role}: {content.strip()}")
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "input_text":
                            exchange_parts.append(f"{role}: {part['text'].strip()}")

            if not exchange_parts:
                return

            response = await context.client.responses.create(
                model=context.model_name,
                instructions=SYSTEM_INSTRUCTION_INF_SUPER_ASSOCIATIVE_EPISODIC,
                input=[{"role": "user", "content": "\n\n".join(exchange_parts)}],
                max_output_tokens=512,
            )

            text = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for part in item.content:
                        if hasattr(part, "text"):
                            text += part.text

            data = _extract_json(text)
            if data:
                save_episodic_memory(data)
                context.log(
                    f"[Episodic Learner] saved: {data.get('dense_summary', '')[:100]}"
                )

        except Exception as e:
            context.log(f"[Episodic Learner] skipped ({e})")


def setup(context: AgentContext) -> AgentPlugin:
    return OGToolsPlugin(context)
