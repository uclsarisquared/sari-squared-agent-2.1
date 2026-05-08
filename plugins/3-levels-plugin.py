"""
3-levels-plugin.py — Three-level sub-agent orchestration for long-horizon tasks.

Architecture:
  Level 0 — Task Planner   : owns memory, spawns Level 1, no env tools
  Level 1 — Item Handler   : spawns Level 2, no env tools, no memory hooks
  Level 2 — Cycle Executor : has all env tools, calls REPORT_TO_PARENT, no SPAWN_SUBAGENT

A single ThreeLevelsPlugin class adapts its tool set, system prompt, and lifecycle
hooks based on self.depth. Each spawned child gets a fresh plugin instance at depth+1,
avoiding the shared-instance bug in the original inherit_plugins approach.
"""

import asyncio
import copy
import json
import re

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import TabbedContent, TabPane

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
from utils.llm_streaming import LLMResponse
from utils.tui_widgets import UserPrompt
from utils.utils import AgentContext, AgentPlugin, ToolDefinition

# ---------------------------------------------------------------------------
# System prompts per level
# ---------------------------------------------------------------------------

_LEVEL_0_BASE = (
    "You are a Task Planner Embodied AI Orchestrator for a 3D convenience store task. "
    "You do NOT interact with the environment directly. "
    "You delegate all perception and action to sub-agents via SPAWN_SUBAGENT.\n\n"
    "**Your role:**\n"
    "- Maintain the full item list and track what has been grabbed.\n"
    "- Inject current semantic and episodic memory into each sub-agent's input_prompt.\n"
    "- Interpret sub-agent reports and decide the next sub-goal.\n"
    "- Update your understanding of the environment from each report's context field.\n\n"
    "**Decision loop:**\n"
    "1. Read the last sub-agent report.\n"
    "2. Assess progress — did it succeed? What did it observe?\n"
    "3. Decide the next sub-goal.\n"
    "4. Spawn the next Level 1 sub-agent with fresh memory context in input_prompt.\n"
    "5. Repeat until all items are grabbed.\n\n"
    "**input_prompt format when spawning Level 1:**\n"
    "  Main task: {full task description}\n"
    "  Target item: {current item}\n"
    "  Semantic memory: {relevant facts from BASE_SEMANTIC_MEMORY and learned facts}\n"
    "  Episodic memory: {dense_summary / what_worked / what_to_avoid}\n"
    "  Last known state: {agent position, visible scene, progress so far}\n"
    "  Your sub-goal: {specific goal for this Level 1 instance}\n"
)

LEVEL_1_SYSTEM_PROMPT = (
    "You are an Item Handler sub-agent coordinating item retrieval inside a 3D convenience store. "
    "You have been assigned ONE target item. Do not deviate.\n\n"
    "**How you act:**\n"
    "You do NOT interact with the environment directly. You delegate every single action "
    "to a Level 2 step executor sub-agent via SPAWN_SUBAGENT. Each Level 2 instance performs "
    "exactly ONE action (e.g. move_forward once, or pan_left once) then reports back and dies. "
    "You are the loop — you decide the next action after reading each report.\n\n"
    "**Decision loop:**\n"
    "1. Spawn a Level 2 step executor with the current state and the single action you want it to take.\n"
    "2. Read the structured report it returns.\n"
    "3. Assess progress from the report fields: agent position, visible objects, target state, obstacles.\n"
    "4. Decide the next single action.\n"
    "5. Repeat until the item is gripped (rightGrippedState: true, rightHoveredObject matches target) "
    "or declared unreachable.\n"
    "6. Call REPORT_TO_PARENT with the final result.\n\n"
    "**input_prompt format when spawning Level 2:**\n"
    "  Target item: {item}\n"
    "  Last known state: {copy the full context block from the previous report verbatim}\n"
    "  Action to perform: {exactly one tool name and why, e.g. 'pan_left — target is right of center'}\n\n"
    "**Interpreting step reports:**\n"
    "- visible_objects: list of named objects and their screen position — use this to decide pan/move direction.\n"
    "- target_state: absent/visible-far/visible-mid/visible-close/gripped — tracks approach progress.\n"
    "- agent_position / agent_rotation: use to track where the agent is in the store.\n"
    "- rightGrippedState: True means the right hand successfully gripped an item.\n"
    "- rightHoveredObject: confirm it matches the target before declaring success.\n"
    "- obstacle: if present, next action must be move_backward before anything else.\n"
    "- next_hint: Level 2's suggested next action — consider it but make your own decision.\n"
)

LEVEL_2_SYSTEM_PROMPT = (
    "You are a Step Executor sub-agent operating within a 3D convenience store simulation. "
    "You perform exactly ONE action per invocation, then report and stop.\n\n"
    "**Your fixed sequence — follow it exactly, no exceptions:**\n"
    "1. Call get_current_view — observe the scene before acting.\n"
    "2. Call get_agent_state — record your position and rotation.\n"
    "3. Call get_hand_state — record grip and hover state.\n"
    "4. Execute the single action specified in your input_prompt. "
    "Do NOT substitute a different action. Do NOT call more than one action tool.\n"
    "5. Call get_current_view again — observe the result of your action.\n"
    "6. Call get_hand_state again if your action involved the hand (extend, grip, release, etc.).\n"
    "7. Call REPORT_TO_PARENT — this is mandatory and terminal. "
    "Do NOT call any further tools after it.\n\n"
    "**Available tool categories:**\n"
    "- Navigation: move_forward, move_backward, move_left, move_right, pan_left, pan_right, "
    "tilt_up, tilt_down, transform_agent\n"
    "- Manipulation: extend_right_hand_forward, pull_right_hand_backward, raise_right_hand, "
    "lower_right_hand, rotate_right_hand_clockwise, rotate_right_hand_counterclockwise, "
    "grip_right, release_right, transform_hands\n"
    "- Perception: get_current_view, get_scene_json, get_agent_state, get_hand_state\n\n"
    "**REPORT_TO_PARENT — context field format (fill every field, no omissions):**\n"
    "Use this exact template. Do NOT use vague language like 'partially visible' or 'nearby'.\n\n"
    "  agent_position: x={x}, y={y}, z={z}\n"
    "  agent_rotation: x={x}, y={y}, z={z}\n"
    "  action_taken: {exact tool name called}\n"
    "  visible_objects: ({name}, screen_pos={left|center|right}, dist={far|mid|close}); ...\n"
    "  target_state: {absent | visible-far | visible-mid | visible-close | gripped}\n"
    "  target_screen_pos: {left|center|right} at {top|middle|bottom}, approx {N}% of screen height\n"
    "  rightHoveredObject: {object name or null}\n"
    "  rightGrippedState: {true|false}\n"
    "  obstacle: {none | shelf at {left|center|right}, wall ahead, etc.}\n"
    "  next_hint: {your suggested next action and one-line reason, "
    "e.g. 'pan_left — target is right of center at 40% height'}\n\n"
    "**Rules:**\n"
    "1. Never skip the pre-action get_current_view or post-action get_current_view.\n"
    "2. Never call more than one action tool per invocation.\n"
    "3. Never write vague descriptions — use the template fields exactly.\n"
    "4. If the specified action would clearly cause a collision (obstacle directly ahead), "
    "still report — set success=False, obstacle field filled, next_hint='move_backward'.\n"
    "5. Unit limits — NEVER exceed these per call:\n"
    "   - target_state=absent        : move/strafe max 3 units, pan max 3 units\n"
    "   - target_state=visible-far   : move/strafe max 3 units, pan max 2 units\n"
    "   - target_state=visible-mid   : move/strafe max 2 units, pan max 1 unit\n"
    "   - target_state=visible-close : move/strafe max 1 unit,  pan max 1 unit\n"
    "   When in doubt, use fewer units. Small steps prevent overshooting.\n"
    "6. Call get_current_view exactly twice per invocation — once at step 1 (pre-action) "
    "and once at step 5 (post-action). Never call it twice in a row. "
    "Calling it consecutively will be blocked by the system and waste your turn.\n"
)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_SPAWN_SUBAGENT_TOOL = ToolDefinition(
    name="SPAWN_SUBAGENT",
    description=(
        "Spawns a sub-agent with a prompt and goal. "
        "The agent reasons and calls tools in sequence, stopping only when the goal "
        "condition is met or no further progress is possible. "
        "Be as detailed as possible, especially with the end goal and memory context."
    ),
    input_arguments={
        "input_prompt": {
            "type": "string",
            "description": (
                "Input prompt provided to the sub-agent, including memory context "
                "and last known state."
            ),
        },
        "goal_prompt": {
            "type": "string",
            "description": "Goal condition the sub-agent uses to determine when it is done.",
        },
        "subagent_id": {
            "type": "string",
            "description": (
                "Compact unique identifier for this sub-agent. No spaces. "
                "Never reuse an existing ID."
            ),
        },
    },
    required_arguments=["input_prompt", "goal_prompt", "subagent_id"],
)

_REPORT_TO_PARENT_TOOL = ToolDefinition(
    name="REPORT_TO_PARENT",
    description=(
        "ONLY CALL THIS TOOL IF YOU ARE A SUB-AGENT. "
        "Call when your assigned goal is complete or cannot be completed. "
        "Sends a structured summary back to the parent agent. "
        "This is a terminal action — do not call any other tool after it."
    ),
    input_arguments={
        "result": {
            "type": "string",
            "description": "What you found or accomplished.",
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "How certain you are that the goal is satisfied.",
        },
        "success": {
            "type": "boolean",
            "description": "Whether the goal was completed successfully.",
        },
        "context": {
            "type": "string",
            "description": (
                "Structured state block using the required template: agent_position, "
                "agent_rotation, action_taken, visible_objects, target_state, "
                "target_screen_pos, rightHoveredObject, rightGrippedState, obstacle. "
                "Every field is mandatory. No vague language."
            ),
        },
        "next_hint": {
            "type": "string",
            "description": (
                "Your suggested next action for the parent agent, with a one-line reason. "
                "Example: 'pan_left — target is right of center at 40% screen height'. "
                "The parent may ignore this but it must be provided."
            ),
        },
        "subagent_id": {
            "type": "string",
            "description": "Your unique sub-agent ID.",
        },
    },
    required_arguments=["result", "confidence", "success", "context", "next_hint", "subagent_id"],
)

# ---------------------------------------------------------------------------
# Env tool name set (for __getattr__ dispatch at Level 2)
# ---------------------------------------------------------------------------

_LEVEL_2_ENV_TOOLS = [
    t for t in NAVIGATION_TOOLS + MANIPULATION_TOOLS + PERCEPTION_TOOLS
    if t["name"] != "reset_environment"
]

_ENV_TOOL_NAMES = {t["name"] for t in _LEVEL_2_ENV_TOOLS}

# ---------------------------------------------------------------------------
# Subagent tab pane UI
# ---------------------------------------------------------------------------

class _SubagentPane(VerticalScroll):
    def __init__(self, prompt: str, ctx: AgentContext):
        self._prompt = prompt
        self._ctx = ctx
        super().__init__()

    def compose(self) -> ComposeResult:
        yield UserPrompt(self._prompt)
        yield LLMResponse(self._prompt, self._ctx)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
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


def _build_level0_prompt(facts: list, episodic: dict, recall: str = "") -> str:
    parts = [_LEVEL_0_BASE, "\n\n", BASE_SEMANTIC_MEMORY]
    if facts:
        parts.append("\n\n## LEARNED SEMANTIC MEMORY\n" + "\n".join(f"  - {f}" for f in facts))
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

# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class ThreeLevelsPlugin(AgentPlugin):
    PLUGIN_NAME = "Three Levels"
    AGENT_TOOLS = []  # unused — overridden by assemble_tool_dicts

    def __init__(self, context: AgentContext, depth: int = 0):
        # Set before super().__init__() so __getattr__ is safe immediately
        self.depth = depth
        self._subagent_count = 0
        self._finished_tracker: dict[str, asyncio.Event] = {}
        self._results: dict[str, dict] = {}
        self._turn_count = 0
        self._last_tool: str | None = None
        # Wired by parent at spawn time — gives REPORT_TO_PARENT its signal target
        self._parent_event: asyncio.Event | None = None
        self._parent_results: dict | None = None
        # Dynamic system prompt for Level 0 (updated each turn with fresh memories)
        self._dynamic_system_prompt = _build_level0_prompt(
            load_semantic_memory(), load_episodic_memory()
        )
        super().__init__(context)

    # ------------------------------------------------------------------
    # Plugin interface
    # ------------------------------------------------------------------

    @property
    def SYSTEM_PROMPT(self) -> str:
        if self.depth == 0:
            return self._dynamic_system_prompt
        elif self.depth == 1:
            return LEVEL_1_SYSTEM_PROMPT
        else:
            return LEVEL_2_SYSTEM_PROMPT

    def assemble_tool_dicts(self) -> list[dict]:
        tools = []
        if self.depth < 2:
            tools.append(_SPAWN_SUBAGENT_TOOL.to_dict())
        if self.depth > 0:
            tools.append(_REPORT_TO_PARENT_TOOL.to_dict())
        if self.depth == 0:
            tools += list(MEMORY_TOOLS)
        if self.depth == 2:
            tools += _LEVEL_2_ENV_TOOLS
        return tools

    def __getattr__(self, name: str):
        # Intercepts env tool calls for Level 2 and memory tool calls for Level 0.
        # Only reached when the attribute isn't found through normal lookup.
        if name in _ENV_TOOL_NAMES:
            async def handler(args: dict) -> dict:
                self._last_tool = name
                return await _dispatch(name, args)
            return handler
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    async def get_current_view(self, args: dict) -> dict:
        if self.depth == 2 and self._last_tool == "get_current_view":
            return {
                "error": (
                    "Consecutive get_current_view calls are not allowed. "
                    "Call an action tool first, or proceed to REPORT_TO_PARENT."
                )
            }
        self._last_tool = "get_current_view"
        return await _dispatch("get_current_view", args)

    # ------------------------------------------------------------------
    # SPAWN_SUBAGENT
    # ------------------------------------------------------------------

    async def spawn_subagent(self, args: dict) -> dict:
        if self.depth >= 2:
            return {"success": False, "message": "Level 2 agents cannot spawn sub-agents."}

        sid = args["subagent_id"]
        if sid in self._finished_tracker:
            return {
                "success": False,
                "message": f"Conflicting agent ID '{sid}'. Use a unique ID.",
            }

        done_event = asyncio.Event()
        self._finished_tracker[sid] = done_event

        # Fresh child context — empty base so full system prompt comes from the plugin
        sub_ctx = AgentContext(
            base_system_prompt="",
            model_name=self.ctx.model_name,
            thinking_effort=self.ctx.thinking_effort,
            client=self.ctx.client,
            main_app=self.ctx.main_app,
            metadata={},
        )

        # Fresh plugin instance at depth+1, wired to signal this level's event
        child_plugin = ThreeLevelsPlugin(sub_ctx, depth=self.depth + 1)
        child_plugin._parent_event = done_event
        child_plugin._parent_results = self._results

        sub_ctx.plugins.append(child_plugin)
        sub_ctx.reload_system_prompt()
        sub_ctx.reload_tools()

        initial_prompt = (
            f"{args['input_prompt']}\n\n"
            f"**END GOAL:** {args['goal_prompt']}\n\n"
            f"Your sub-agent ID is: `{sid}`"
        )

        tabbed = self.ctx.main_app.query_one(TabbedContent)
        await tabbed.add_pane(
            TabPane(
                f"🤖 L{self.depth + 1}#{self._subagent_count}",
                _SubagentPane(initial_prompt, sub_ctx),
            )
        )
        self._subagent_count += 1

        await done_event.wait()
        return copy.deepcopy(self._results.get(sid, {}))

    # ------------------------------------------------------------------
    # REPORT_TO_PARENT
    # ------------------------------------------------------------------

    async def report_to_parent(self, args: dict) -> dict:
        if self._parent_event is None or self._parent_results is None:
            return {"success": False, "message": "No parent to report to."}

        sid = args.get("subagent_id", "unknown")
        self._parent_results[sid] = args
        self._parent_event.set()

        return {
            "success": True,
            "message": "Report received. You are done — do not call any further tools.",
        }

    # ------------------------------------------------------------------
    # Memory tool pass-throughs (Level 0 only via assemble_tool_dicts)
    # ------------------------------------------------------------------

    async def store_semantic_memory(self, args: dict) -> dict:
        return await _dispatch("store_semantic_memory", args)

    async def recall_semantic_memory(self, args: dict) -> dict:
        return await _dispatch("recall_semantic_memory", args)

    async def clear_semantic_memory(self, args: dict) -> dict:
        return await _dispatch("clear_semantic_memory", args)

    # ------------------------------------------------------------------
    # Lifecycle hooks (Level 0 only)
    # ------------------------------------------------------------------

    async def on_turn_start(self, context: AgentContext) -> None:
        if self.depth != 0:
            return
        self._turn_count += 1
        recall = await self._run_semantic_learner(context)
        self._dynamic_system_prompt = _build_level0_prompt(
            load_semantic_memory(), load_episodic_memory(), recall
        )
        context.reload_system_prompt()

    async def on_turn_end(self, context: AgentContext) -> None:
        if self.depth != 0:
            return
        await self._run_episodic_learner(context)
        # Refresh prompt without recall after the turn settles
        self._dynamic_system_prompt = _build_level0_prompt(
            load_semantic_memory(), load_episodic_memory()
        )
        context.reload_system_prompt()

    # ------------------------------------------------------------------
    # Semantic Associative Learner
    # ------------------------------------------------------------------

    async def _run_semantic_learner(self, context: AgentContext) -> str:
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

            text = "".join(
                part.text
                for item in response.output if hasattr(item, "content")
                for part in item.content if hasattr(part, "text")
            )

            data = _extract_json(text)
            if data:
                new_fact = data.get("new_semantic_memory", "").strip()
                recall = data.get("recall", "").strip()
                if new_fact:
                    facts = load_semantic_memory()
                    facts.append(f"[t={self._turn_count}] {new_fact}")
                    _save_semantic_memory(facts)
                    context.log(f"[L0 Semantic] +fact: {new_fact[:80]}")
                if recall:
                    context.log(f"[L0 Semantic] recall: {recall[:80]}")
                return recall

        except Exception as e:
            context.log(f"[L0 Semantic] skipped ({e})")

        return ""

    # ------------------------------------------------------------------
    # Episodic Associative Learner
    # ------------------------------------------------------------------

    async def _run_episodic_learner(self, context: AgentContext) -> None:
        try:
            exchange_parts = []
            for msg in context.messages[-12:]:
                role = msg.get("role") or msg.get("type", "")
                content = msg.get("content") or msg.get("output") or msg.get("arguments", "")
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

            text = "".join(
                part.text
                for item in response.output if hasattr(item, "content")
                for part in item.content if hasattr(part, "text")
            )

            data = _extract_json(text)
            if data:
                save_episodic_memory(data)
                context.log(f"[L0 Episodic] saved: {data.get('dense_summary', '')[:80]}")

        except Exception as e:
            context.log(f"[L0 Episodic] skipped ({e})")


def setup(context: AgentContext) -> AgentPlugin:
    return ThreeLevelsPlugin(context, depth=0)
