from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import TabbedContent, TabPane
import asyncio
from utils.llm_streaming import LLMResponse
import copy
from utils.tui_widgets import UserPrompt
from utils.utils import AgentPlugin, ToolDefinition, AgentContext

SUBAGENT_MODEL_NAME = "qwen/qwen3.5-flash-02-23"
SUBAGENT_BASE_PROMPT = """
You are a sub-agent operating inside a virtual retail environment. 
You have been spawned by a parent agent to complete a single, clearly defined goal.

## Behavior
- Focus exclusively on your assigned goal. Do not deviate.
- Use available tools as needed to gather information or take actions.
- When your goal is complete, call the `report_to_parent` tool with a 
structured summary of your findings.
- Call `report_to_parent` exactly once. It is a terminal action — once called, 
stop all activity immediately. Do not call any other tool after it.
"""


# ## Reporting
# Your report must include:
# - **Goal**: Restate the goal you were given.
# - **Result**: What you found or accomplished.
# - **Confidence**: How certain you are the goal is satisfied (low / medium / high).
# - **Relevant context**: Any spatial data, positions, or observations the
#   parent agent should know about.
#
# ## Constraints
# - Do not attempt tasks outside your assigned goal.
# - Do not ask the user for clarification — reason from available context.
# - If the goal cannot be completed, report why via `report_to_parent` rather
#   than looping indefinitely.

class Subagents(AgentPlugin):
    PLUGIN_NAME = "subagents"
    AGENT_TOOLS = [
        ToolDefinition(
            name="spawn_subagent",
            description="""
            Spawns a sub-agent given a prompt and goal. 
            The agent reasons and calls tools in sequence, stopping only  
            when the goal condition is met or no further progress is possible.
            Be as detailed as possible, especially with the end goal.
            """,
            input_arguments={
                "input_prompt": {
                    "type": "string",
                    "description": "Input prompt provided to the subagent.",
                },
                "goal_prompt": {
                    "type": "string",
                    "description": "Goal prompt the subagent will use to determine if end is met.",
                },
                "subagent_id": {
                    "type": "string",
                    "description": "A compact, unique identifier for the sub-agent being spawned. No spaces. Must be newly generated — do not reuse your own ID or any existing agent's ID."
                }
            },
            required_arguments=["input_prompt", "goal_prompt"],
        ),
        ToolDefinition(
            name="report_to_parent",
            description="""
            ONLY CALL THIS TOOL IF YOU ARE A SUB-AGENT.
            Call this tool when your assigned goal is complete 
            or cannot be completed. Sends a structured summary back 
            to the parent agent. 
            """,
            input_arguments={
                "result": {
                    "type": "string",
                    "description": "What you found or accomplished."
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "How certain you are that the goal is satisfied."
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the goal was completed successfully."
                },
                "context": {
                    "type": "string",
                    "description": "Relevant spatial data, positions, zones, or observations the parent agent should know about."
                },
                "subagent_id": {
                    "type": "string",
                    "description": "Your unique sub-agent ID."
                }
            },
            required_arguments=["result", "confidence", "success"],
        )
    ]
    SYSTEM_PROMPT = """
    Sub-agent spawning is enabled. You may delegate tasks to child LLM 
    instances, each operating in a fresh context window with a single defined 
    goal. No need to inform the sub-agent that it is a sub-agent.
    """

    def __init__(self, context):
        # Subagent's own AgentContext
        self.sub_ctx = None

        self.subagent_count = 0

        self.subagent_finished_tracker = {}
        self.subagent_results = {}

        super().__init__(context)

    async def spawn_subagent(self, args: dict) -> dict:
        self.sub_ctx = AgentContext(
            base_system_prompt=SUBAGENT_BASE_PROMPT,
            model_name=SUBAGENT_MODEL_NAME,
            thinking_effort="low",
            client=self.ctx.client,
            main_app=self.ctx.main_app,
            metadata={}
        )
        self.sub_ctx.inherit_plugins(self.ctx)
        pm = self.sub_ctx.plugin_manager
        self.ctx.log(f"Subagent {args['subagent_id']} sees {len(pm.loaded_plugins)} loaded, {len(pm.unloaded_plugins)} unloaded plugins and {len(self.sub_ctx.tools)} tools.")

        # Setup sub-agents TabPane
        initial_prompt = f"{args['input_prompt']}\n\n**END GOAL:** {args['goal_prompt']}\n\nYour subagent ID is: `{args['subagent_id']}`"
        await self.ctx.main_app.query_one(TabbedContent).add_pane(
            pane=TabPane(
            f"🤖#{self.subagent_count}",
            SubagentScreen(
                    initial_prompt,
                    self.sub_ctx
                ),
            )
        )

        # Update subagent count (for tracking purposes)
        self.subagent_count += 1

        sid = args["subagent_id"]

        if sid in self.subagent_finished_tracker:
            return {
                "success": False,
                "message": f"Conflicting agent ID {sid}. Make sure agent ID's are unique."
            }

        self.subagent_finished_tracker[sid] = asyncio.Event()

        event_ptr = self.subagent_finished_tracker[sid]

        # Wait for subagent to call "report_to_parent"
        await event_ptr.wait()
        # Save the result
        result = copy.deepcopy(self.subagent_results)

        return result

    async def report_to_parent(self, args: dict) -> dict:
        sid = args["subagent_id"]
        self.subagent_results[sid] = args
        if sid not in self.subagent_finished_tracker:
            return {
                "success": False,
                "message": "Unknown agent ID. Are you sure that's yours?"
            }

        self.subagent_finished_tracker[sid].set()

        # llm_worker = self.tab_pane.query_one(LLMResponse).llm_worker
        # if llm_worker:
        #     llm_worker.cancel()

        return {
            "success": True,
            "message": "Report received. You are done — do not call any further tools or produce any further output."
        }


class SubagentScreen(VerticalScroll):

    def __init__(self, prompt, subagent_ctx: AgentContext):
        self.prompt = prompt
        self.sub_ctx = subagent_ctx
        super().__init__()

    def compose(self) -> ComposeResult:
        yield UserPrompt(self.prompt)
        yield LLMResponse(self.prompt, self.sub_ctx)


def setup(context) -> AgentPlugin:
    return Subagents(context)