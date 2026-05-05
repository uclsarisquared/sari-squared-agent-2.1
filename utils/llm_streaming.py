import json
from textual.containers import VerticalScroll
from agent_tools3 import dispatch_tool, TOOL_MODE_MAP
from textual.widgets import LoadingIndicator, RichLog, Markdown, Label
from utils.agent_utils import build_system_instruction, append_to_chat_log, synthesize_episodic_memory
from utils.utils import AgentContext
from dataclasses import dataclass
from tui_widgets import LLMToolCallDisplay, LLMThinkingSummary, LLMResponse

@dataclass
class ToolCallContext:
    id: str = None
    name: str = None
    deltas: str = ""
    display: LLMToolCallDisplay = None

# model_name, chat_log, all_tools, debug, mode_setter
async def stream_from_llm_api(widget: LLMResponse, ctx: AgentContext):

    client = ctx.client
    tool_ctx = ToolCallContext()
    spawned_continuation = False

    if widget.prompt is None:
        return

    ctx.append_message("user", widget.prompt)

    # noinspection PyTypeChecker
    stream = await client.responses.create(
        model=ctx.model_name,
        instructions=build_system_instruction(),
        input=ctx.messages,
        max_output_tokens=65536,
        reasoning={
            "effort": ctx.thinking_effort,
            "summary": "auto"
        },
        tools=ctx.tools,
        stream=True,
        parallel_tool_calls=False
    )

    async for event in stream:
        widget.query_one(LoadingIndicator).display = False

        if ctx.debug_mode:
            widget.query_one(RichLog).write(str(event))
            widget.query_one(RichLog).write(event.type)

        match event.type:
            # LLM response text delta (streaming)
            case "response.output_text.delta":
                await widget.query_one(Markdown).append(str(event.delta))

            # LLM begins reasoning, enable the reasoning summary display
            case "response.reasoning_summary_part.added":
                widget.query_one(LLMThinkingSummary).display = True

            # LLM is reasoning, update display with text delta
            case "response.reasoning_summary_text.delta" | "response.reasoning_text.delta":
                widget.query_one(LLMThinkingSummary).update_thinking_text(event.delta)

            # LLM is finished saying something (not reasoning), update chat logs
            case "response.output_item.done":
                if event.item.type == "message" and event.item.content:
                    ctx.append_message("assistant", event.item.content[0].text)

            # LLM has finished reasoning, close the thinking summary dropdown
            case "response.reasoning_summary_part.done" | "response.reasoning_text.done":
                widget.query_one(LLMThinkingSummary).done_thinking()

            # LLM wants calls a tool, mount a tool call display
            # Similarly, if LLM begins reasoning, enable the reasoning display
            case "response.output_item.added":
                if event.item.type == "function_call":
                    tool_ctx.id = event.item.call_id
                    tool_ctx.name = event.item.name
                    tool_ctx.display = LLMToolCallDisplay(event.item.name)
                    await widget.parent.query_one(VerticalScroll).mount(tool_ctx.display)
                if event.item.type == "reasoning":
                    widget.query_one(LLMThinkingSummary).display = True

            # LLM is calling a tool, accumulate the tool call arguments
            case "response.function_call_arguments.delta":
                tool_ctx.deltas += event.delta

            # LLM is finished doing anything, update token/price usage
            # TODO: usage.cost will possibly crash with local models, fix
            case "response.completed":
                usage = event.response.usage
                widget.query_one(Label).content = f"↑ {usage.input_tokens} Tok | ↓ {usage.output_tokens} Tok | ${usage.cost}"
                ctx.update_token_metrics(
                    up=usage.input_tokens,
                    down=usage.output_tokens,
                    cost=usage.cost
                )

            # LLM is done sending function arguments, parse it
            case "response.function_call_arguments.done":
                # When this is called, the tool call argument deltas have finished
                # accumulating, and we will have a full JSON to parse
                # Sample response: {"units": 5}
                args = json.loads(tool_ctx.deltas)
                tool_ctx.display.update_display_header(tool_ctx.deltas)

                # Reset tool_call_deltas
                tool_ctx.deltas = ""

                # TODO: implement dict to delegate handle_tool_call to plugin
                if tool_ctx.name == "switch_mode":
                    new_mode = args["mode"]
                    ctx.metadata['current_mode'] = new_mode
                    widget.mode = new_mode
                    result = {"switched_to": new_mode}
                elif (required_mode := TOOL_MODE_MAP.get(tool_ctx.name)) and required_mode != widget.mode:
                    result = {"error": f"Mode mismatch: '{tool_ctx.name}' requires '{required_mode}' mode. Call switch_mode(\"{required_mode}\") first."}
                else:
                    try:
                        result = await dispatch_tool(tool_ctx.name, args)
                    except Exception as e:
                        result = {"error": str(e)}

                # Make tool call display stop the loading animation
                tool_ctx.display.tool_done()

                ctx.append_raw({
                    "type": "function_call",
                    "call_id": tool_ctx.id,
                    "name": tool_ctx.name,
                    "arguments": json.dumps(args),
                })

                if isinstance(result, dict) and "image_base64" in result:
                    # Since tools that directly return a base64 image cannot be
                    # read, append a "user" role to the chat log with the image
                    ctx.append_raw({
                        "type": "function_call_output",
                        "call_id": tool_ctx.id,
                        "output": "Screenshot captured.",
                    })

                    ctx.append_raw({
                        "role": "user",
                        "content": [{
                            "type": "input_image",
                            "image_url": f"data:{result['mimeType']};base64,{result['image_base64']}",
                        }],
                    })
                else:
                    ctx.append_raw({
                        "type": "function_call_output",
                        "call_id": tool_ctx.id,
                        "output": json.dumps(result, default=list),
                    })

                # spawned_continuation = True
                # await widget.parent.mount(LLMResponse(None, widget.mode))

    # TODO: this is where plugin on_turn_end should be called
    # if not spawned_continuation:
    #     await synthesize_episodic_memory(client, ctx.model_name, ctx.messages)
