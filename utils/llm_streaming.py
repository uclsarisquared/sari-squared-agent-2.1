import json
from textual.containers import VerticalGroup, VerticalScroll
from agent_tools3 import dispatch_tool, TOOL_MODE_MAP
from textual.widgets import LoadingIndicator, RichLog, Markdown, Label
from utils.agent_utils import build_system_instruction, append_to_chat_log, synthesize_episodic_memory

async def stream_from_llm_api(widget: VerticalGroup, client, model_name, chat_log, all_tools, debug, mode_setter):
    # Lazy import avoids a circular dependency: sari_tui2 → llm_streaming → sari_tui2.
    # By call time all modules are fully loaded so this is safe.
    from sari_tui import LLMThinkingSummary, LLMToolCallDisplay, ModeDisplay, LLMResponse

    tool_call_deltas = ""
    tool_call_id = None
    tool_call_name = None
    tool_call_display = None
    spawned_continuation = False

    # TODO: Fix type checking here, make widget type hint LLMResponse
    if widget.prompt is not None:
        append_to_chat_log(chat_log, "user", widget.prompt)

    # noinspection PyTypeChecker
    stream = await client.responses.create(
        model=model_name,
        instructions=build_system_instruction(),
        input=chat_log,
        max_output_tokens=65536,
        reasoning={
            "effort": "low",
            "summary": "auto"
        },
        tools=all_tools,
        stream=True,
        parallel_tool_calls=False
    )

    async for event in stream:
        widget.query_one(LoadingIndicator).display = False

        if debug:
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
                    append_to_chat_log(chat_log, "assistant", event.item.content[0].text)

            # LLM has finished reasoning, close the thinking summary dropdown
            case "response.reasoning_summary_part.done" | "response.reasoning_text.done":
                widget.query_one(LLMThinkingSummary).done_thinking()

            # LLM calls a tool, mount a tool call display
            case "response.output_item.added":
                if event.item.type == "function_call":
                    tool_call_id = event.item.call_id
                    tool_call_name = event.item.name
                    tool_call_display = LLMToolCallDisplay(event.item.name)
                    await widget.parent.query_one(VerticalScroll).mount(tool_call_display)
                if event.item.type == "reasoning":
                    widget.query_one(LLMThinkingSummary).display = True

            # LLM is calling a tool, accumulate the tool call arguments
            case "response.function_call_arguments.delta":
                tool_call_deltas += event.delta

            # LLM is finished doing anything, update token/price usage
            # TODO: Will possibly crash with local models, fix
            case "response.completed":
                widget.query_one(Label).content = f"↑ {event.response.usage.input_tokens} Tok | ↓ {event.response.usage.output_tokens} Tok | ${event.response.usage.cost}"

            # LLM is done sending function arguments, parse it
            case "response.function_call_arguments.done":
                # When this is called, the tool call argument deltas have finished
                # accumulating, and we will have a full JSON to parse
                # Sample response: {"units": 5}
                args = json.loads(tool_call_deltas)
                tool_call_display.update_func_args(tool_call_deltas)
                # Reset tool_call_deltas
                tool_call_deltas = ""

                if tool_call_display.tool_name == "switch_mode":
                    new_mode = args["mode"]
                    mode_setter(new_mode)
                    widget.mode = new_mode
                    widget.app.query_one(ModeDisplay).update_mode(new_mode)
                    result = {"switched_to": new_mode}
                elif (required_mode := TOOL_MODE_MAP.get(tool_call_display.tool_name)) and required_mode != widget.mode:
                    result = {"error": f"Mode mismatch: '{tool_call_display.tool_name}' requires '{required_mode}' mode. Call switch_mode(\"{required_mode}\") first."}
                else:
                    try:
                        result = await dispatch_tool(tool_call_display.tool_name, args)
                    except Exception as e:
                        result = {"error": str(e)}

                # Make tool call display stop the loading animation
                tool_call_display.tool_done()

                chat_log.append({
                    "type": "function_call",
                    "call_id": tool_call_id,
                    "name": tool_call_name,
                    "arguments": json.dumps(args),
                })

                if isinstance(result, dict) and "image_base64" in result:
                    # Since tools that directly return a base64 image cannot be
                    # read, append a "user" role to the chat log with the image
                    chat_log.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": "Screenshot captured.",
                    })

                    chat_log.append({
                        "role": "user",
                        "content": [{
                            "type": "input_image",
                            "image_url": f"data:{result['mimeType']};base64,{result['image_base64']}",
                        }],
                    })
                else:
                    chat_log.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(result, default=list),
                    })

                spawned_continuation = True
                await widget.parent.mount(LLMResponse(None, widget.mode))

    if not spawned_continuation:
        await synthesize_episodic_memory(client, model_name, chat_log)
