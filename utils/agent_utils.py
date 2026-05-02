import json
import re

from agent_tools3 import (load_semantic_memory, load_episodic_memory, save_episodic_memory)

EPISODIC_SYNTHESIS_INSTRUCTION = (
    "You are a memory synthesis module for an embodied agent. "
    "Given a recent agent-environment exchange, return ONLY a valid JSON object "
    "(no markdown fences, no extra text) with exactly these four keys:\n"
    "  dense_summary  — 1-2 sentence summary of what happened\n"
    "  surprise       — anything unexpected or that contradicted prior beliefs\n"
    "  what_worked    — actions or strategies that were effective\n"
    "  what_to_avoid  — actions or approaches that failed or should not be repeated"
)


def build_system_instruction() -> str:
    """Build the system prompt, injecting current semantic and episodic memory."""
    facts = load_semantic_memory()
    episodic = load_episodic_memory()

    parts = [
        "You are Sari, an embodied agent operating in a 3D environment. "
        "Use your tools to navigate, manipulate objects, and perceive the scene. "
        "Use memory tools to store important facts and recall them across interactions. "
        "Always switch to the correct mode before using navigation, manipulation, or perception tools."
    ]

    if facts:
        facts_str = "\n".join(f"- {f}" for f in facts)
        parts.append(f"\n## SEMANTIC MEMORY\n{facts_str}")

    if episodic:
        parts.append(
            f"\n## EPISODIC MEMORY (last session)\n"
            f"- Summary: {episodic.get('dense_summary', 'N/A')}\n"
            f"- Surprise: {episodic.get('surprise', 'N/A')}\n"
            f"- What worked: {episodic.get('what_worked', 'N/A')}\n"
            f"- What to avoid: {episodic.get('what_to_avoid', 'N/A')}"
        )

    return "\n".join(parts)


async def synthesize_episodic_memory(client, model_name: str, chat_log: list) -> None:
    """Run a post-turn LLM call to synthesize episodic memory from the last exchange."""
    exchange_parts = []
    for msg in chat_log[-10:]:
        role = msg.get("role") or msg.get("type", "")
        content = msg.get("content") or msg.get("output") or msg.get("arguments", "")
        if isinstance(content, str) and content.strip():
            exchange_parts.append(f"{role}: {content.strip()}")

    if not exchange_parts:
        return

    exchange_text = "\n\n".join(exchange_parts)

    try:
        response = await client.responses.create(
            model=model_name,
            instructions=EPISODIC_SYNTHESIS_INSTRUCTION,
            input=[{"role": "user", "content": exchange_text}],
            max_output_tokens=512,
        )

        text = ""
        for item in response.output:
            if hasattr(item, "content"):
                for part in item.content:
                    if hasattr(part, "text"):
                        text += part.text

        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            entry = json.loads(match.group())
            save_episodic_memory(entry)
    except Exception:
        pass  # episodic synthesis is best-effort; never block the main flow


def append_to_chat_log(chat_log: list, role: str, message) -> None:
    chat_log.append({"role": role, "content": message})
