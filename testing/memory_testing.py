"""
test_memory.py — Interactive CLI chat loop to test MemoryManager with Groq.

Usage:
    python test_memory.py

Requirements:
    pip install langchain-groq langchain-core python-dotenv rank_bm25

Commands you can try in the chat:
    "remember that my name is Prateek"
    "remember that I prefer Python"     → agent calls remember()
    "what do you know about me?"        → agent calls recall()
    "forget my name"                    → agent calls forget()
    "show all memories"                 → agent calls recall("*")
    "exit" or "quit"                    → exits the loop
"""

from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ── Memory imports ────────────────────────────────────────────────────────────
# Adjust this import path to match your project structure
from core.memory.manager import MemoryManager
from core.llm import get_llm, LLMConfig, stream_response


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Memory instance  (single shared instance for the whole session)
# ─────────────────────────────────────────────────────────────────────────────

mem = MemoryManager(memory_dir="./context/memory")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tools the LLM can call
# ─────────────────────────────────────────────────────────────────────────────

@tool
def remember(content: str, key: str) -> str:
    """
    Store a piece of information in long-term memory.

    Args:
        content: The information to remember (e.g. "Prateek prefers Python").
        key:     A short snake_case identifier for this memory
                 (e.g. "user_name", "user_preference").
    """
    result = mem.remember(content, key=key)
    print(f"\n  💾  [TOOL] remember({key!r}) → {result}")
    return result


@tool
def recall(query: str) -> str:
    """
    Search long-term memory for information relevant to the query.
    Pass '*' to retrieve all stored memories.

    Args:
        query: A natural-language question or keyword to search memory.
               Use '*' to dump everything.
    """
    result = mem.recall(query)
    print(f"\n  🔍  [TOOL] recall({query!r}) →\n{result}")
    return result


@tool
def forget(key: str) -> str:
    """
    Delete a memory entry by its key.

    Args:
        key: The exact key of the memory entry to remove.
    """
    result = mem.forget(key)
    print(f"\n  🗑️   [TOOL] forget({key!r}) → {result}")
    return result


TOOLS = [remember, recall, forget]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LLM setup
# ─────────────────────────────────────────────────────────────────────────────

def build_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌  GROQ_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
        streaming=False,           # keep simple for tool-call loop
        groq_api_key=api_key,
    ).bind_tools(TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Tool-call execution loop (handles multi-step tool use)
# ─────────────────────────────────────────────────────────────────────────────

def run_tool_loop(llm: ChatGroq, messages: list) -> str:
    """
    Keep calling the LLM until it stops issuing tool calls.
    Returns the final text response.
    """
    while True:
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls → we have the final answer
        if not response.tool_calls:
            return response.content or ""

        # Execute every tool the LLM requested
        for tc in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn is None:
                tool_result = f"Unknown tool: {tc['name']}"
            else:
                try:
                    tool_result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"],
                )
            )
        # Loop again so the LLM can process the tool results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  System prompt (includes boot context from memory)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a helpful assistant with long-term memory.

You have access to three memory tools:
- remember(content, key)  → store a fact
- recall(query)           → search memory; use '*' to see everything
- forget(key)             → delete a memory entry

Always use these tools proactively:
- When the user tells you something personal, call remember().
- When the user asks what you know about them, call recall().
- When the user asks you to forget something, call forget().

--- Existing Memory (from previous sessions) ---
{boot_context}
"""


def build_system_message() -> SystemMessage:
    boot = mem.boot_context()
    if not boot:
        boot = "(no memories yet)"
    return SystemMessage(content=SYSTEM_TEMPLATE.format(boot_context=boot))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Interactive CLI loop
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 60)
    print("  🧠  Memory Test — Interactive CLI")
    print("=" * 60)
    print("  Commands:")
    print("    Type anything to chat with the agent.")
    print("    'show memories'  → dump all stored memories")
    print("    'exit' / 'quit'  → quit")
    print("=" * 60 + "\n")


def main():
    print_banner()

    llm = get_llm("groq", "openai/gpt-oss-120b")

    # Conversation history (persists for the whole session)
    messages: list = [build_system_message()]

    # Show what's already in memory at boot
    boot = mem.boot_context()
    if boot:
        print("📚  Boot context loaded:\n")
        print(boot)
        print("\n" + "-" * 60 + "\n")
    else:
        print("📭  No existing memories found. Starting fresh.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Bye! 👋")
            break

        # Shortcut: dump all memories without going through the LLM
        if user_input.lower() in {"show memories", "list memories", "dump memory"}:
            all_mem = mem.recall("*")
            print(f"\n📋  All memories:\n{all_mem}\n")
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            reply = run_tool_loop(llm, messages)
        except Exception as exc:
            print(f"\n❌  Error: {exc}\n")
            # Remove the failed human message so history stays clean
            messages.pop()
            continue

        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()