"""
Full system test for PythonClaw tools (LC + backend)

Run:
    python test_tools_system.py
"""

import os

# IMPORTANT: configure sandbox BEFORE using tools
from core.tools import set_sandbox
from core.langtools import ALL_LC_TOOLS

# Allow writing in current directory
set_sandbox([os.getcwd()])


def safe(tool, args):
    try:
        print(f"\n🔧 {tool.name}")
        out = tool.invoke(args)
        print(out)
    except Exception as e:
        print(f"❌ Error: {e}")


def test_primitive():
    print("\n===== PRIMITIVE =====")

    for t in ALL_LC_TOOLS:
        if t.name == "write_file":
            safe(t, {"path": "test.txt", "content": "hello world"})

        elif t.name == "read_file":
            safe(t, "test.txt")

        elif t.name == "list_files":
            safe(t, ".")

        elif t.name == "run_command":
            safe(t, "echo hello")

        elif t.name == "send_file":
            safe(t, {"path": "test.txt"})


def test_memory():
    print("\n===== MEMORY =====")

    for t in ALL_LC_TOOLS:
        if t.name == "remember":
            safe(t, {"key": "name", "content": "Prateek"})

        elif t.name == "recall":
            safe(t, "name")

        elif t.name == "memory_list_files":
            safe(t, {})

        elif t.name == "memory_get":
            safe(t, "MEMORY.md")

        elif t.name == "forget":
            safe(t, "name")


def test_web():
    print("\n===== WEB SEARCH =====")

    for t in ALL_LC_TOOLS:
        if t.name == "web_search":
            safe(t, {"query": "AI news", "max_results": 2})


def test_skill():
    print("\n===== SKILL =====")

    for t in ALL_LC_TOOLS:
        if t.name == "create_skill":
            safe(t, {
                "name": "test_skill",
                "description": "test skill",
                "instructions": "## Instructions\nDo nothing"
            })

        elif t.name == "use_skill":
            safe(t, "test_skill")

        elif t.name == "list_skill_resources":
            safe(t, "test_skill")


def test_cron():
    print("\n===== CRON =====")

    for t in ALL_LC_TOOLS:
        if t.name == "cron_add":
            safe(t, {
                "job_id": "job1",
                "cron": "* * * * *",
                "prompt": "Say hello"
            })

        elif t.name == "cron_list":
            safe(t, {})

        elif t.name == "cron_remove":
            safe(t, "job1")


def test_kb():
    print("\n===== KNOWLEDGE BASE =====")

    for t in ALL_LC_TOOLS:
        if t.name == "consult_knowledge_base":
            safe(t, "test query")


if __name__ == "__main__":
    print("\n🚀 STARTING FULL TOOL TEST\n")

    test_primitive()
    test_memory()
    test_web()
    test_skill()
    test_cron()
    test_kb()

    print("\n✅ DONE\n")