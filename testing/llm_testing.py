from langchain_core.messages import SystemMessage, HumanMessage
from core.llm import get_llm, LLMConfig, stream_response

# ── Style 1: quick ────────────────────────────────────────────────────────────
llm = get_llm("groq", "openai/gpt-oss-120b")


# ── Regular invoke ────────────────────────────────────────────────────────────
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="what is the capital of France?"),
]
response = llm.invoke(messages)
print(response.content)

# ── Streaming ─────────────────────────────────────────────────────────────────
for chunk in stream_response(llm, messages):
    print(chunk, end="", flush=True)

# ── Async streaming ───────────────────────────────────────────────────────────
import asyncio
from core.llm.streaming import astream_response

async def main():
    async for chunk in astream_response(llm, messages):
        print(chunk, end="", flush=True)

asyncio.run(main())