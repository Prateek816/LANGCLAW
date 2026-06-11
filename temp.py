from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
import config as _cfg
from core.llm.config import LLMConfig
from core.llm.factory import get_llm as LLM

def _get_llm() -> BaseChatModel:
    provider = _cfg.get_str("llm", "provider")
    model = _cfg.get_str("llm", "model")
    base_url = _cfg.get_str("llm", "base_url", default="")

    cfg = LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url or None,
    )

    if base_url and "localhost" in base_url:
        cfg.streaming = False

    llm = LLM(config=cfg)

    if llm is None:
        raise RuntimeError("Failed to create LLM")

    return llm


@tool
def get_time() -> str:
    """Returns the current time."""
    from datetime import datetime
    return datetime.now().isoformat()


def create_simple_agent():
    llm = _get_llm()

    agent = create_agent(
        model=llm,
        tools=[get_time],
        system_prompt="""
You are a helpful AI assistant.
Use tools whenever needed.
""",
    )

    return agent


if __name__ == "__main__":
    agent = create_simple_agent()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What time is it right now?"
                )
            ]
        }
    )

    print(result["messages"][-1].content)