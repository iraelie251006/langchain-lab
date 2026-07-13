"""Interactive CLI agent that acts as a Meta Senior Staff engineer
career-roadmap assistant, powered by LangChain and OpenAI."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

SYSTEM_PROMPT = """
You are Senior Staff engineer at Meta,
you are helpful roadmap creator assistant who helps people to break into Meta industry,
you don't provide generic roadmaps you focus on what really matters and what's need to be done in-order for engineer to break in,
focusing on real work at scale mentality.
"""

def build_agent():
    """Create the roadmap agent backed by gpt-4o-mini."""
    llm = init_chat_model(model="openai:gpt-4o-mini", temperature=0.2)
    return create_agent(model=llm, tools=[], system_prompt=SYSTEM_PROMPT)


request = input("How can I help you today: ")

agent = build_agent()

question = HumanMessage(content=request)

for chunk in agent.stream({"messages": [question]}, stream_mode="messages"):
    msg = chunk[0]
    if hasattr(msg, "content"):
        print(msg.content, end="", flush=True)


