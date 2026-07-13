"""Interactive CLI agent that acts as a Meta Senior Staff engineer
career-roadmap assistant, powered by LangChain and OpenAI."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

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


def stream_response(agent, request: str) -> None:
    """Stream the agent's answer for a single user request to stdout."""
    question = HumanMessage(content=request)
    for chunk in agent.stream({"messages": [question]}, stream_mode="messages"):
        msg = chunk[0]
        if isinstance(msg, AIMessage) and msg.content:
            print(msg.content, end="", flush=True)
    print()


def main() -> None:
    """Run an interactive question/answer loop until the user exits."""
    agent = build_agent()
    print("Ask me about breaking into Meta. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            request = input("\nHow can I help you today: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not request:
            print("Please enter a question.")
            continue
        if request.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            stream_response(agent, request)
        except Exception as error:  # noqa: BLE001
            print(f"\nSomething went wrong: {error}")


if __name__ == "__main__":
    main()


