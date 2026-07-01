from dotenv import load_dotenv

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(model="gpt-4.o-mini")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system',
         """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
         """ 
        ),
        ('placeholder', "{char_history}"),
        ('human', "{query}"),
        ('placeholder', "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_agent(
    model = llm,
    tools = [],
    system_prompt = prompt
)

response = agent.invoke({
    "query": "what is the capital city of Rwanda?"
})

print(response)

