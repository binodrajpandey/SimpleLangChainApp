from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

search = TavilySearch()
tools = [search]
agent = create_agent(
    model=model,
    system_prompt="you are friendly assistant called Max.",
    tools=tools
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in Tokyo today?"}
    ]
})

print(response)

#TODO:// LCEL tool
