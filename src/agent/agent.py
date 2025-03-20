from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "you are friendly assistant called Max."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

search = TavilySearchResults()
tools = [search]
agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools
)

response = agent_executor.invoke({
    "input": "What is the weather in Tokyo today?"
})

print(response)

#TODO:// LCEL tool
