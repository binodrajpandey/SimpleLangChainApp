# reference https://python.langchain.com/v0.1/docs/modules/agents/quick_start/
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
load_dotenv()

search = TavilySearchResults()

result = search.invoke("what is the weather in SF")
# print(result)

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langsmith_search",
    description="Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
)

tools = [search, retriever_tool]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=100
)

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt.messages)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Check each one by uncommenting to see which one it invokes. It is stateless.
# result = agent_executor.invoke({"input": "hi!"})
# result = agent_executor.invoke({"input": "how can langsmith help with testing?"})
# result = agent_executor.invoke({"input": "whats the weather in sf?"})

# Adding chat_history
# Here we pass in an empty list of messages for chat_history because it is the first message in the chat
# result = agent_executor.invoke({"input": "hi! my name is Binod", "chat_history": []})
# result = agent_executor.invoke({
#     "input": "What is My Name?",
#     "chat_history": [
#         HumanMessage(content="hi! my name is Binod"),
#         AIMessage(content="Hello Binod! How can I assist you today?"),
#     ],
# })
# print(result)


# Let's use ChatHistory to save message outside of chain.
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda sessionId: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
result = agent_with_chat_history.invoke({
    "input": "hi! I am Binod"},
    config={"configurable": {"session_id": "<foo>"}}
)

result = agent_with_chat_history.invoke(
    {"input": "what's my name?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
print(result)
