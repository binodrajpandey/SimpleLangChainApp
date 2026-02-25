from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# -------------------------
# Tools
# -------------------------

search = TavilySearch()

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
).split_documents(docs)

vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name="langsmith_search",
    description="Search for information about LangSmith."
)

tools = [search, retriever_tool]

# -------------------------
# LLM
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",  # better than gpt-3.5-turbo
    temperature=0
)

# -------------------------
# Agent (1.x style)
# -------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant."
)

# -------------------------
# Chat History
# -------------------------

message_history = ChatMessageHistory()

agent_with_history = RunnableWithMessageHistory(
    agent,
    lambda session_id: message_history,
    input_messages_key="messages",
    history_messages_key="messages",
)

# -------------------------
# Run
# -------------------------

result = agent_with_history.invoke(
    {
        "messages": [
            {"role": "user", "content": "Hi! I am Binod"}
        ]
    },
    config={"configurable": {"session_id": "foo"}}
)

result = agent_with_history.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's my name?"}
        ]
    },
    config={"configurable": {"session_id": "foo"}}
)

print(result)
