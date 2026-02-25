from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.postgres import PostgresSaver
from dotenv import load_dotenv

load_dotenv()

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"



DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        state_schema=CustomAgentState,
        checkpointer=checkpointer,
    )

    # result = agent.invoke(
    #     {
    #         "messages": [{"role": "user", "content": "Hello tell me user name"}],
    #         "user_id": "user_123",
    #         "preferences": {"theme": "dark"}
    #     },
    #     {"configurable": {"thread_id": "1"}})

    result = agent.invoke({
        "messages": "look up user information",
        "user_id": "user_123"
    },
        {"configurable": {"thread_id": "1"}}
    )
    print(result["messages"][-1].content)

    # print(result)

from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
