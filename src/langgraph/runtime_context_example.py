from typing import TypedDict, Annotated

from langgraph.graph import StateGraph
from langgraph.types import RunnableConfig
import operator

# State schema
class AppState(TypedDict):
    messages: Annotated[list[str], operator.add]

# Node using runtime context
def add_message(state: AppState, config: RunnableConfig) -> AppState:
    user = config["configurable"].get("user_id", "anonymous")
    return {"messages": [f"Hello from {user}"]}

# Build graph
builder = StateGraph(AppState)
builder.add_node("greeter", add_message)
builder.set_entry_point("greeter")
graph = builder.compile()

# Run with runtime context
config = {"configurable": {"user_id": "user123"}, "run_name": "demo-run"}
print(graph.invoke({"messages": ["initial message"]}, config))
