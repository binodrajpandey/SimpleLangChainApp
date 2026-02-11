from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict

# Define graph state
class MyState(TypedDict):
    message: str
    counter: int

# Node function
def increment(state: MyState):
    return {"counter": state["counter"] + 1}

# Create graph
builder = StateGraph(MyState)
builder.add_node("increment", increment)
builder.set_entry_point("increment")
builder.set_finish_point("increment")

# Add persistence
checkpointer = SqliteSaver.from_conn_string("sqlite:///graph.db")

graph = builder.compile(checkpointer=checkpointer)

# Run with thread_id (VERY IMPORTANT)
config = {"configurable": {"thread_id": "user-1"}}

result = graph.invoke(
    {"message": "hello", "counter": 0},
    config=config
)

print(result)
