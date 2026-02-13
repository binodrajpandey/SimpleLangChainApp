from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

# Define graph state
class MyState(TypedDict):
    number: int

# Node function
def increment(state: MyState):
    return {"number": state["number"] + 1}

def multiply(state: MyState):
    return {"number": state["number"] *5 }


# Create graph
builder = StateGraph(MyState)
builder.add_node("increment", increment)
builder.add_node("multiply", multiply)
builder.set_entry_point("increment")
builder.add_edge("increment", "multiply")
builder.set_finish_point("multiply")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {
        "thread_id": "1"
    }
}

result =graph.invoke(input={"number": 0}, config= config)
print(result)

state = graph.get_state(config)
print(state)
print("result from state", state.values)
