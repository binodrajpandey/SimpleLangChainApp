from typing import Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from operator import add


class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

def node_1(state):
    return {"foo": 2}  # This is treated as an update to the state

def node_2(state):
    return {"bar": ["bye"]}

builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)
graph = builder.compile()

input_data = {"foo": 1, "bar": ["hi"]}
result = graph.invoke(input_data)
print(result)

# After applying node_1 update: {"foo": 2, "bar": ["hi"]}
# After applying node_2 update: {"foo": 2, "bar": ["hi", "bye"]}
