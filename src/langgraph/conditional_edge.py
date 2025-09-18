from typing import TypedDict, Literal

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    number: int

# Node A
def add_one_node(state: State) -> State:
    number = state["number"]
    number = number + 1
    return {"number": number}

# Node B
def multiply_node(state) -> State:
    number = state["number"]
    number = number * 2
    return {"number": number}

def next_edge_after_addition(
    state: State,
) -> Literal["multiply", END]:
    if state.get("number") % 2 == 0:
        return "multiply"
    else:
        return END

# Build graph
builder = StateGraph(State)

# Add nodes
builder.add_node("add_one", add_one_node)
builder.add_node("multiply", multiply_node)

# Set entry, flow, and finish
builder.add_edge(START, "add_one")
builder.add_conditional_edges("add_one", next_edge_after_addition)
builder.add_edge("multiply", END)

# Compile graph
graph = builder.compile()

# Run with initial number.
input_data: State = {"number": 4} # Also test with value 3
result = graph.invoke(input_data)

print("Final Result:", result["number"])

for event in graph.stream(input_data):
        for value in event.values():
            print(value)
