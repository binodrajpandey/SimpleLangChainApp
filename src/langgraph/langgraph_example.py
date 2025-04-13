from pathlib import Path
from typing import TypedDict
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

# Build graph
builder = StateGraph(State)

# Add nodes
builder.add_node("add_one", add_one_node)
builder.add_node("multiply", multiply_node)

# Set entry, flow, and finish
builder.set_entry_point("add_one")
builder.add_edge("add_one", "multiply")
builder.set_finish_point("multiply")

# Compile graph
graph = builder.compile()

# Run with initial number
input_data: State = {"number": 3}
result = graph.invoke(input_data)

print("Final Result:", result["number"])

for event in graph.stream(input_data):
        for value in event.values():
            print(value)

img_data = graph.get_graph().draw_mermaid_png()
image_path = Path("graph_output.png")
with open(image_path, "wb") as f:
    f.write(img_data)

print(f"Image saved to {image_path.resolve()}")