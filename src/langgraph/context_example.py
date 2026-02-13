# State → the workflow data that flows between nodes
#
# RunnableConfig → execution settings
#
# Context → external dependencies / environment
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from src.llm import get_llm


class AppState(TypedDict):
    topic: list
    result: str


class ContextSchema(TypedDict):
    llm: any

def generate_summary(state: AppState, config: RunnableConfig, runtime: Runtime[ContextSchema]):
    llm = runtime.context["llm"]
    summary = llm.invoke(state["topic"])
    return {"result": summary}

builder = StateGraph(AppState)
builder.add_node("generate_summary", generate_summary)
builder.set_entry_point("generate_summary")
graph = builder.compile()


result = graph.invoke(
    input={"topic": "Valentine day special"},
    config={
        "configurable": {"thread_id": "abc"}
    },
    context={
        "llm": get_llm()
    }
)

print(result["result"].content)
