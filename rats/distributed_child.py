# server.py
import os
from langchain_openai import ChatOpenAI
import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from langsmith.run_helpers import tracing_context
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig

load_dotenv(dotenv_path=".env", override=True)
os.environ["LANGSMITH_PROJECT"] = "distributed-child"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ChildState(TypedDict):
    value: int

async def child_node(state: ChildState):
    generation = llm.invoke("What is " + str(state["value"]) + " 1? Respond with a single number, no extra text.")
    return {"value": int(generation.content)}

child_builder = StateGraph(ChildState)
child_builder.add_node("child_node", child_node)
child_builder.add_edge(START, "child_node")
child_builder.add_edge("child_node", END)
child_graph = child_builder.compile()


app = FastAPI()  # Or Flask, Django, or any other framework


@app.post("/tracing")
async def tracing(request: Request):
    # request.headers:  {"langsmith-trace": "..."}
    # as well as optional metadata/tags in `baggage`
    parent_headers = {
        "langsmith-trace": request.headers.get("langsmith-trace"),
        "baggage": request.headers.get("baggage"),
    }
    # replicas = [("distributed-child", None), ("distributed-parent", None)]
    with tracing_context(parent=parent_headers):
        # config = RunnableConfig(langsmith_extra={"replicas": replicas})
        data = await request.json()
        result = await child_graph.ainvoke({"value": data["value"]})
        print(f"Child graph returned: {result}")
        return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)