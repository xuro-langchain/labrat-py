# server.py
import os
import httpx
import asyncio
import threading
from typing_extensions import TypedDict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langsmith.run_helpers import tracing_context, get_current_run_tree
from langsmith import Client


load_dotenv(dotenv_path=".env", override=True)
os.environ["LANGSMITH_PROJECT"] = "distributed-parent"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Server (Child) ------------------------------------------------------------ #
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

app = FastAPI() 

@app.post("/tracing")
async def tracing(request: Request):
    parent_headers = {
        "langsmith-trace": request.headers.get("langsmith-trace"),
        "baggage": request.headers.get("baggage"),
    }
    # THIS IS THE DIFFERENCE
    with tracing_context(parent=parent_headers, replicas=[("distributed-parent", None), ("distributed-child", None)]):
        data = await request.json()
        result = await child_graph.ainvoke({"value": data["value"]})
        return result

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Client (Parent) ------------------------------------------------------------ #

class ParentState(TypedDict):
    input_value: int
    output_value: int

async def parent_node(state: ParentState) -> ParentState:
    print(f"Parent graph received input: {state['input_value']}")
    headers = {}
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        run_tree = get_current_run_tree()
        headers.update(run_tree.to_headers())
        response = await client.post("/tracing", headers=headers, json={"value": state["input_value"]})
        result = response.json()
        print(f"Child graph returned: {result['value']}")
        return {"input_value": state["input_value"], "output_value": result["value"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("parent_node", parent_node)
parent_builder.add_edge(START, "parent_node")
parent_builder.add_edge("parent_node", END)
parent_graph = parent_builder.compile()

async def run_client():
    # Run the parent graph with initial input
    with tracing_context(replicas=[("distributed-parent", None), ("distributed-child", None)]):
        result = await parent_graph.ainvoke({"input_value": 10, "output_value": 0})
        return result["output_value"]


# ---------- Main ---------- #
async def main():
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    await asyncio.sleep(1)

    result = await run_client()
    print("Server replied:", result)
    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())