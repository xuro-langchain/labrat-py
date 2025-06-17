import os
import httpx
import asyncio
from dotenv import load_dotenv

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langsmith.run_helpers import tracing_context, get_current_run_tree
from langsmith import Client

load_dotenv(dotenv_path=".env", override=True)
os.environ["LANGSMITH_PROJECT"] = "distributed-parent"


# Define parent graph state and nodes
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


async def run_parent_graph():
    # Run the parent graph with initial input
    result = await parent_graph.ainvoke({"input_value": 10, "output_value": 0})
    print(f"Final result from parent graph: {result['output_value']}")

async def main():
    await run_parent_graph()

if __name__ == "__main__":
    asyncio.run(main())
