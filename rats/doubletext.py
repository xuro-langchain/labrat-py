import random 
import asyncio
from typing import Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, Interrupt, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from IPython.display import Image, display


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    """
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str

class InputState(TypedDict):
    question: str


async def wait(state: GraphState):
    question = state["question"]
    return {"question": question, "messages": [HumanMessage(content=question)]}

async def respond(state: GraphState):
    await asyncio.sleep(3)
    question = state["question"]
    answer = "Question: " + question + "\nAnswer: How am I supposed to know?"
    return {"answer": answer, "messages": [AIMessage(content=answer)]}



def make_graph(memory):
    graph = StateGraph(GraphState, input=InputState)

    graph.add_node("wait", wait)
    graph.add_node("respond", respond)


    graph.add_edge(START, "wait")
    graph.add_edge("wait", "respond")
    graph.add_edge("respond", END)

    return graph.compile(checkpointer=memory)


def print_messages(response):
    if isinstance(response, dict):
        print("STATE UPDATE ----------------")
        for key in response:
            if key == "messages":
                print("messages: " + response["messages"][-1].content)
            else:
                print(key + ": " + str(response[key]))


async def run(graph: StateGraph, thread_id: int, state: GraphState):
    config = {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }
    
   
    async for output in graph.astream(state, config, stream_mode="updates"):
        if END in output or START in output:
            continue
        for key, value in output.items():
            print_messages(value)


async def main():            
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        graph = make_graph(memory)
        thread_id = random.randint(0, 1000000)
        state = {}
        
        state["question"] = "What is the capital of France?"
        task1 = asyncio.create_task(run(graph, thread_id, state))
        await asyncio.sleep(1)
        state["question"] = "What is the capital of Germany?"
        task2 = asyncio.create_task(run(graph, thread_id, state))
        await asyncio.gather(task1, task2)

if __name__ == "__main__":
    asyncio.run(main())