import random 
import asyncio
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from contextlib import asynccontextmanager

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Interrupt, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langsmith import Client
from langsmith.run_helpers import tracing_context

from IPython.display import Image, display


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=3)

# Configure LangSmith client to hide data
langsmith_client = Client(
    hide_inputs=lambda inputs: {},
    hide_outputs=lambda outputs: {}
)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    """
    question: str
    query: str
    documents: List[str]
    messages: List[str]


class InputState(TypedDict):
    question: str


async def clarify(state: GraphState):
    question = state["question"]
    prompt = """You are an expert in Google Search. You know the optimal query for any question.
    Your job is to find the perfect search query for the question.

    Question: {question} 
    Answer:"""
    formatted = prompt.format(question=question)
    generation = await llm.ainvoke([HumanMessage(content=formatted)])
    return {"query": generation.content, "question": question}


async def search(state: GraphState):
    query = state["query"]
    documents = state.get("documents", [])

    web_docs = await web_search_tool.ainvoke({"query": query})
    for d in web_docs:
        web_results = Document(page_content=d["content"])
        documents.append(web_results)

    return {"documents": documents}



async def answer(state: GraphState):
    question = state["question"]
    documents = state.get("documents", [])
   
    prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
    Your job is to summarize the answer to a question based on relevant background context. 

    Question: {question} 

    Context: {context}

    Answer:"""
    formatted = prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = await llm.ainvoke([HumanMessage(content=formatted)])
    return {"messages": [generation]}

@asynccontextmanager
async def make_graph(memory):
    graph = StateGraph(GraphState, input=InputState)

    graph.add_node("clarify", clarify)
    graph.add_node("search", search, metadata={"category": "docs"})
    graph.add_node("answer", answer)

    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", "search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)
    with tracing_context(client=langsmith_client):
        yield graph.compile(checkpointer=memory)


def print_messages(response):
    if isinstance(response, dict):
        for key in response:
            if key == "documents":
                print("# of Documents: " + str(len(response["documents"])))
            elif key == "messages":
                print("messages: " + response["messages"][-1].content)
            else:
                print(key + ": " + str(response[key]))


async def run(graph: StateGraph):
    state: GraphState = {
        "messages": [],
    }

    thread_id = random.randint(0, 1000000)
    config = {
        "configurable": {
            "thread_id": str(thread_id),
            "checkpoint_ns": "para",
        },
        "langsmith_client": langsmith_client
    }
    while True:
        user = input('User (q to quit): ')
        if user in {'q', 'Q'}:
            print('AI: Goodbye!')
            break
        state["question"] = user
        turn_input = state

        try:
            async for chunk in graph.astream(turn_input, config, stream_mode=["messages"]):
                if isinstance(chunk, tuple) and isinstance(chunk[0], AIMessage):
                    print(chunk[0].content, flush=True, end="")
            print("\n")
        except Exception as e:
            print(f"Error: {str(e)}")
            raise e


async def main():            
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        async with make_graph(memory) as graph:
            await run(graph)

if __name__ == "__main__":
    asyncio.run(main())