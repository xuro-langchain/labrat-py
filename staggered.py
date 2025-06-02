import random 
import asyncio
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Interrupt, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from IPython.display import Image, display


# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Initialize web search tool
web_search_tool = TavilySearchResults(max_results=3)



class GraphState(TypedDict):
    """
    Represents the state of our graph.

    """
    question: str
    documents: List[str]
    backstory: str
    messages: List[str]
    docs_approval: bool
    backstory_approval: bool

class InputState(TypedDict):
    question: str


async def trim(state: GraphState):
    print("TRIM------------------")
    question = state["question"]
    question = question.strip()
    return {"question": question}


async def search(state: GraphState):
    print("SEARCH------------------")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    web_docs = await web_search_tool.ainvoke({"query": question})
    for d in web_docs:
        web_results = Document(page_content=d["content"])
        documents.append(web_results)

    return {"documents": documents}


async def confirm_docs(state: GraphState):
    print("CONFIRM DOCS------------------")
    human_response = interrupt({"query": "Please confirm that the documents look relevant. (Y/n)"})
    response = human_response
    if response.lower() != "y" and response.lower() != "yes":
        return {"docs_approval": False}
    return {"docs_approval": True}


async def sleep(state: GraphState):
    print("SLEEP------------------")
    question = state["question"]
    for i in range(5):
        print("sleeping...")
        await asyncio.sleep(1)
    return {"question": question}


async def gen_backstory(state: GraphState):
    print("BACKSTORY------------------")
    prompt = """You are an expert improv comedian and actor. 
    Based on the question asked, create an engaging persona you believe will entertain the user.
    Create a short 2 sentence backstory.

    Question: {question}

    Backstory:"""
    question = state["question"]
    formatted = prompt.format(question=question)
    generation = await llm.ainvoke([HumanMessage(content=formatted)])
    return {"backstory": generation.content}


async def confirm_backstory(state: GraphState):
    print("CONFIRM BACKSTORY------------------")
    human_response = interrupt({"query": "Do you find this persona interesting? (Y/n)"})
    response = human_response
    if response.lower() != "y" and response.lower() != "yes":
        return {"backstory_approval": False}
    return {"backstory_approval": True}



async def answer(state: GraphState):
    print("ANSWER------------------")
    question = state["question"]
    documents = state.get("documents", [])
    backstory_approval = state.get("backstory_approval", False)
    if backstory_approval:
        prompt = """You are an eccentric professor with an interesting past. 
        Your job is to summarize the answer to a question based on relevant background context.
        You must weave in your backstory wherever possible.

        Question: {question} 

        Backstory: {backstory}

        Context: {context}

        Answer:"""
        backstory = state["backstory"]
        formatted = prompt.format(question=question, backstory=backstory, context="\n".join([d.page_content for d in documents]))
    else:
        prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
        Your job is to summarize the answer to a question based on relevant background context. 

        Question: {question} 

        Context: {context}

        Answer:"""
        formatted = prompt.format(question=question, context="\n".join([d.page_content for d in documents]))
    generation = await llm.ainvoke([HumanMessage(content=formatted)])
    return {"messages": [generation]}


def make_graph(memory):
    graph = StateGraph(GraphState, input=InputState)

    graph.add_node("trim", trim)

    graph.add_node("search", search, metadata={"category": "docs"})
    graph.add_node("confirm_docs", confirm_docs, metadata={"category": "docs"})

    graph.add_node("gen_backstory", gen_backstory, metadata={"category": "backstory"})
    graph.add_node("confirm_backstory", confirm_backstory, metadata={"category": "backstory"})
    graph.add_node("sleep", sleep, metadata={"category": "backstory"})
    graph.add_node("answer", answer, defer=True)


    graph.add_edge(START, "trim")
    graph.add_edge("trim", "search")
    graph.add_edge("trim", "gen_backstory")

    graph.add_edge("search", "confirm_docs")
    graph.add_edge("confirm_docs", "answer")

    graph.add_edge("gen_backstory", "confirm_backstory")
    graph.add_edge("confirm_backstory", "sleep")
    graph.add_edge("sleep", "answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=memory)


def print_messages(response):
    if isinstance(response, tuple):
        print(response)
    if isinstance(response, dict):
        for key in response:
            if key == "documents":
                print("# of Documents: " + str(len(response["documents"])))
            elif key == "messages":
                print("messages: " + response["messages"][-1].content)
            else:
                print(key + ": " + str(response[key]))


async def run(graph: StateGraph):
    display(Image(graph.get_graph().draw_mermaid_png()))
    state: GraphState = {
        "messages": [],
    }

    thread_id = random.randint(0, 1000000)
    config = {
        "configurable": {
            "thread_id": str(thread_id),
            "checkpoint_ns": "para",
        }
    }
    interrupted = None
    while True:
        if not interrupted:
            user = input('User (q to quit): ')
            if user in {'q', 'Q'}:
                print('AI: Goodbye!')
                break
            state["question"] = user
            turn_input = state
        else:
            print("INTERRUPT==================")
            user = input('User: ')
            resumable = {interrupted.interrupt_id: user}
            turn_input = Command(resume=resumable)
            interrupted = None
        try:
            # Stream responses
            async for output in graph.astream(turn_input, config, stream_mode="updates"):
                if END in output or START in output:
                    continue
                # Print any node outputs
                for key, value in output.items():
                    print_messages(value)
                    if not interrupted and key == "__interrupt__":
                        interrupted = value[0]
        except Exception as e:
            print(f"Error: {str(e)}")
            raise e


async def main():            
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        graph = make_graph(memory)
        await run(graph)

if __name__ == "__main__":
    asyncio.run(main())