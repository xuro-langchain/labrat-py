// agent.ts

import 'dotenv/config'
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver, StateGraph, START, END, type StreamMode,
    Annotation, type Messages, messagesStateReducer } 
    from "@langchain/langgraph";
import { BaseMessage, HumanMessage, 
  AIMessage, AIMessageChunk } from "@langchain/core/messages";
import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";
import * as readline from 'readline';
import { Client } from "langsmith";

// Initialize LangSmith client
const langsmithClient = new Client({
  hideInputs: (inputs) => ({}),
  hideOutputs: (outputs) => ({}),
});
const tracer = new LangChainTracer({
  client: langsmithClient,
});

// Create readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Create a promise-based prompt function
const prompt = (question: string): Promise<string> => {
    return new Promise((resolve) => {
        rl.question(question, (answer) => {
            resolve(answer);
        });
    });
};

// Define the state schema (messages array)
const StateAnnotation = Annotation.Root({
    messages: Annotation<BaseMessage[], Messages>({
      reducer: messagesStateReducer,
    }),
});

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });
const checkpointer = new MemorySaver();

async function chatbot(state: any) {
    const response = await llm.invoke(state.messages);
    return { messages: [...state.messages, response] };
}

const graphBuilder = new StateGraph(StateAnnotation);
graphBuilder
.addNode("chatbot", chatbot)
.addEdge(START, "chatbot")
.addEdge("chatbot", END);

const graph = graphBuilder.compile({checkpointer}).withConfig({
  runName: "scrubbed",
  callbacks: [tracer],
});

async function runChat() {
    const config = {
      configurable: {
        thread_id: "your_thread_id",
      },
    };
    try {
        while (true) {
            const userInput = await prompt("User: ");
            if (!userInput) {
                console.log("Goodbye!");
                break;
            }
            if (["quit", "exit", "q"].includes(userInput.toLowerCase())) {
                console.log("Goodbye!");
                break;
            }
            const inputs = {messages: [new HumanMessage(userInput)]}
            for await (const chunk of await graph.stream(inputs, { ...config, streamMode: ["events", "messages"] as StreamMode[] })) {
              if (chunk[0] instanceof AIMessageChunk) {
                process.stdout.write(String(chunk[0].content));
              } else if (chunk[0] === 'messages' && chunk[1][0] instanceof AIMessageChunk) {
                process.stdout.write(String(chunk[1][0].content));
              }
            }
            console.log(); // Add newline after response
        }
    } finally {
        rl.close();
    }
}
  
runChat();
