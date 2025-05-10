"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from agent.state import State

class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    llm: str



    

graph_builder = StateGraph(State)


llm = init_chat_model("openai:gpt-4.1")

def chatbot(state: State, config: RunnableConfig):
    llm_name = config["configurable"].get("llm", "openai:gpt-4.1")
    llm = init_chat_model(llm_name)
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

if __name__ == "__main__":
    config = {"configurable": {"llm": "openai:gpt-4.1-mini"}}
    messages = []

    def stream_graph_updates(messages, user_input: str, config=None):
        if config is None:
            config = {"configurable": {"llm": "openai:gpt-4.1-mini"}}
        # Add the new user message to the history
        messages.append({"role": "user", "content": user_input})
        for event in graph.stream({"messages": messages}, config=config):
            for value in event.values():
                # Add the assistant's response to the history
                messages.append({"role": "assistant", "content": value["messages"][-1].content})
                print("Assistant:", value["messages"][-1].content)
        return messages

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            messages = stream_graph_updates(messages, user_input, config)
        except:
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            messages = stream_graph_updates(messages, user_input, config)
            break