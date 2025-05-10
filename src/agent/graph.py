"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import State
from agent.configuration import Configuration

tool = TavilySearch(max_results=2)
tools = [tool]

def chatbot(state: State, config: RunnableConfig):
    llm_name = config["configurable"].get("llm", "openai:gpt-4.1")
    llm = init_chat_model(llm_name)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

def build_graph():
    graph_builder = StateGraph(State)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    return graph_builder.compile()

graph = build_graph()

def stream_graph_updates(graph, messages, user_input: str, config=None):
    if config is None:
        config = {"configurable": {"llm": "openai:gpt-4.1-mini"}}
    messages.append({"role": "user", "content": user_input})
    for event in graph.stream({"messages": messages}, config=config):
        for value in event.values():
            messages.append({"role": "assistant", "content": value["messages"][-1].content})
            print("Assistant:", value["messages"][-1].content)
    return messages

def main():
    config = Configuration(llm="openai:gpt-4.1-mini")
    messages = []
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            messages = stream_graph_updates(graph, messages, user_input, config)
        except Exception:
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            messages = stream_graph_updates(graph, messages, user_input, config)
            break

if __name__ == "__main__":
    main()