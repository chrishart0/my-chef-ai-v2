"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import State
from agent.configuration import Configuration

graph_builder = StateGraph(State)


tool = TavilySearch(max_results=2)
tools = [tool]

def chatbot(state: State, config: RunnableConfig):
    llm_name = config["configurable"].get("llm", "openai:gpt-4.1")
    llm = init_chat_model(llm_name)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

#################
### Tool Node ###

tool_node = ToolNode(tools=[tool])

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


#########################

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

if __name__ == "__main__":
    config = Configuration(llm="openai:gpt-4.1-mini")
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