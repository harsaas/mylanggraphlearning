from dotenv import load_dotenv
import os
from pathlib import Path
from langgraph.graph import END, MessagesState, StateGraph
from node import run_agent_reasoning, tool_node
from langchain_core.messages import HumanMessage


load_dotenv()

#constant for agent reasoning
AGENT_REASON = "agent_reason"
ACT = "act"


def should_continue(state: MessagesState) -> str:
    """Route to tool node if the last AI message contains tool calls."""
    messages = state["messages"]
    if not messages:
        return END

    last = messages[-1]

    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return ACT

    additional = getattr(last, "additional_kwargs", None) or {}
    if additional.get("tool_calls"):
        return ACT

    return END


flow = StateGraph(MessagesState)

# Define the flow of the graph
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

flow.set_entry_point(AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue, {ACT: ACT, END: END})
flow.add_edge(ACT, AGENT_REASON)

graph = flow.compile()
#print the graph to visualize the flow of the agent reasoning and tool calls. This will help us understand how the agent processes the input and when it decides to call tools based on the messages it generates.
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    out_path = Path(__file__).with_name("agent_flow.png")
    out_path.write_bytes(png_bytes)
except Exception as e:
    print(f"Skipping graph PNG render: {e}")


if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling!")
    res = graph.invoke({"messages": [HumanMessage(content="What is the temperature in Dallas Texas? List it as table and then triple it")]})  
    print(res["messages"][-1].content) 