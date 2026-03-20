from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Annotated, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from chains import generativechain , reflectionchain
from langgraph.graph.message import add_messages

load_dotenv()
#Start -> Generate -> should continue -> Generate -> should continue -> Generate -> End
# crtique as input to Generate until the finite output received from generate as part of REFLECTION agent. The agent will keep generating responses and critiquing them until it receives a final answer that does not require further tool calls or reasoning steps. This allows the agent to iteratively refine its response based on the retrieved context and the tools available, ultimately providing a more accurate and comprehensive answer to the user's query.
#add_message is a reducer in langgraph that allows us to add messages to the state. This is useful for keeping track of the conversation history and allowing the agent to have access to previous messages when generating responses. By using add_message, we can ensure that the agent has the necessary context to generate informed critiques and recommendations for the user's tweet, as well as to generate a viral tweet based on those critiques and recommendations.
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int

REFLECT = "reflect"
GENERATE = "generate"

#Create the genaration node for the agent to generate a response based on the critique and recommendations provided by the reflection node. The generation node will use the generativechain defined in chains.py to generate a response based on the input it receives from the reflection node. This allows the agent to iteratively refine its response based on the critiques and recommendations provided, ultimately generating a viral tweet that meets the user's request.
def generate_node(state: MessageGraph):
    messages = state["messages"]
    response = generativechain.invoke({"messages": messages})
    iteration = int(state.get("iteration", 0)) + 1
    return {"messages": [response], "iteration": iteration}

#Create the reflection node for the agent to generate critiques and recommendations based on the user's tweet. The reflection node will use the reflectionchain defined in chains.py to generate critiques and recommendations based on the input it receives from the user. This allows the agent to provide detailed feedback on the user's tweet, including suggestions for length, virality, style, etc., which can then be used by the generation node to create a more viral tweet.
def reflection_node(state: MessageGraph):
    messages = state["messages"]
    response = reflectionchain.invoke({"messages": messages})
    return {"messages": [HumanMessage(content=getattr(response, "content", str(response)))]}
#Trick : we casting the message response from reflection as Human message 
# so LLM will think its generated from HUMAN CRITIC so it generates better response

flow_builder = StateGraph(MessageGraph)
flow_builder.add_node(REFLECT, reflection_node)
flow_builder.add_node(GENERATE, generate_node)
flow_builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph) -> str:
    """After each generation, go to reflection until max rounds reached."""
    max_rounds = 3
    iteration = int(state.get("iteration", 0))
    if iteration >= max_rounds:
        return END
    return REFLECT

flow_builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
flow_builder.add_edge(REFLECT, GENERATE)
relector_agent_graph = flow_builder.compile()

# print the graph to visualize the flow of the reflection agent.
try:
    png_bytes = relector_agent_graph.get_graph().draw_mermaid_png()
    out_path = Path(__file__).with_name("Reflectionagent_flow.png")
    out_path.write_bytes(png_bytes)
except Exception as e:
    print(f"Skipping graph PNG render: {e}")

if __name__ == "__main__":
    print("Hello LangGraph with Reflection agent!")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ],
        "iteration": 0,
    }
    response = relector_agent_graph.invoke(inputs)
    messages = response.get("messages", [])
    if messages:
        print(messages[-1].content)
    else:
        print(response)
