from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Literal
from langgraph import graph
from langgraph.graph import END, START, MessagesState, StateGraph
from node import run_agent_reasoning, tool_node
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()

from reflexion_chain import first_responder_chain, reviser_chain
from reflexion_tools import tools_to_execute

#No of iteratiosn for reflexion agent to run, we can set this to a higher number if we want the agent to have more iterations of reasoning and tool calls before giving a final answer. This will allow the agent to refine its answer based on the feedback it receives from the tools and its own reasoning process.
MAX_ITERATIONS = 2

#define the nodes to add in graph

#draft node the first node to get output from llm 
def draft_node(state: MessagesState):

    """Draft the initial response."""
    response = first_responder_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

#reviser node to get the critique and recommendations for the initial response from llm
def reviser_node(state: MessagesState):
    """Revise the initial response based on the critique and recommendations."""
    messages = state["messages"]
    response = reviser_chain.invoke({"messages": messages})
    return {"messages": [response]}

#define the event loop to run the reflexion agent with the defined nodes and tool calls. The agent will start with the draft node to generate an initial response, then it will go to the reviser node to get critiques and recommendations for that response. Based on the output from the reviser node, it will decide whether to call the tools or to end the process. This loop will continue until the agent has iterated the specified number of times or until it receives a final answer that does not require further tool calls or reasoning steps.
def event_loop(state: MessagesState) -> str:
    """Event loop to run the reflexion agent."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "tools_to_execute"
#If tool messages state is greater than max iterations we set, we end the loop otherwise we continue to execute the tools and get the output from the tools to feed it back to the draft node for next iteration of reasoning and tool calls. This allows the agent to iteratively refine its response based on the feedback it receives from the tools and its own reasoning process, ultimately providing a more accurate and comprehensive answer to the user's query.

#Build the langgraph

reflexion_flow = StateGraph(MessagesState)
reflexion_flow.add_node("draft",draft_node)
reflexion_flow.add_node("revisor",reviser_node)
reflexion_flow.add_node("tools_to_execute",tools_to_execute)
reflexion_flow.add_edge(START,"draft")
reflexion_flow.add_edge("draft","tools_to_execute")
reflexion_flow.add_edge("tools_to_execute","revisor")
reflexion_flow.add_conditional_edges("revisor",event_loop,{"tools_to_execute":"tools_to_execute",END:END})
reflexion_agent_graph = reflexion_flow.compile()

print(reflexion_agent_graph.get_graph().draw_mermaid())

reflexion_agent_response = reflexion_agent_graph.invoke(
    {
        "messages" : [
            {
                "role": "user",
                "content":"What are the health benefits of drinking green tea, will that really reduce belly fat?",
            }
        ]
    }
)

# Extract the final answer from the last message with tool calls

#Checks two things:
#The last message is an AIMessage (so it came from the LLM)
#That AIMessage contains tool calls (OpenAI-style function/tool calling), stored in final_answer.tool_calls
#If both are true, it means the LLM’s “final output” is structured as a tool call payload (like your AnswerQuestion / ReviseAnswer schema),
messages = reflexion_agent_response["messages"]

# Extract the final answer from the last message with tool calls
last_message = reflexion_agent_response["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(reflexion_agent_response)

#TOOL_CALLS [{'name': 'AnswerQuestion', 'args': {'answer': 'Drinking green tea has several health benefits, including:\n\n1. Antioxidant Properties: Green tea is rich in antioxidants called catechins, which can help protect cells from damage and reduce inflammation.\n\n2. Weight Loss: Some studies suggest that green tea can boost metabolism and increase fat burning, which may aid in weight loss, including reducing belly fat.\n\n3. Heart Health: Green tea may help lower cholesterol levels and improve blood flow, reducing the risk of heart disease.\n\n4. Brain Health: The antioxidants in green tea may also have protective effects on the brain, potentially reducing the risk of neurodegenerative diseases.\n\n5. Cancer Prevention: Some research indicates that the antioxidants in green tea may help prevent certain types of cancer by protecting cells from damage.\n\nOverall, while green tea has many potential health benefits, it is not a magic solution for weight loss or belly fat reduction. It should be consumed as part of a balanced diet and healthy lifestyle for best results.'}}]