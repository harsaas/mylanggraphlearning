from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num: float) -> float:
    """Tool to triple a number."""
    return num * 3

#list of tools to be used by the agent, we can add more tools as needed. These tools will be available for the agent to use when generating responses. The agent can call these tools to perform specific actions or retrieve information as part of its reasoning process.
tools = [TavilySearch(max_results=10), triple]

llm = init_chat_model(model="gpt-5.2", model_provider="openai").bind_tools(tools)
