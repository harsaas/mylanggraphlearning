from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import tool , StructuredTool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from reflexion_agent_output_schema import AnswerQuestion, Reflection , ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs) -> list[dict[str, str]]:
    """Tool to run queries using the TavilySearch tool."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

tools_to_execute = ToolNode(
    [StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__, description="Tool to answer questions using the TavilySearch tool. Takes a list of search queries and returns a list of results for each query."),
    StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__, description="Tool to revise answers using the TavilySearch tool. Takes a list of search queries and returns a list of results for each query.")],
    
)
#Same search engire is used for answer question phase initial call to llm
#Also during the revise answer phase to llm based on response from llm with critique 
#so same search tool run_queries is used and called two times with diff names in the ToolNode
