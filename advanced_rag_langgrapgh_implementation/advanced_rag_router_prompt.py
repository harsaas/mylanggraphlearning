from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from advanced_rag_generate import generation
from advanced_rag_state import MessageGraph

class RouterPromptInput(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["retriever", "web_search"] = Field(...,
        description="The datasource to route the user query to. 'retriever' means the question will be routed to the retrieve step, and the retriever will retrieve relevant documents from a vector database. 'web_search' means the question will be routed to the web search step, and a web search will be performed to retrieve relevant information."
    )

llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=500)
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a router deciding which datasource is more relevant for a user query. The two datasources are: 1) a retriever that retrieves documents from a vector database, and 2) a web search tool that retrieves information from the web. If the user query is more likely to be answered by information in the vector database, route to the retriever. If the user query is more likely to be answered by information from the web, route to the web search tool."),
        ("human", "User query: {query} \n\n Route the query to the most relevant datasource by answering with either 'retriever' or 'web_search'."),
    ]
)

router_chain = router_prompt | llm.with_structured_output(RouterPromptInput)