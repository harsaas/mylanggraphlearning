from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from advanced_rag_state import MessageGraph

load_dotenv()

tavily_search = TavilySearch(max_results=3)

def web_search(state: MessageGraph) -> Dict[str, Any]:
    question = state["question"]
    search_results = tavily_search.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [result["content"] 
        for result in search_results]
    )
    web_search_result_doc = Document(page_content=joined_tavily_result)
    documents = list(state.get("documents") or [])
    documents.append(web_search_result_doc)
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "llm agent memory", "documents": []})