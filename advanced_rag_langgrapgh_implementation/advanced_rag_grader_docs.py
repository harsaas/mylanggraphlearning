from typing import Any, Dict
from dotenv import load_dotenv
from advanced_rag_retrievegrader import retrieve_grader
from advanced_rag_state import MessageGraph
load_dotenv()
def grade_documents(state: MessageGraph) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieve_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = True
            continue
   
    return {"documents": filtered_docs, "question": question, "web_search": web_search}