from typing import Any, Dict

from advanced_rag_ingestion import retriever
from advanced_rag_state import MessageGraph


def retrieve(advanced_rag_state: MessageGraph) -> Dict[str, Any]:
    print("Retrieve the documents")
    question = advanced_rag_state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
