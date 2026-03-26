from typing import Any, Dict

from advanced_rag_generation_chain import generation_chain
from advanced_rag_state import MessageGraph

def generation(advanced_rag_state: MessageGraph) -> Dict[str, any]:
    question = advanced_rag_state["question"]
    documents = advanced_rag_state["documents"]

    context = "\n\n".join(
        getattr(doc, "page_content", str(doc)) for doc in (documents or [])
    )
    generated_answer = generation_chain.invoke({"question": question, "context": context})
    return {"generated_answer": generated_answer, "question": question, "documents": documents}