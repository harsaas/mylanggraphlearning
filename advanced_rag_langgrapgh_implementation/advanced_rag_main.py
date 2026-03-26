from dotenv import load_dotenv
from advanced_rag_graph import rag_graph


load_dotenv()

if __name__ == "__main__":
    print("Advanced RAG feature")
    question = "agent memory?"
    result = rag_graph.invoke({"question": question})

    documents = result.get("documents") if isinstance(result, dict) else None
    print(f"Question: {question}")
    if isinstance(documents, list):
        print(f"Documents retrieved: {len(documents)}")

    answer = None
    if isinstance(result, dict):
        answer = result.get("generated_answer") or result.get("answer")

    print("\nAnswer:\n")
    print(answer if answer is not None else result)
