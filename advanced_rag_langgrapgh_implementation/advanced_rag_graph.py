from advanced_rag_generate import generation
from advanced_rag_retrieve import retrieve
from advanced_rag_grader_docs import grade_documents
from advanced_rag_state import MessageGraph
from dotenv import load_dotenv
from advanced_rag_websearch import web_search
from advanced_rag_consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from advanced_rag_answer_grader import answer_grader
from advanced_rag_hallucination_grader import hallucination_grader
from advanced_rag_router_prompt import router_chain
from langgraph.graph import StateGraph, START, END
load_dotenv()

# define a functin to go to web_Search or generate based on the router prompt output
def route_to_websearch_or_generate(advanced_rag_state: MessageGraph) -> MessageGraph:
    if advanced_rag_state["web_search"]:
        print("Routing to web search based on document grading...")
        return WEBSEARCH
    else:
        print("Routing to generate based on document grading...")
        return GENERATE

# function to define function to grade based on the hallucination grader output

def route_to_grade_answer_or_hallucination(advanced_rag_state: MessageGraph) -> MessageGraph:
    question = advanced_rag_state["question"]
    generated_answer = advanced_rag_state["generated_answer"]
    documents = advanced_rag_state["documents"]
    hallucination_score = hallucination_grader.invoke(
        {"question": question, "generated_answer": generated_answer, "documents": documents}
    ).binary_score

    if hallucination_score == "yes":
        print("Routing to answer grading based on hallucination grading...")
        score = answer_grader.invoke({"question": question, "generated_answer": generated_answer})
        if score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        print("Routing to hallucination grading...")
        print("---DECISION: GENERATION IS HALLUCINATED---")
        return "not_supported"

def route_question(advanced_rag_state: MessageGraph) -> MessageGraph:
    router_result = router_chain.invoke({"query": advanced_rag_state["question"]})
    return router_result.datasource
    
rag_workflow = StateGraph(MessageGraph)
rag_workflow.add_node(RETRIEVE, retrieve)
rag_workflow.add_node(GRADE_DOCUMENTS, grade_documents)
rag_workflow.add_node(GENERATE, generation)
rag_workflow.add_node(WEBSEARCH, web_search)
rag_workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": WEBSEARCH,
        "retriever": RETRIEVE,
    },
)
rag_workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
rag_workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    route_to_websearch_or_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
rag_workflow.add_conditional_edges(
    GENERATE, route_to_grade_answer_or_hallucination, 
    {
        "not_supported": WEBSEARCH,
        "useful": END,
        "not_useful": WEBSEARCH,
    }
    )
rag_workflow.add_edge(WEBSEARCH, GENERATE)



rag_graph = rag_workflow.compile()
rag_graph.get_graph().draw_mermaid_png(output_file_path="rag_generation.png")