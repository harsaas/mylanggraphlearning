from typing import Any, TypedDict

class MessageGraph(TypedDict, total=False):
    """ Represent the state of the graph
    
    Attributes:
    Question: Question to LLM
    Answer : Answer generated from LLM
    Websearch: Wether to add any web search
    Documents: Fecth the documents list from the URL's
    """
    question: str
    documents: list[Any]
    web_search: bool
    generated_answer: str

    # Backwards-compatible/legacy keys used in some modules
    answer: str
    websearch: bool


