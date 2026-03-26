# to grade the retrieved documents list to check its relevant or not

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()


llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=500)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="yes if the retrieved document is relevant to the question, no otherwise."
    )


# with structured output with llm
llm_grader = llm.with_structured_output(GradeDocuments)


system = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
"""


grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved document:\n\n{document}\n\nUser question: {question}",
        ),
    ]
)


retrieve_grader = grader_prompt | llm_grader
