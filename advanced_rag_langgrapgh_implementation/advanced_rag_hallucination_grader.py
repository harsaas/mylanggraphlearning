from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=500)
from dotenv import load_dotenv
load_dotenv()
from advanced_rag_generate import generation
from advanced_rag_state import MessageGraph

class HallucinationGrade(BaseModel):
    """Binary score for hallucination check on generated answer."""

    binary_score: Literal["yes", "no"] = Field(
        description="yes if the generated answer is relevant to the question and the retrieved documents, no otherwise."
    )

# Use function-calling structured output for compatibility with models that don't support
# OpenAI's Structured Output (json_schema) API.
hallucination_structured_llm_grader = llm.with_structured_output(
    HallucinationGrade, method="function_calling"
)
system = """You are a grader assessing whether an LLM generation is grounded in  supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in  supported by the set of facts. 'No' means that the answer is not grounded in / supported by the set of facts, and is likely hallucinated."""
hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "User question: \n\n {question} \n\n Retrieved documents: \n\n {documents} \n\n LLM generation: {generated_answer}",
        ),
    ]
)
hallucination_grader: RunnableSequence = hallucination_grader_prompt | hallucination_structured_llm_grader