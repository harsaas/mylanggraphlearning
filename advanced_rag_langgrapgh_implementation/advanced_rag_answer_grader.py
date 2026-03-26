from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from advanced_rag_generate import generation
from advanced_rag_state import MessageGraph


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(model="gpt-5.2", temperature=0, max_tokens=500)
# Use function-calling structured output for compatibility with models that don't support
# OpenAI's Structured Output (json_schema) API.
answer_structured_llm_grader = llm.with_structured_output(
    GradeAnswer, method="function_calling"
)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generated_answer}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | answer_structured_llm_grader