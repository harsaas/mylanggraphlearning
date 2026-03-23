from __future__ import annotations

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    """Schema for the output of the reflexion agent."""
    missing: str = Field(description="missing info from the critique")
    notneeded: str = Field(description="Unnecessary info from the critique")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="Agent answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="Websearching  improvements to address the critique of your current answer."
    )



class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: list[str] = Field(
        description="List of URLs or sources you used to revise your answer based on the critique and search queries."
    )