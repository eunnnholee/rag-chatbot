from typing import List

from langchain.schema import Document
from pydantic import BaseModel, Field


class CorrectiveRAGState(BaseModel):
    """Base State model for Corrective RAG flow."""

    question: str = Field(..., description="The user's original question.")
    # generation: str = Field(default="", description="The current generated answer.")
    relevant_docs: List[Document] = Field(
        default_factory=list, description="Documents retrieved from the database."
    )
    num_generations: int = Field(
        default=0, ge=0, le=1, description="Number of generations attempted."
    )

    class Config:
        arbitrary_types_allowed = True
