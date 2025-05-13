from typing import List

from pydantic import BaseModel, Field


class InformationItem(BaseModel):
    content: str = Field(..., description="Extracted information content.")
    source: str = Field(..., description="Source of the information.")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1.")
    faithfulness_score: float = Field(
        ..., description="Faithfulness score between 0 and 1."
    )


class ExtractedInformation(BaseModel):
    strips: List[InformationItem] = Field(
        ..., description="List of extracted information items."
    )
    query_relevance: float = Field(
        ..., description="Overall confidence score between 0 and 1."
    )


class RefinedQuestion(BaseModel):
    """Refined user question with justification."""

    question_refined: str = Field(
        ..., description="Refined version of the original question."
    )
    reason: str = Field(..., description="Explanation why the question was refined.")
