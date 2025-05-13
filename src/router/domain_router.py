from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.enums.domain_enum import LegalDomain

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class DomainRoutingOutput(BaseModel):
    domain: Literal["INSURANCE", "LABOR", "GENDER"] = Field(
        description="Classify query. Choose ONLY one: INSURANCE (고용보험법), LABOR (근로기준법), GENDER (남녀고용평등과 일·가정 양립 지원에 관한 법률)"
    )


def route_domain(query: str) -> LegalDomain:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a domain classification expert.  
            Given a user query, classify which law domain it belongs to.
            Choose ONLY ONE of these categories:
            1. INSURANCE → Employment Insurance Act (고용보험법)
            2. LABOR → Labor Standards Act (근로기준법)
            3. GENDER → Gender Equality Employment Act (남녀고용평등과 일·가정 양립 지원에 관한 법률)
            """,
            ),
            ("human", "{query}"),
        ]
    )

    classifier = llm.with_structured_output(schema=DomainRoutingOutput)
    result: DomainRoutingOutput = classifier.invoke(prompt.format(query=query))

    return LegalDomain[result.domain]
