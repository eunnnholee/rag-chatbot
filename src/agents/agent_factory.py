from langgraph.graph import END, START, StateGraph

from src.enums.domain_enum import LegalDomain
from src.nodes.common_nodes import RAGNodeBuilder, should_continue
from src.prompts import (GENDER_ANSWER_PROMPT, GENDER_EXTRACTION_PROMPT,
                         GENDER_REWRITE_PROMPT, INSURANCE_ANSWER_PROMPT,
                         INSURANCE_EXTRACTION_PROMPT, INSURANCE_REWRITE_PROMPT,
                         LABOR_ANSWER_PROMPT, LABOR_EXTRACTION_PROMPT,
                         LABOR_REWRITE_PROMPT)
from src.retrievers import (get_gender_retriever, get_insurance_retriever,
                            get_labor_retriever)
from src.state_models import (EmploymentInsuranceState, GenderEqualityState,
                              LaborStandardsState)


class AgentFactory:
    @staticmethod
    def build_graph(domain: LegalDomain):
        if domain == LegalDomain.INSURANCE:
            retriever = get_insurance_retriever()
            state_cls = EmploymentInsuranceState
            prompts = (
                INSURANCE_EXTRACTION_PROMPT,
                INSURANCE_REWRITE_PROMPT,
                INSURANCE_ANSWER_PROMPT,
            )
        elif domain == LegalDomain.LABOR:
            retriever = get_labor_retriever()
            state_cls = LaborStandardsState
            prompts = (
                LABOR_EXTRACTION_PROMPT,
                LABOR_REWRITE_PROMPT,
                LABOR_ANSWER_PROMPT,
            )
        elif domain == LegalDomain.GENDER:
            retriever = get_gender_retriever()
            state_cls = GenderEqualityState
            prompts = (
                GENDER_EXTRACTION_PROMPT,
                GENDER_REWRITE_PROMPT,
                GENDER_ANSWER_PROMPT,
            )

        nodes = RAGNodeBuilder(
            retriever=retriever,
            extraction_prompt=prompts[0],
            rewrite_prompt=prompts[1],
            answer_prompt=prompts[2],
        )

        graph = StateGraph(state_schema=state_cls)
        graph.add_node("retrieve", nodes.retrieve_documents())
        graph.add_node("extract_and_evaluate", nodes.extract_and_evaluate())
        graph.add_node("rewrite_query", nodes.rewrite_query())
        graph.add_node("generate_answer", nodes.generate_answer())

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "extract_and_evaluate")
        graph.add_conditional_edges(
            "extract_and_evaluate",
            should_continue,
            {"continue": "rewrite_query", "stop": "generate_answer"},
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", END)

        return graph.compile()
