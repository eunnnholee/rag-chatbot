# src/nodes/common_nodes.py

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.state_models.common_models import (ExtractedInformation,
                                            RefinedQuestion)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class RAGNodeBuilder:
    def __init__(self, retriever, extraction_prompt, rewrite_prompt, answer_prompt):
        self.retriever = retriever
        self.extraction_prompt = extraction_prompt
        self.rewrite_prompt = rewrite_prompt
        self.answer_prompt = answer_prompt

    def retrieve_documents(self):
        def _retrieve(state):
            query = state.rewritten_question or state.question
            print(f"\n[DEBUG] Retrieving documents for query: {query}")
            docs = self.retriever.invoke(query)
            print(f"[DEBUG] Retrieved {len(docs)} documents")
            return {"relevant_docs": docs}

        return _retrieve

    def extract_and_evaluate(self):
        def _extract(state):
            print(f"\n[DEBUG] Starting extraction and evaluation")
            print(f"[DEBUG] Number of documents to process: {len(state.relevant_docs)}")
            extracted_strips = []
            extractor = llm.with_structured_output(schema=ExtractedInformation)

            for i, doc in enumerate(state.relevant_docs):
                print(f"[DEBUG] Processing document {i+1}/{len(state.relevant_docs)}")
                prompt = ChatPromptTemplate.from_template(self.extraction_prompt)
                response = extractor.invoke(
                    prompt.format(
                        question=state.question, document_content=doc.page_content
                    )
                )
                print(
                    f"[DEBUG] Document {i+1} query relevance: {response.query_relevance}"
                )
                if response.query_relevance < 0.8:
                    print(f"[DEBUG] Document {i+1} rejected due to low relevance")
                    continue

                valid_strips = [
                    strip
                    for strip in response.strips
                    if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7
                ]
                print(f"[DEBUG] Document {i+1} valid strips: {len(valid_strips)}")
                extracted_strips.extend(valid_strips)

            print(f"[DEBUG] Total extracted strips: {len(extracted_strips)}")
            return {
                "extracted_info": extracted_strips,
                "num_generations": state.num_generations + 1,
            }

        return _extract

    def rewrite_query(self):
        def _rewrite(state):
            print(f"\n[DEBUG] Starting query rewrite")
            extracted_text = "\n".join(strip.content for strip in state.extracted_info)
            rewriter = llm.with_structured_output(schema=RefinedQuestion)
            prompt = ChatPromptTemplate.from_template(self.rewrite_prompt)
            response = rewriter.invoke(
                prompt.format(question=state.question, extracted_info=extracted_text)
            )
            print(f"[DEBUG] Rewritten query: {response.question_refined}")
            return {"rewritten_question": response.question_refined}

        return _rewrite

    def generate_answer(self):
        def _answer(state):
            print(f"\n[DEBUG] Generating final answer")
            info_str = "\n".join(
                f"Content: {s.content}\nSource: {s.source}\nRelevance: {s.relevance_score}\nFaithfulness: {s.faithfulness_score}"
                for s in state.extracted_info
            )
            prompt = ChatPromptTemplate.from_template(self.answer_prompt)
            response = llm.invoke(
                prompt.format(question=state.question, extracted_info=info_str)
            )
            print(f"[DEBUG] Generated answer: {response.content}")
            return {"final_answer": response.content}

        return _answer


def should_continue(state) -> Literal["continue", "stop"]:
    print(f"\n[DEBUG] Checking if should continue")
    print(f"[DEBUG] Number of generations: {state.num_generations}")
    print(f"[DEBUG] Number of extracted info: {len(state.extracted_info)}")
    if state.num_generations >= 2 or len(state.extracted_info) >= 1:
        print("[DEBUG] Decision: stop")
        return "stop"
    print("[DEBUG] Decision: continue")
    return "continue"
