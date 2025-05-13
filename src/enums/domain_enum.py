"""
모든 graph, retriever, prompt, agent에서 도메인 구분용 기준
IDE auto-completion, type hinting, validation에 필수수
"""

from enum import Enum


class LegalDomain(str, Enum):
    """Supported legal domains for multi-agent RAG system."""

    INSURANCE = "employment_insurance"  # 고용보험법
    LABOR = "labor_standards"  # 근로기준법
    GENDER = "gender_equality"  # 남녀고용평등과 일가정 양립 지원에 관한 법률

    def __str__(self) -> str:
        return str(self.value)
