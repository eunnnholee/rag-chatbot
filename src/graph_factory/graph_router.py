from src.agents.agent_factory import AgentFactory
from src.enums.domain_enum import LegalDomain


def get_graph_by_domain(domain: LegalDomain):
    """
    Main system entrypoint.
    Pass domain enum -> returns fully compiled agent graph.
    """
    return AgentFactory.build_graph(domain)
