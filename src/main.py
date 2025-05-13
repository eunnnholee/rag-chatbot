# src/main.py

import sys
from pathlib import Path

from src.graph_factory.graph_router import get_graph_by_domain
from src.router.domain_router import route_domain

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    print("===== Multi-Agent Legal Assistant (LangGraph System) =====")
    while True:
        user_query = input("\n💬 User: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting system.")
            break

        # Step 1: router → domain 결정
        domain = route_domain(user_query)
        print(f"[INFO] → routed to domain: {domain.value}")

        # Step 2: domain agent graph 로드
        agent_graph = get_graph_by_domain(domain)

        # Step 3: agent 실행
        result = agent_graph.invoke({"question": user_query})

        # Step 4: 결과 출력
        print("\n🤖 AI Answer:")
        print(result.get("final_answer") or "No answer generated.")


if __name__ == "__main__":
    main()
