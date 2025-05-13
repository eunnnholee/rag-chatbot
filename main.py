from rag_engine import RAGEngine

rag = RAGEngine()

while True:
    query = input("질문을 입력하세요 ('exit' 입력 시 종료): ")
    if query.lower() == "exit":
        break
    answer = rag.run(query)
    print("\n[응답]")
    print(answer)
    print("\n" + "-"*50 + "\n")