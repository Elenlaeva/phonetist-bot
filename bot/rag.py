import os

KNOWLEDGE_BASE_PATH = "knowledge_base"

def load_documents():
    documents = []
    for root, _, files in os.walk(KNOWLEDGE_BASE_PATH):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
    return documents


def find_relevant_text(query: str, documents: list[str]) -> str:
    query = query.lower()
    scored = []

    for doc in documents:
        score = sum(1 for word in query.split() if word in doc.lower())
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])

    if scored and scored[0][0] > 0:
        return scored[0][1]

    return "К сожалению, информация по данному вопросу не найдена в базе знаний."
