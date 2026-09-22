import os

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "gemini-3.6-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
KNOWLEDGE_BASE_PATH = "tesla_knowledge_base.txt"

QUESTIONS = [
    "What is Tesla's mission?",
    "What is the Tesla Model S?",
    "Who founded Tesla?",
]


def load_facts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def retrieve_best_fact(question, vectorizer, fact_vectors, facts):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, fact_vectors)[0]
    best_index = similarities.argmax()
    return facts[best_index], similarities[best_index]


def build_prompt(question, fact):
    return (
        f"Fact: {fact}\n"
        f"Question: {question}\n\n"
        "Answer the question using only the fact above. "
        "If the fact does not contain the answer, say so explicitly instead of guessing."
    )


def generate_answer(client, question, fact):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(question, fact)}],
    )
    return response.choices[0].message.content


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    facts = load_facts(KNOWLEDGE_BASE_PATH)
    vectorizer = TfidfVectorizer()
    fact_vectors = vectorizer.fit_transform(facts)

    for question in QUESTIONS:
        fact, score = retrieve_best_fact(question, vectorizer, fact_vectors, facts)
        answer = generate_answer(client, question, fact)

        print(f"Question: {question}")
        print(f"Retrieved fact: {fact}")
        print(f"Similarity score: {score:.4f}")
        print(f"Answer: {answer}")
        print()


if __name__ == "__main__":
    main()
