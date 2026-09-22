# Tesla RAG Q&A

A small retrieval-augmented question answering system. It retrieves the most relevant fact from a
local knowledge base using TF-IDF and cosine similarity, then asks Gemini to answer the question
grounded in that fact.

```text
question -> retrieve best-matching fact (TF-IDF + cosine similarity) -> Gemini answers using only that fact
```

## Setup

```bash
python3.14 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai scikit-learn
export GEMINI_API_KEY="your-key"
```

## Run

```bash
python rag_qa.py
```

## Example output

```
Question: What is Tesla's mission?
Retrieved fact: Tesla's mission is to accelerate the world's transition to sustainable energy.
Similarity score: 0.3976
Answer: Tesla's mission is to accelerate the world's transition to sustainable energy.

Question: What is the Tesla Model S?
Retrieved fact: The Tesla Model S is a battery-electric luxury sedan.
Similarity score: 0.5679
Answer: Based on the fact provided, the Tesla Model S is a battery-electric luxury sedan.

Question: Who founded Tesla?
Retrieved fact: Tesla, Inc. was founded in 2003 by Martin Eberhard and Marc Tarpenning.
Similarity score: 0.3330
Answer: Based on the fact provided, Tesla was founded by Martin Eberhard and Marc Tarpenning.
```

## Reflection

**What does the similarity score measure?**
It measures how similar the words in the question are to the words in each fact using TF-IDF and cosine similarity.

**Does the highest-scoring fact always contain the complete answer?**
No. It is just the closest match, so it could still be missing information or not fully answer the question.

**How would the system behave if the knowledge base did not contain the answer?**
It would still return the best matching fact, even if the match is bad. The model should then say that the provided fact does not contain the answer instead of making something up.

**Why is retrieving context safer than asking the model to answer from memory alone?**
It gives the model information it can base its answer on and that we can verify. This makes it less likely to hallucinate or use outdated information.

**What would you improve about the retriever or prompt?**
I would add a minimum similarity score so it can say when no relevant fact was found. I could also retrieve multiple facts instead of one or use embeddings to understand meaning better than just matching words.
