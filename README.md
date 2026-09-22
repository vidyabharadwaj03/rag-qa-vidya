# Lab: Build a Retrieval-Augmented Q&A System

> **Environment reminder:** use a separate Python 3.14 virtual environment for this lab, the same way as in the lessons.
> From this directory, run:
>
> ```bash
> python3.14 -m venv venv
> source venv/bin/activate
> python -m pip install --upgrade pip
> python -m pip install openai scikit-learn
> ```

## Scenario

You are building a small question-answering system about Tesla. The system should
retrieve the most relevant fact from a local knowledge base before asking a language
model to answer the question. This is the basic retrieval-augmented generation
(RAG) pattern:

```text
question → retrieve relevant fact → generate grounded answer
```

The starter knowledge base is provided in `tesla_knowledge_base.txt`. Each line is
one fact that the retriever can return.

## Learning objectives

By completing this lab, you will be able to:

- Load a text knowledge base into Python.
- Use TF-IDF to represent documents and questions as vectors.
- Use cosine similarity to retrieve the best matching fact.
- Pass retrieved context to an LLM.
- Explain why retrieval helps ground an answer and why similarity is not a truth score.

## Your task

Build a Python script that performs the following steps:

1. Load `tesla_knowledge_base.txt`, keeping one fact per list item.
2. Create a `TfidfVectorizer` and fit it to the knowledge-base facts.
3. Write a retrieval function that accepts a question, vectorizes it, calculates
   cosine similarity against every fact, and returns the highest-scoring fact and
   its similarity score.
4. Create an LLM client using the `GEMINI_API_KEY` environment variable. Do not put
   an API key directly in your code.
5. Build a prompt containing the retrieved fact and the original question. Tell the
   model to answer using only the retrieved fact and to say when the fact does not
   contain the answer.
6. Test the complete pipeline with these questions:

   - `What is Tesla's mission?`
   - `What is the Tesla Model S?`
   - `Who founded Tesla?`

7. For every question, print:

   - The question
   - The retrieved fact
   - The cosine similarity score
   - The generated answer

## Setup

Install the required packages (with the `venv` from the top of this page active):

```bash
python -m pip install openai scikit-learn
```

Set your Gemini API key in the terminal:

```bash
export GEMINI_API_KEY="your-key"
```

Use Gemini's OpenAI-compatible endpoint (`base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`)
and a currently available Gemini model, for example `gemini-3.6-flash` (the 2.5 models no longer work).
Keep the model name in your script as a clearly visible configuration value.

## Expected workflow

Your program should follow this structure:

```text
load facts
fit TF-IDF vectorizer
for each question:
    retrieve highest-scoring fact
    send question plus fact to the model
    print retrieval and answer
```

## Reflection

Answer these questions after running your script:

- What does the similarity score measure?
- Does the highest-scoring fact always contain the complete answer?
- How would the system behave if the knowledge base did not contain the answer?
- Why is retrieving context safer than asking the model to answer from memory alone?
- What would you improve about the retriever or prompt?

## Deliverable

Submit:

- Your Python script.
- Terminal output for all three questions.
- Short answers to the reflection questions.

Keep the implementation focused on this retrieval-and-generation workflow. Do not
include API keys in submitted files.

## Extras: Try another dataset

After completing the Tesla example, try replacing the knowledge base with a small
dataset on a topic that interests you. Keep the same format: one searchable fact or
short passage per line.

Useful places to explore:

- [Hugging Face Datasets](https://huggingface.co/datasets) — text, question-answering,
  classification, and summarization datasets.
- [Kaggle Datasets](https://www.kaggle.com/datasets) — public datasets across many
  business, science, and social topics.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/) — classic datasets
  for experimentation and learning.
- [Data.gov](https://www.data.gov/) — public U.S. government datasets.
- [Wikimedia Commons](https://commons.wikimedia.org/) — openly licensed media and
  descriptive information.

Choose a small topic-specific collection, convert the relevant text into the same
line-based format, and test whether retrieval still finds the right context. Document
the dataset source, license, any preprocessing, and at least three questions that you
used to evaluate it.
