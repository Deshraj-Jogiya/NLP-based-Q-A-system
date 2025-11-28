# Member Messages QA System

[![ci](https://github.com/Deshraj-Jogiya/NLP-based-Q-A-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Deshraj-Jogiya/NLP-based-Q-A-system/actions/workflows/ci.yml) ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

Answers natural-language questions about members from their unstructured messages. Questions are matched to messages with
SentenceTransformer embeddings, the person being asked about is found with spaCy NER, and specific details are extracted with rules. The
same data also runs through LlamaIndex, Semantic Kernel and a LangChain RAG chain.

## Highlights

- **Semantic search with an `/ask` API:** ranks messages by cosine similarity over `all-MiniLM-L6-v2` embeddings; `api.py` serves it as
  `POST /ask` (FastAPI) and `main.py` as an interactive prompt.
- **NER first:** spaCy NER with fuzzy first-name matching identifies the subject before ranking.
- **Measured on labelled questions:** `eval_qa.py` scores 26 hand-written, paraphrased questions (`eval_set.json`), each labelled with the
  message that answers it: the right member 100%, the right message 96%, the right answer 96% (85% before counts stopped reading dates
  such as "November 20" as quantities and learned number words such as "four"). CI fails if answer accuracy drops below 90%.
- **Extraction:** regex rules pull contact details, quantities and companion/group references, falling back to the closest message.
- **Approach chosen on purpose:** five designs were compared (rule-based matching, knowledge graphs, fine-tuned LLMs, hybrid retrieval,
  embeddings); see `alternate_readme.md`.
- **Three more paths over the same data:** a LlamaIndex `VectorStoreIndex` retriever, Semantic Kernel plugin functions orchestrated by a
  Kernel, and a retrieve-then-generate LangChain RAG chain running a local model.

Measured results, each with the question, the test and the decision: [docs/FINDINGS.md](docs/FINDINGS.md).

## Architecture

```mermaid
flowchart LR
    Q[question] --> N[spaCy NER<br/>who is it about?]
    N --> E[SentenceTransformer<br/>embeddings]
    E --> R[rank messages<br/>cosine similarity]
    R --> X[rule-based extraction]
    X --> A[answer]
    R -. alternative .-> RAG[LangChain RAG<br/>local model]
    R -. alternative .-> LI[LlamaIndex retriever]
    N -. alternative .-> SK[Semantic Kernel plugins]
```

## Stack

Python, Sentence-Transformers, spaCy, scikit-learn, pandas, LlamaIndex, Semantic Kernel, LangChain, FastAPI, pytest, GitHub Actions.

## Run it

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python main.py                          # interactive prompt
uvicorn api:app --port 8000             # POST /ask {"question": "..."}
python eval_qa.py                       # score the labelled questions
pytest test_api.py test_llamaindex_search.py test_semantic_kernel_plugin.py test_rag_chain.py test_counts.py -v
```

## Data

`messages.json` holds 100 member messages from 10 members (one page of a 3,349-message feed). `data_insight_README.MD` lists the
anomalies and inconsistencies found in it.
