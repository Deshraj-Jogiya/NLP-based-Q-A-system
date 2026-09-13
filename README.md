# Member Messages QA System

This project implements a **question-answering (QA) system** that can answer natural-language questions about member data using messages retrieved from a public API. The system leverages **semantic embeddings**, **named entity recognition (NER)**, and **lightweight inference logic** to provide relevant answers.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [System Architecture](#system-architecture)  
4. [Design Notes and Alternative Approaches](#design-notes-and-alternative-approaches)  
5. [Rationale for Chosen Approach](#rationale-for-chosen-approach)  
6. [Real-Time Market Relevance](#real-time-market-relevance)  
7. [Setup and Installation](#setup-and-installation)  
8. [Usage](#usage)  
9. [Future Improvements](#future-improvements)  
10. [References](#references)  

---

## Project Overview

The goal of this project is to provide an **interactive API endpoint** (`/ask`) where users can submit natural-language questions about members, such as:

- “When is Layla planning her trip to London?”  
- “How many cars does Vikram Desai have?”  
- “What are Amira’s favorite restaurants?”  

The system detects the **subject (member)**, ranks messages based on **semantic similarity**, and dynamically extracts structured information from unstructured text.

**Key Challenge:**  
The messages dataset is **unstructured, concise, and varies in content**, requiring semantic understanding rather than simple keyword matching.

---

## Features

- **Semantic Search:** Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) for robust message ranking.  
- **Named Entity Recognition (NER):** Detects person names from user queries using **spaCy** (`en_core_web_sm`).  
- **Dynamic Information Extraction:**  
  - Emails and phone numbers  
  - Numeric information (tickets, items, reservations)  
  - Group or companion information inferred from message text  
- **Fallback Logic:** Returns closest relevant message if question cannot be mapped to structured data.  
- **Answer Formatting:** Cleans and truncates messages for readability.  
- **Extensible Design:** New users and messages can be added without modifying core logic.

---

## System Architecture

The QA system consists of three main components:

1. **Data Loader**  
   - Dynamically loads JSON messages from the public API.  
   - Detects message content, username, and timestamp columns automatically.  

2. **Embedding & Semantic Search Engine**  
   - Converts all messages into vector embeddings using **SentenceTransformer**.  
   - Incoming questions are embedded and compared with message embeddings using **cosine similarity**.  
   - Returns a ranked list of relevant messages per subject.  

3. **Answer Extraction & Logic**  
   - Identifies the subject using **NER** and fuzzy first-name matching.  
   - Extracts information based on question type:  
     - Contact info → regex extraction  
     - Numeric quantities → regex and pattern detection  
     - Companions → inferred from numbers in text  
   - Fallback returns the most relevant message if no structured info is detected.

---

## Design Notes and Alternative Approaches

During the design phase, several approaches were considered:

### 1. **Embedding-Based Semantic Search (Implemented)**
- **Method:**  
  - Convert all messages into embeddings.  
  - Compute cosine similarity with the question embedding to rank messages.  
- **Pros:**  
  - Handles free-form natural-language questions.  
  - Scalable to medium datasets (hundreds of thousands of messages).  
  - Lightweight and fast; no large LLM inference needed.  
- **Cons:**  
  - May miss facts if relevant message wording differs significantly.  
- **Why Chosen:**  
  - Best balance of **speed, semantic understanding, and maintainability**.  

### 2. **Rule-Based / Keyword Matching**
- **Method:** Predefined rules and regex patterns.  
- **Pros:** Deterministic, fast, easy to debug.  
- **Cons:** Cannot handle paraphrased or ambiguous queries.  
- **Why Not Chosen:** Natural language variation in user queries requires semantic understanding.

### 3. **Knowledge Graph / Structured Database**
- **Method:** Convert messages into structured entities for graph querying.  
- **Pros:** Efficient multi-hop queries, precise info retrieval.  
- **Cons:** High preprocessing cost; less flexible for unstructured messages.  
- **Why Not Chosen:** Dataset is largely unstructured; graph conversion adds unnecessary complexity.

### 4. **Fine-Tuned LLM**
- **Method:** Train or fine-tune transformer models to answer questions directly.  
- **Pros:** High accuracy, can handle paraphrased and complex queries.  
- **Cons:** Expensive, risk of hallucination, requires labeled data.  
- **Why Not Chosen:** Overkill for current dataset size; slower and more costly in real-time scenarios.

### 5. **Hybrid Approach**
- **Method:** Combine embeddings for candidate selection with lightweight LLM or rule engine for extraction.  
- **Pros:** Balances accuracy and speed.  
- **Cons:** Slightly more complex pipeline.  
- **Future Consideration:** Could be applied to larger datasets or enterprise-scale systems.

---

## Rationale for Chosen Approach

- **Real-Time Performance:** Embeddings can be computed quickly, suitable for live queries.  
- **Semantic Understanding:** Captures meaning rather than literal words, handling paraphrasing.  
- **Lightweight and Cost-Effective:** No need for heavy LLM inference for every query.  
- **Scalability:** Adding messages or users requires only re-computing embeddings.  
- **Ease of Deployment:** Simple Python stack; easy to containerize and deploy.  
- **Maintainability:** Minimal maintenance; no complex rules or structured conversions needed.

---

## Real-Time Market Relevance

- **Embedding-based retrieval** is widely used in modern production QA and chatbot systems:  
  - **Slack, Notion AI, Intercom:** Semantic search for documents and messages.  
  - **Customer support systems:** Embeddings + cosine similarity is the standard lightweight method for instant query response.  
- Provides a **good tradeoff between accuracy, cost, and latency**.  
- Future upgrades could integrate hybrid LLM approaches if dataset grows in size or query complexity.

---

## Real RAG Pipeline (`rag_chain.py`)

`llamaindex_search.py` and `semantic_kernel_plugin.py` both do retrieval only -- no
LLM is ever called. `rag_chain.py` is a genuine retrieve-then-generate RAG pipeline,
orchestrated with **LangChain**: it reuses that same retriever, then generates a real
natural-language answer grounded in the retrieved messages using a small, free,
CPU-feasible local model (`HuggingFaceTB/SmolLM2-360M-Instruct`, via LangChain's
`ChatHuggingFace`) -- no API key, no cost.

```bash
python rag_chain.py "who wants to book a flight?"
```

Tests (`test_rag_chain.py`) cover both the retrieval-grounding wiring (fast,
deterministic, via LangChain's own `FakeListLLM` test double) and one real
end-to-end run with the actual local model, proving real generation happens, not
just chain wiring. Building this surfaced two real bugs: the installed
`transformers` version dropped clean pipeline support for encoder-decoder
(seq2seq) models like the originally-planned `flan-t5-small`, and an
instruction-tuned causal model fed a bare completion-style prompt (instead of its
proper chat template) generates nothing useful -- fixed by switching to
`ChatHuggingFace`.
