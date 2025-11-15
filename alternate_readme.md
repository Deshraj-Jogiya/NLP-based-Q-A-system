# Member Messages QA System – Alternative Approach

This project implements an **alternative question-answering (QA) system** for member messages using a combination of **semantic embeddings**, **heuristic-based extraction**, and **relative date processing**. It is designed to answer natural-language questions about member activities, travel plans, and counts of items such as tickets or cars.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [System Architecture](#system-architecture)  
4. [Implementation Details](#implementation-details)  
5. [Comparison to Main Approach](#comparison-to-main-approach)  
6. [Advantages and Limitations](#advantages-and-limitations)  
7. [Setup and Usage](#setup-and-usage)  
8. [Future Improvements](#future-improvements)  
9. [References](#references)  

---

## Project Overview

The goal of this alternative approach is to provide a **more rule-driven QA system** that can:

- Identify the **user** mentioned in the question using **partial first-name matching**.  
- Perform **semantic search** to find the most relevant message.  
- Extract **travel or event information** (destination and relative date).  
- Extract **numerical information** (tickets, cars, reservations).  
- Provide **fallback answers** when no structured info is detected.

This approach adds **heuristic extraction rules** for specific question types, such as "next Monday" or "first week of December", to make answers more context-aware.

---

## Features

- **Partial Name Matching:** Detects users based on first names in the question.  
- **Semantic Search:** Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`) to rank relevant messages.  
- **Relative Date Conversion:** Converts vague temporal phrases (e.g., "this Friday", "next Monday", "first week of December") into calendar dates.  
- **Travel/Event Extraction:** Detects trips, flights, villas, concerts, and opera events in messages.  
- **Count Extraction:** Extracts numeric info for items such as tickets, reservations, and cars.  
- **Fallback Message Retrieval:** Returns the most relevant message if no structured info is found.  

---

## System Architecture

1. **Data Loader:**  
   - Loads messages from JSON and builds a Pandas DataFrame.  
   - Converts timestamp to `datetime` format.  

2. **Embedding & Semantic Search Engine:**  
   - Encodes messages and questions into vector embeddings.  
   - Ranks messages using cosine similarity.  

3. **Heuristic Extraction Engine:**  
   - **Travel/Event Info:** Extracts keywords (e.g., flight, trip, villa, opera) and relative dates.  
   - **Count Info:** Detects numeric references to tickets, cars, or reservations.  
   - **Name Matching:** Partial first-name detection for user-specific messages.  

4. **Answer Formatting & Fallback:**  
   - Cleans messages and formats the final answer.  
   - Provides a readable default message if no structured info matches.

---

## Implementation Details

- **Language & Libraries:** Python, `pandas`, `numpy`, `sentence-transformers`, `sklearn`, `datetime`, `re`.  
- **Embedding Model:** `all-MiniLM-L6-v2` for semantic similarity.  
- **Date Parsing:** Converts relative dates to absolute dates using `datetime` and `calendar`.  
- **Message Analysis:** Keyword-based heuristics detect travel destinations and numeric counts.  

**Sample Outputs:**

| Question | Answer |
|----------|--------|
| `where is layla planning?` | `Layla Kawaguchi is going to the Santorini on 2025-12-01.` |
| `how many tickets does armand need?` | `Armand Dupont has 2 tickets.` |
| `what did sophia receive?` | `Sophia Al-Farsi said: "The concert tickets I received were perfect; thank you for arranging everything."` |

---

## Comparison to Main Approach

| Feature | Main Approach | Alternative Approach |
|---------|---------------|--------------------|
| **Semantic Search** | Yes, full embedding-based ranking | Yes, but combined with heuristics |
| **NER for User Detection** | SpaCy NER + fuzzy matching | Partial first-name matching only |
| **Dynamic Info Extraction** | Regex-based for contact, numbers, companions | Heuristic-based for travel/events and numeric counts |
| **Relative Date Handling** | Not implemented | Yes, converts vague temporal phrases to dates |
| **Complexity** | Lightweight, purely embedding + regex | Slightly more complex with additional heuristics |
| **Accuracy for Travel Queries** | Generic answer from closest message | More precise date & destination extraction |
| **Scalability** | Very scalable; embedding updates only | Scalable, but additional heuristics increase maintenance |
| **Real-Time Suitability** | Excellent, low-latency | Good, but heuristic checks add minor latency |

**Analysis:**  

- The **alternative approach is stronger for travel or event-specific queries**, as it parses relative dates and destinations.  
- The **main approach is more generalizable**, easier to maintain, and better for a variety of question types, including contacts and companions.  
- **Tradeoff:** Alternative approach adds complexity and is slightly less robust for unknown question types or less structured messages.  

---

## Advantages and Limitations

**Advantages:**

- More **context-aware for event/travel planning questions**.  
- Handles **relative date phrases**, which the main approach does not.  
- Can extract **destination + date information** automatically.

**Limitations:**

- Partial name matching is **less robust** than NER; may miss full names or ambiguities.  
- Heuristic rules are **hard-coded**, limiting adaptability to new types of messages or keywords.  
- Slightly more **maintenance-intensive** than the embedding-only approach.  
- Less effective for general queries unrelated to travel, contacts, or numeric counts.