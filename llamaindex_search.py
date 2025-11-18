"""Semantic search over the member messages dataset using LlamaIndex,
as an alternative retrieval path to main.py's raw
SentenceTransformer + cosine_similarity implementation. Uses the same
embedding model (all-MiniLM-L6-v2) via LlamaIndex's HuggingFace
embedding integration and a local, in-memory vector index -- no LLM
API key or external service required for retrieval.

Run it directly:
    python llamaindex_search.py "book a flight"
"""

import json
import sys

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

JSON_PATH = "messages.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_messages(json_path: str = JSON_PATH) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", data)


def build_index(messages: list[dict]) -> VectorStoreIndex:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    documents = [
        Document(
            text=item["message"],
            metadata={"id": item["id"], "user_name": item.get("user_name", "")},
        )
        for item in messages
        if item.get("message")
    ]
    return VectorStoreIndex.from_documents(documents)


def search(index: VectorStoreIndex, query: str, top_k: int = 5) -> list[dict]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [
        {
            "score": round(node.score, 4),
            "message": node.node.get_content(),
            "user_name": node.node.metadata.get("user_name", ""),
        }
        for node in nodes
    ]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "book a flight"
    messages = load_messages()
    print(f"[*] Building LlamaIndex vector index over {len(messages)} messages...")
    index = build_index(messages)

    print(f"[*] Query: {query!r}\n")
    for i, result in enumerate(search(index, query), start=1):
        print(f"{i}. (score={result['score']}) {result['user_name']}: {result['message']}")


if __name__ == "__main__":
    main()
