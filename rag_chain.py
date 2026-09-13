"""A real RAG (Retrieval-Augmented Generation) pipeline, orchestrated
with LangChain, over the same member-messages dataset and retriever
llamaindex_search.py already builds.

This is a genuine retrieve-then-generate pipeline -- unlike
llamaindex_search.py and semantic_kernel_plugin.py, which only do
retrieval (no LLM call), this actually generates a natural-language
answer grounded in the retrieved messages. Uses a small, free,
CPU-feasible local HuggingFace model (HuggingFaceTB/SmolLM2-360M-Instruct)
via LangChain's ChatHuggingFace wrapper, so it's real generation with no
API key and no cost -- not a mock.

Run it directly:
    python rag_chain.py "who wants to book a flight?"
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from llamaindex_search import build_index, load_messages, search

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If the context doesn't contain the answer, say you don't know."
)


def load_local_llm() -> ChatHuggingFace:
    """A small (360M param), instruction-tuned, CPU-feasible causal model --
    picked specifically so this pipeline needs no API key, no GPU, and no
    paid service to run for real.

    Wrapped in ChatHuggingFace rather than used as a raw HuggingFacePipeline
    completion model: instruction-tuned models are fine-tuned against a
    specific chat template (system/user/assistant turns), and feeding one a
    bare "Context:...Question:...Answer:" completion prompt instead of that
    template produces an empty/garbage response -- a real bug this
    implementation hit and fixed."""
    pipeline = HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        task="text-generation",
        # return_full_text=False: the "text-generation" pipeline's default
        # echoes the whole input prompt back concatenated with the new
        # tokens -- without this, the response includes the prompt itself.
        pipeline_kwargs={"max_new_tokens": 128, "return_full_text": False},
    )
    return ChatHuggingFace(llm=pipeline)


def build_rag_chain(llm) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    return prompt | llm


def answer_question(index, chain: Runnable, question: str, top_k: int = 3) -> dict:
    """Retrieves real context via the LlamaIndex retriever, then generates
    a grounded answer with the LangChain chain. Returns the sources
    alongside the answer -- transparency over a black-box response."""
    sources = search(index, question, top_k=top_k)
    context = "\n".join(f"- {s['user_name']}: {s['message']}" for s in sources)

    response = chain.invoke({"context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)

    return {"answer": answer.strip(), "sources": sources}


def main() -> None:
    question = " ".join(sys.argv[1:]) or "who wants to book a flight?"
    messages = load_messages()
    print(f"[*] Building retrieval index over {len(messages)} messages...")
    index = build_index(messages)

    print("[*] Loading local SmolLM2-360M-Instruct for generation (first run downloads the model)...")
    llm = load_local_llm()
    chain = build_rag_chain(llm)

    result = answer_question(index, chain, question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print("\nSources used:")
    for s in result["sources"]:
        print(f"  - (score={s['score']}) {s['user_name']}: {s['message']}")


if __name__ == "__main__":
    main()
