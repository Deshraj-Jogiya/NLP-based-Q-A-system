"""Tests the RAG chain's retrieval-grounding wiring (fast, deterministic,
via LangChain's own FakeListLLM test double -- no model download) and,
separately, does one real end-to-end run with the actual local model
to prove real generation happens, not just wiring."""

from langchain_core.language_models.fake import FakeListLLM

from llamaindex_search import build_index
from rag_chain import answer_question, build_rag_chain, load_local_llm


def _fixture_messages():
    return [
        {"id": 1, "user_name": "Alice", "message": "I want to book a flight to Paris next week."},
        {"id": 2, "user_name": "Bob", "message": "Can someone recommend a good pizza place downtown?"},
        {"id": 3, "user_name": "Carol", "message": "Looking for a flight from NYC to Paris in June."},
    ]


def test_chain_grounds_the_fake_llm_response_in_real_retrieved_context():
    index = build_index(_fixture_messages())
    fake_llm = FakeListLLM(responses=["Alice and Carol are asking about flights to Paris."])
    chain = build_rag_chain(fake_llm)

    result = answer_question(index, chain, "who is asking about flights?", top_k=2)

    assert result["answer"] == "Alice and Carol are asking about flights to Paris."
    retrieved_names = {s["user_name"] for s in result["sources"]}
    assert retrieved_names == {"Alice", "Carol"}


def test_chain_does_not_retrieve_unrelated_messages_as_top_hits():
    index = build_index(_fixture_messages())
    fake_llm = FakeListLLM(responses=["placeholder"])
    chain = build_rag_chain(fake_llm)

    result = answer_question(index, chain, "flight to Paris", top_k=1)

    assert result["sources"][0]["user_name"] in {"Alice", "Carol"}


def test_real_local_model_generates_a_real_grounded_answer():
    """No fake, no mock: loads the real local flan-t5-small model and
    checks it produces a real, non-empty answer grounded in the real
    retrieved context -- the one test in this file that proves actual
    generation works, not just chain wiring."""
    index = build_index(_fixture_messages())
    llm = load_local_llm()
    chain = build_rag_chain(llm)

    result = answer_question(index, chain, "Who wants to go to Paris?", top_k=2)

    assert result["answer"]
    assert "paris" in result["answer"].lower() or any(
        s["user_name"].lower() in result["answer"].lower() for s in result["sources"]
    )
