from llamaindex_search import build_index, search


def test_semantic_search_finds_relevant_message():
    messages = [
        {"id": "1", "user_name": "A", "message": "Please book a flight to Tokyo for Friday."},
        {"id": "2", "user_name": "B", "message": "What's the weather like tomorrow?"},
        {"id": "3", "user_name": "C", "message": "Can you cancel my dentist appointment?"},
    ]
    index = build_index(messages)
    results = search(index, "reserve an airline ticket", top_k=1)
    assert len(results) == 1
    assert "flight" in results[0]["message"].lower()
