"""The /ask endpoint answers through the same engine as the command-line loop."""

from fastapi.testclient import TestClient

import main
from api import app

client = TestClient(app)


def test_health_reports_the_loaded_messages():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["messages"] == len(main.flat_messages) > 0


def test_ask_returns_the_engine_answer():
    user = main.known_usernames[0]
    question = f"What did {user.split()[0]} ask for?"
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200
    assert r.json() == main.answer_question(question)


def test_empty_question_is_handled():
    assert client.post("/ask", json={"question": "  "}).json() == {"answer": "Please ask a valid question."}
