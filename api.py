"""HTTP front end for the question-answering engine in main.py: POST /ask with {"question": "..."}."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

import main

app = FastAPI(title="Member Messages QA")


class Question(BaseModel):
    question: str = Field(..., max_length=500)


@app.post("/ask")
def ask(body: Question) -> dict:
    return main.answer_question(body.question)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "messages": len(main.flat_messages)}
