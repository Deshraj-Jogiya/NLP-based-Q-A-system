import json

import eval_qa
from main import extract_counts, infer_companions


def test_counts_ignore_dates_and_read_number_words():
    assert extract_counts("I need four front-row seats for the game on November 20.") == [4]
    assert extract_counts("Book two VIP tickets to the Broadway show on December 3rd.") == [2]


def test_companions_come_from_the_party_size():
    assert infer_companions("Book a dinner reservation for eight at Blue Hill for this Friday.") == "Yes, going with 7 person(s)."
    assert infer_companions("I'm flying to San Francisco, book the first class for two on November 10.") == "Yes, going with 1 person(s)."
    assert infer_companions("Please book a private jet to Paris for this Friday.") == "No, going alone."


def test_labelled_questions_stay_above_ninety_percent():
    with open("eval_set.json", encoding="utf-8") as f:
        scores = eval_qa.summarize(eval_qa.evaluate(json.load(f)))
    assert scores["answer_ok"] >= 0.9
