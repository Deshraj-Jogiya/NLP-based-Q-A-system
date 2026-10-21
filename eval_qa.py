"""Score the QA engine on a hand-labelled set of paraphrased questions (eval_set.json).

Each item names the message that answers it (its index in messages.json) and a phrase the answer must contain (or a list of
accepted phrases, such as "four" or "4"). Reports subject
accuracy (NER found the right member), retrieval accuracy (the top-ranked message is the labelled one) and answer accuracy (the
returned answer contains the phrase), and writes the per-question results to eval_results.csv.
"""
import csv
import json
import os

import main

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(items):
    rows = []
    for item in items:
        expected = main.flat_messages[item["message"]]
        accepted = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        subject = main.find_subject(item["question"])
        candidates = [m for m in main.flat_messages if m["subject"] == subject]
        ranked = main.rank_messages(item["question"], candidates)
        answer = main.answer_question(item["question"])["answer"]
        rows.append({
            "question": item["question"],
            "subject_ok": subject == expected["subject"],
            "retrieval_ok": bool(ranked) and ranked[0]["idx"] == item["message"],
            "answer_ok": any(p.lower() in answer.lower() for p in accepted),
            "answer": answer,
        })
    return rows


def summarize(rows):
    n = len(rows)
    return {key: sum(r[key] for r in rows) / n for key in ("subject_ok", "retrieval_ok", "answer_ok")}


if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "eval_set.json"), encoding="utf-8") as f:
        items = json.load(f)
    rows = evaluate(items)
    with open(os.path.join(BASE_DIR, "eval_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scores = summarize(rows)
    print(f"{len(rows)} questions: subject {scores['subject_ok']:.0%}, retrieval {scores['retrieval_ok']:.0%}, "
          f"answer {scores['answer_ok']:.0%}")
    for r in rows:
        if not r["answer_ok"]:
            print(f"  wrong: {r['question']} -> {r['answer']}")
