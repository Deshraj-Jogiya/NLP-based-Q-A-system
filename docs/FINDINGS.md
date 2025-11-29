# Findings

Measured results from this repository. Every number below comes from a file in the repository, named under each finding.

## How often does the engine find the right member, message and answer?

**How it was tested:** `eval_qa.py` scores 26 hand-written, paraphrased questions (`eval_set.json`), each labelled with the message
that answers it and a phrase the answer must contain. Subject = spaCy NER with fuzzy first-name matching picked the right member;
retrieval = the top-ranked message (cosine similarity over `all-MiniLM-L6-v2` embeddings) is the labelled one; answer = the returned
answer contains the phrase.

| Measure | Right | Of | Share |
|---|---|---|---|
| Member (subject) | 26 | 26 | 100% |
| Message (retrieval) | 25 | 26 | 96% |
| Answer | 25 | 26 | 96% |

The one miss asks for Lorenzo's Thursday booking; the top-ranked message is his Saturday dinner at The Fat Duck.

**Decision:** CI fails if answer accuracy drops below 90%.

**Skills shown:** sentence-transformers embeddings, cosine-similarity ranking, spaCy NER, fuzzy matching, labelled evaluation set,
CI quality gate.

**Source:** `eval_results.csv`, `eval_set.json`, `eval_qa.py`; the 90% gate is `test_counts.py` (`assert scores["answer_ok"] >= 0.9`),
run in `.github/workflows/ci.yml`.

## What did fixing how counts are read change?

**How it was tested:** the same 26 questions before and after the extraction rules stopped reading dates such as "November 20" as
quantities and learned number words such as "four".

| Run | Answers right | Share |
|---|---|---|
| Before the fix | 22 of 26 | 85% |
| After the fix | 25 of 26 | 96% |

Before the fix, "How many front-row seats does Hans need?" was answered "Hans Müller has 20 item(s)", and two companion questions
about Hans were answered wrongly.

**Decision:** the quantity rules skip dates and accept number words; the evaluation stays in CI so a regression shows up.

**Skills shown:** regex extraction rules, error analysis on labelled questions.

**Source:** `README.md`, `test_counts.py` (the date and number-word cases); the earlier `eval_results.csv` in commit 79e2417 (22 of 26), the current `eval_results.csv` (25 of 26).
