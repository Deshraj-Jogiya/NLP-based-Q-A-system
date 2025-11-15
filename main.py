import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re

# Config
JSON_PATH = "messages.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# Load dataset dynamically
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("items", data)
df = pd.DataFrame(items)

# dynamically detect columns
msg_col = next((c for c in df.columns if "message" in c.lower()), df.columns[0])
user_col = next((c for c in df.columns if "user" in c.lower() and "name" in c.lower()), None)
time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

df[msg_col] = df[msg_col].astype(str)
known_usernames = df[user_col].dropna().unique().tolist() if user_col else []

# Load models
print("Loading models...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
nlp = spacy.load(SPACY_MODEL)
print("Models loaded.")

# compute embeddings
df["embedding"] = list(embedder.encode(df[msg_col].tolist(), convert_to_numpy=True))
flat_messages = [{
    "idx": i,
    "subject": (row[user_col] if user_col and pd.notna(row[user_col]) else "__UNKNOWN__"),
    "message": row[msg_col],
    "embedding": row["embedding"]
} for i, row in df.reset_index().iterrows()]

# Regex patterns
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b")
NUM_RE = re.compile(r"\b\d+\b")

# Helper functions
def map_person_to_username(person_text):
    """Map PERSON entity to closest username"""
    if not known_usernames:
        return None
    p = person_text.strip().lower()
    for name in known_usernames:
        if p in str(name).lower() or str(name).lower() in p:
            return name
    return None

def find_subject(question):
    """Detect subject from question using NER or first name match"""
    doc = nlp(question)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            mapped = map_person_to_username(ent.text)
            if mapped:
                return mapped

    # fallback: first name match, robust to punctuation
    q_clean = re.sub(r"[^a-zA-Z\s]", " ", question).lower()
    tokens = q_clean.split()
    for name in known_usernames:
        first = str(name).split()[0].lower()
        if first in tokens:
            return name
    return None

def extract_contact(text):
    """Extract email/phone from message"""
    emails = EMAIL_RE.findall(text)
    phones = [m.group(0).strip() for m in PHONE_RE.finditer(text)]
    return emails + phones

def infer_companions(message):
    """Infer if person is going with someone based on numbers"""
    nums = [int(n) for n in NUM_RE.findall(message)]
    total_people = max(nums) if nums else 1
    if total_people > 1:
        return f"Yes, going with {total_people - 1} person(s)."
    else:
        return "No, going alone."

def rank_messages(question, messages):
    """Rank all messages for the user based on semantic similarity (no top-N limit)."""
    if not messages:
        return []
    q_emb = embedder.encode([question], convert_to_numpy=True)
    cand_emb = np.vstack([m["embedding"] for m in messages])
    sims = cosine_similarity(q_emb, cand_emb)[0]
    order = sims.argsort()[::-1]  # rank all messages
    return [messages[i] for i in order]

def clean_text(text):
    text = text.strip()
    if len(text) > 250:
        text = text[:247] + "..."
    return text

# Main QA function
def answer_question(question):
    q = question.strip()
    if not q:
        return {"answer": "Please ask a valid question."}

    subject = find_subject(q)

    # STRICT user validation
    if subject is None:
        return {"answer": "No user found in your question."}
    if subject not in known_usernames:
        return {"answer": f"No user named '{subject}' found in the dataset."}

    # Only messages for this subject
    candidates = [m for m in flat_messages if m["subject"] == subject]
    ranked = rank_messages(q, candidates)
    if not ranked:
        return {"answer": "No relevant information found for this user."}

    top_msg = ranked[0]
    msg_text = top_msg["message"]

    # Dynamic inference
    q_lower = q.lower()
    if any(k in q_lower for k in ["contact", "phone", "email", "number"]):
        contacts = extract_contact(msg_text)
        if contacts:
            return {"answer": f"{subject}'s contact: {contacts[0]}"}
        else:
            return {"answer": f"No contact info found for {subject}."}

    if any(k in q_lower for k in ["how many", "tickets", "reservations"]):
        nums = [int(n) for n in NUM_RE.findall(msg_text)]
        if nums:
            return {"answer": f"{subject} has {nums[0]} item(s)."}

    if any(k in q_lower for k in ["someone", "with"]):
        return {"answer": f"{subject}: {infer_companions(msg_text)}"}

    # fallback: return cleaned message only
    return {"answer": f"{subject}: {clean_text(msg_text)}"}

# Interactive loop

if __name__ == "__main__":
    print("QA is ready. Type 'exit' to quit.")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print(answer_question(q))