import json
import pandas as pd
import re
from datetime import datetime, timedelta
import calendar
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Load JSON and build DataFrame

json_file = "messages.json"  # replace with your downloaded JSON path

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data["items"], columns=["id", "user_id", "user_name", "timestamp", "message"])
df['timestamp'] = pd.to_datetime(df['timestamp'])


# Step 2: Build embeddings for semantic search

print("Loading embedding model and encoding messages...")
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = list(model.encode(df['message'], convert_to_numpy=True))
print("Embeddings ready.")


# Step 3: Helper functions


def extract_name_partial(question, df):
    """Partial name matching: first name in question matches full name in df"""
    question_lower = question.lower()
    for full_name in df['user_name'].unique():
        first_name = full_name.split()[0].lower()
        if first_name in question_lower:
            return full_name
    return None

def convert_relative_date(text, base_date):
    """
    Convert vague phrases like 'this Friday', 'next Monday', 'first week of December'
    into actual calendar date relative to base_date
    """
    text = text.lower()
    today = base_date.date()
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    # this/next weekday
    m = re.search(r"(this|next)\s+(\w+day)", text)
    if m:
        prefix = m.group(1)
        day = m.group(2)
        target = weekday_map[day]
        days_ahead = target - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        if prefix == "next":
            days_ahead += 7
        return today + timedelta(days=days_ahead)

    # first week of Month
    m2 = re.search(r"first week of (\w+)", text)
    if m2:
        month_name = m2.group(1)
        try:
            month_number = list(calendar.month_name).index(month_name.capitalize())
            return datetime(today.year, month_number, 1).date()
        except:
            return None

    return None

def extract_travel_info(message, timestamp):
    """
    Extract travel/event info: destination, event type, date
    """
    travel_keywords = ['trip', 'flight', 'private jet', 'villa', 'hotel', 'opera', 'Paris', 'London', 'Milan', 'Bali', 'Santorini']
    msg_lower = message.lower()
    found = [word for word in travel_keywords if word.lower() in msg_lower]
    if not found:
        return None

    # Extract destination/event (first keyword match)
    destination = found[-1]  # heuristic: last match is usually destination
    # Extract relative date
    date = convert_relative_date(message, timestamp)
    return destination, date

def extract_count_info(message, item_keywords):
    """
    Extract number of items from message
    """
    message_lower = message.lower()
    for item in item_keywords:
        pattern = r'(\d+)\s+' + re.escape(item)
        m = re.search(pattern, message_lower)
        if m:
            return int(m.group(1)), item
    return None, None


# Step 4: Main QA function

def answer_question(question, df):
    question = question.strip()
    if not question:
        return {"answer": "Please ask a valid question."}

    # Semantic search
    q_emb = model.encode([question], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, np.vstack(df['embedding'].values))[0]
    top_idx = sims.argsort()[::-1][0]
    top_msg = df.iloc[top_idx]

    # Name extraction
    user_name = extract_name_partial(question, df)
    if user_name:
        # Try to pick top message from this user
        user_msgs = df[df['user_name'] == user_name]
        if not user_msgs.empty:
            sims_user = cosine_similarity(q_emb, np.vstack(user_msgs['embedding'].values))[0]
            top_idx_user = sims_user.argmax()
            top_msg = user_msgs.iloc[top_idx_user]
        else:
            return {"answer": f"No clear answer found for {user_name} based on available messages."}

    msg_text = top_msg['message']
    ts = top_msg['timestamp']

    # Check for travel/event info
    travel_info = extract_travel_info(msg_text, ts)
    if travel_info:
        dest, date = travel_info
        date_str = date.strftime("%Y-%m-%d") if date else "unknown date"
        return {"answer": f"{top_msg['user_name']} is going to the {dest} on {date_str}."}

    # Check for count info
    items_to_check = ['car', 'cars', 'ticket', 'tickets', 'reservation', 'reservations']
    count, item = extract_count_info(msg_text, items_to_check)
    if count:
        return {"answer": f"{top_msg['user_name']} has {count} {item}."}

    # Fallback: return cleaned message without time
    msg_clean = re.sub(r'\s+on\s+\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*', '', msg_text)
    return {"answer": f"{top_msg['user_name']} said: \"{msg_clean}\"."}


# Step 5: Terminal loop

if __name__ == "__main__":
    print("Ask any question. Type 'exit' to quit.\n")
    while True:
        q = input("Your question: ")
        if q.lower() == "exit":
            break
        ans = answer_question(q, df)
        print(ans)
