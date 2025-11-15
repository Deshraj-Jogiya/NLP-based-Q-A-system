import json
import pandas as pd
import numpy as np
import re


# Config

JSON_PATH = "messages.json"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\s\(\)]{7,}\d)\b")
NUM_RE = re.compile(r"\b\d+\b")


# Load data

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data.get("items", data))


# Detect key columns

msg_col = next((c for c in df.columns if "message" in c.lower()), df.columns[0])
user_col = next((c for c in df.columns if "user" in c.lower() and "name" in c.lower()), None)
time_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

print(f"Message Column: {msg_col}")
print(f"User Column: {user_col}")
print(f"Timestamp Column: {time_col}\n")


# Missing Data Analysis

missing_summary = df.isna().sum()
missing_percent = df.isna().mean() * 100
print("=== Missing Data Summary ===")
print(missing_summary)
print("=== Missing Data Percent ===")
print(missing_percent, "\n")


# Duplicate Messages

duplicate_count = df.duplicated(subset=[user_col, msg_col]).sum()
print(f"Duplicate messages detected: {duplicate_count}\n")


# Message Content Analysis

df['msg_length'] = df[msg_col].astype(str).str.len()
print("=== Message Length Stats ===")
print(df['msg_length'].describe(), "\n")


# User Analysis

unique_users = df[user_col].nunique() if user_col else 0
print(f"Unique users: {unique_users}")

msg_per_user = df.groupby(user_col)[msg_col].count().sort_values(ascending=False)
print("Top 10 users by message count:\n", msg_per_user.head(10), "\n")


# Contact Info Extraction

df['emails'] = df[msg_col].str.findall(EMAIL_RE)
df['phones'] = df[msg_col].str.findall(PHONE_RE)

emails_count = df['emails'].apply(lambda x: len(x) > 0).sum()
phones_count = df['phones'].apply(lambda x: len(x) > 0).sum()

print(f"Messages containing emails: {emails_count}")
print(f"Messages containing phones: {phones_count}\n")


# Numeric Data Extraction

df['numbers'] = df[msg_col].str.findall(NUM_RE).apply(lambda x: [int(n) for n in x] if x else [])

all_numbers = [n for sublist in df['numbers'] for n in sublist]
if all_numbers:
    print("=== Numeric Stats in Messages ===")
    print(f"Min: {np.min(all_numbers)}, Max: {np.max(all_numbers)}, Median: {np.median(all_numbers)}\n")


# Temporal Analysis (if timestamp exists)

if time_col:
    df.set_index(time_col, inplace=True)
    daily_counts = df.resample('D')[msg_col].count()
    print("=== Daily Message Counts Stats ===")
    print(daily_counts.describe(), "\n")


# Anomalies & Inconsistencies

# Name inconsistencies
print("=== Name Inconsistencies / Duplicates Check ===")
if user_col:
    user_variants = df[user_col].str.lower().str.strip().value_counts()
    print(user_variants[user_variants > 1].head(10), "\n")

# Sparse users
sparse_users = msg_per_user[msg_per_user <= 2]
print(f"Sparse users (<=2 messages): {len(sparse_users)}\n")


# Summary Table

summary = {
    "Missing usernames": f"{missing_summary.get(user_col, 0)} messages",
    "Missing timestamps": f"{missing_summary.get(time_col, 0)} messages" if time_col else "N/A",
    "Duplicate messages": duplicate_count,
    "Messages with emails": emails_count,
    "Messages with phones": phones_count,
    "Max message length": df['msg_length'].max(),
    "Min/Max numbers": f"{np.min(all_numbers) if all_numbers else 'N/A'} / {np.max(all_numbers) if all_numbers else 'N/A'}",
    "Sparse users (<=2 msgs)": len(sparse_users),
}

print("=== Summary of Key Findings ===")
for k, v in summary.items():
    print(f"{k}: {v}")