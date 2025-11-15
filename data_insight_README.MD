# Bonus Goal 2 – Member Messages Data Insights

This document summarizes the **data analysis conducted on the member messages dataset** (`messages.json`) to identify anomalies, inconsistencies, and insights that can help improve the QA system and overall data quality.

---

## Table of Contents

1. [Objective](#objective)  
2. [Dataset Overview](#dataset-overview)  
3. [Analysis Methodology](#analysis-methodology)  
4. [Findings & Results](#findings--results)  
    - Missing Data  
    - Duplicate Messages  
    - Message Length  
    - User Activity  
    - Contact Information  
    - Numeric Information  
    - Temporal Analysis  
    - Name Consistency  
5. [Summary of Key Insights](#summary-of-key-insights)  
6. [Recommendations](#recommendations)  
7. [Conclusion](#conclusion)  

---

## Objective

The goal of this analysis was to **perform a professional-grade exploratory data analysis (EDA)** on the unstructured messages dataset. Specifically, we wanted to:

- Detect **missing or incomplete data**.  
- Identify **duplicate messages**.  
- Analyze **message length and content structure**.  
- Detect **user activity patterns** and sparse contributors.  
- Extract potential **contact information and numeric data** for QA purposes.  
- Identify **temporal patterns and anomalies**.  
- Highlight **name inconsistencies** for reliable NER-based QA.  

This ensures the QA system can **accurately answer questions** and handle anomalies gracefully.

---

## Dataset Overview

- **File:** `messages.json`  
- **Total messages:** 100  
- **Columns detected:** `id`, `user_id`, `user_name`, `timestamp`, `message`  
- **Unique users:** 10  

**Note:** The dataset is unstructured text-heavy, with varying content lengths and some messages containing numeric information.

---

## Analysis Methodology

Since the dataset is **unlabeled and unstructured**, we followed these professional steps:

1. **Column Detection** – Automatically detect message, username, and timestamp columns.  
2. **Missing Data Analysis** – Identify null or empty fields.  
3. **Duplicate Message Detection** – Check for repeated messages per user.  
4. **Message Length Analysis** – Calculate min, max, mean, and quartiles to understand content size.  
5. **User Activity Analysis** – Identify top contributors and sparse users.  
6. **Contact Information Extraction** – Use regex patterns to find potential emails or phone numbers.  
7. **Numeric Data Analysis** – Extract numbers to identify possible ticket counts, reservations, or numeric anomalies.  
8. **Temporal Analysis** – Evaluate daily message distribution for patterns or gaps.  
9. **Name Consistency Check** – Ensure usernames are consistent and ready for NER matching.  

**Why regex for emails, phones, and numbers?**  
Even without prior knowledge of the dataset, regex allows us to **discover hidden structured data** that may appear in free-text messages. This is essential for:

- Detecting contact info for QA answers.  
- Identifying numeric facts (tickets, items, companions).  
- Finding anomalies or data entry errors (e.g., extreme numeric values).  

---

## Findings & Results

### 1. Missing Data

| Column        | Missing Count | Missing % |
|---------------|---------------|-----------|
| user_name     | 0             | 0%        |
| timestamp     | 0             | 0%        |
| message       | 0             | 0%        |

**Insight:** All messages have valid users and timestamps. The QA system can reliably link messages to users.

---

### 2. Duplicate Messages

- **Duplicates detected:** 0  

**Insight:** No repeated messages → embedding-based semantic search will not be biased by duplicate content.

---

### 3. Message Length

| Statistic | Characters |
|-----------|------------|
| Min       | 47         |
| Max       | 84         |
| Mean      | 63.6       |
| Median    | 63.5       |

**Insight:** Messages are concise and suitable for direct QA responses. No truncation required.

---

### 4. User Activity

| Top Users                | Messages |
|---------------------------|---------|
| Sophia Al-Farsi           | 16      |
| Fatima El-Tahir           | 15      |
| Hans Müller               | 11      |
| Layla Kawaguchi           | 10      |
| Lily O'Sullivan           | 10      |
| Vikram Desai              | 10      |

- **Sparse users (≤2 messages):** 0  

**Insight:** All users have sufficient messages for QA ranking. No additional handling for sparse users needed.

---

### 5. Contact Information

| Type    | Count |
|---------|-------|
| Emails  | 0     |
| Phones  | 9     |

**Insight:**  
- Emails are absent → QA will return “No contact info” for email queries.  
- Phone numbers are detectable → QA can provide phone info when queried.

---

### 6. Numeric Information

- **Min:** 1  
- **Max:** 123,456,789  
- **Median:** 555  

**Insight:**  
- Extreme outlier (123 million) suggests a data entry error.  
- QA system should **flag or handle numeric outliers** to avoid misleading answers.

---

### 7. Temporal Analysis

- **Average messages/day:** 0.28  
- **Max messages/day:** 2  
- **Most days:** 0 messages  

**Insight:** Message distribution is sparse. Temporal patterns cannot be heavily relied upon for QA relevance.

---

### 8. Name Consistency

- Names normalized to lowercase are consistent.  
- No significant variants detected.  

**Insight:** NER + exact or fuzzy matching will work reliably for user detection.

---

## Summary of Key Insights

| Category                  | Finding                                                                 |
|----------------------------|------------------------------------------------------------------------|
| Missing usernames          | 0 messages → all messages linked                                        |
| Missing timestamps         | 0 messages → temporal data complete                                      |
| Duplicate messages         | 0 duplicates → no bias in semantic search                                |
| Messages with emails       | 0 → QA fallback needed for email queries                                 |
| Messages with phones       | 9 → QA can retrieve phone info                                           |
| Max message length         | 84 → short messages, truncation not required                              |
| Numeric anomalies          | Outlier: 123,456,789 → handle carefully                                  |
| Sparse users (≤2 messages) | 0 → all users have sufficient messages                                    |

---

## Recommendations

1. **QA System Enhancements**
   - Handle numeric outliers gracefully.  
   - Include fallback for email queries since no emails exist.  
   - Leverage name consistency to improve NER accuracy.

2. **Data Quality Improvements**
   - Verify extreme numeric values for accuracy.  
   - Standardize phone number formats.  
   - Monitor incoming messages for duplicates or missing fields.

3. **Future Analysis**
   - If dataset grows, consider embedding clustering to detect anomalies or topic trends.  
   - Track user activity over time for engagement insights.

---

## Conclusion

Even with a small, unstructured dataset:

- The **data is mostly clean**, with no missing critical fields.  
- **QA system can reliably link messages to users** and answer numeric or contact queries.  
- **Minor anomalies** (numeric outlier, sparse phone info) require attention.  
- Regex-based EDA, numeric analysis, and temporal checks provide a **comprehensive understanding** of the dataset before deployment.

This structured approach ensures that **the QA system is robust, reliable, and ready for real-world usage**.