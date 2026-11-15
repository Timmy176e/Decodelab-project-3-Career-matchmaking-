# Career Matchmaker Engine

A content-based recommendation system that suggests career paths based on a user's skills — built as Project 3 of the DecodeLabs AI Engineering Industrial Training (Batch 2026).

## How It Works

Unlike collaborative filtering (which needs historical user data — "users who liked X also liked Y"), this engine works from day one with zero interaction history. It matches **skills to roles directly**, using:

1. **TF-IDF Vectorization** — each career path is represented as a "skill document." TF-IDF converts these into weighted vectors, giving more weight to specific/unique skills (e.g. "Kubernetes") and less to common ones (e.g. "Git").
2. **Cosine Similarity** — a user's skill input is vectorized with the same vocabulary, then compared against every career vector by the *angle* between them, not raw distance — so it isn't thrown off by differing skill-list lengths.
3. **Top-N Ranking** — all roles are scored, sorted, and the top 3 matches are returned.

## Tech Stack

- Python
- scikit-learn (`TfidfVectorizer`, `cosine_similarity`)
- Streamlit

## Project Structure

```
├── app.py              # Streamlit app
├── requirements.txt    # Dependencies
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example

**Input:** `Python SQL Machine Learning`

**Output:**
- Data Scientist: 0.70
- Backend Developer: 0.27
- Data Analyst: 0.24

## Supported Career Paths

Data Scientist, DevOps Engineer, Backend Developer, Cloud Engineer, Sys Admin, Frontend Developer, Data Analyst

## Limitations

TF-IDF matches on exact vocabulary, not meaning — a skill described differently in the dataset than in the user's input (e.g. "ML" vs. "Machine Learning") won't be recognized as a match.

## Author

Tim Visual — DecodeLabs AI Engineering Industrial Training, Batch 2026
