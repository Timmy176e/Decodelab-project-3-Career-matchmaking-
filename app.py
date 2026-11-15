import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup (cached so it only runs once, not on every click) ---
@st.cache_resource
def build_engine():
    career_paths = {
        "Data Scientist": "Python SQL Machine Learning Data Analysis Statistics Pandas",
        "DevOps Engineer": "AWS Docker Kubernetes CI/CD Linux Git",
        "Backend Developer": "Java Python SQL APIs REST Git",
        "Cloud Engineer": "AWS Azure Google Cloud Terraform Docker Kubernetes Networking",
        "Sys Admin": "Linux Windows Server Networking Bash Scripting Security Backup",
        "Frontend Developer": "JavaScript React HTML CSS TypeScript Git UI",
        "Data Analyst": "SQL Excel Data Analysis Visualization Tableau Statistics Python"
    }
    career_names = list(career_paths.keys())
    career_docs = list(career_paths.values())

    vectorizer = TfidfVectorizer()
    career_vectors = vectorizer.fit_transform(career_docs)

    return vectorizer, career_vectors, career_names

vectorizer, career_vectors, career_names = build_engine()

def recommend_career(skills_input, top_n=3):
    user_vector = vectorizer.transform([skills_input])
    scores = cosine_similarity(user_vector, career_vectors)[0]
    ranked = list(zip(career_names, scores))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]

# --- UI ---
st.title("Career Matchmaker Engine")
st.write("Enter your skills and get your Top 3 matching career paths.")

skills_input = st.text_input("Your skills (comma or space separated):")

if st.button("Find my match"):
    if skills_input.strip() == "":
        st.warning("Please enter at least one skill.")
    else:
        results = recommend_career(skills_input)
        st.subheader("Top 3 matches:")
        for name, score in results:
            st.write(f"**{name}**: {score:.2f}")
