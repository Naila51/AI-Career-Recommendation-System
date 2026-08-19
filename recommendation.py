import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    df = pd.read_csv("data/careers.csv")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Replace missing values
    df = df.fillna("")

    return df


# =====================================================
# CREATE CAREER PROFILES
# =====================================================

def create_profiles(df):

    df["profile"] = (
        df["skills"] + " "
        + df["interests"] + " "
        + df["education"] + " "
        + df["experience"] + " "
        + df["description"]
    )

    return df


# =====================================================
# TF-IDF MODEL
# =====================================================

def create_tfidf_model(df):

    vectorizer = TfidfVectorizer()

    career_vectors = vectorizer.fit_transform(
        df["profile"]
    )

    return vectorizer, career_vectors


# =====================================================
# CAREER RECOMMENDATION
# =====================================================

def recommend_careers(
    df,
    vectorizer,
    career_vectors,
    skills,
    interests,
    education,
    experience
):

    # -------------------------------------------------
    # Create user profile
    # -------------------------------------------------

    user_profile = (
        " ".join(skills)
        + " "
        + " ".join(interests)
        + " "
        + education
        + " "
        + experience
    )

    # -------------------------------------------------
    # Convert user profile into TF-IDF vector
    # -------------------------------------------------

    user_vector = vectorizer.transform(
        [user_profile]
    )

    # -------------------------------------------------
    # Calculate cosine similarity
    # -------------------------------------------------

    tfidf_scores = cosine_similarity(
        user_vector,
        career_vectors
    ).flatten()

    # -------------------------------------------------
    # Calculate skill matching score
    # -------------------------------------------------

    skill_scores = []

    user_skills = {
        skill.lower().strip()
        for skill in skills
    }

    for _, row in df.iterrows():

        career_skills = {
            skill.lower().strip()
            for skill in row["skills"].split(",")
        }

        if career_skills:

            matched_skills = (
                user_skills.intersection(career_skills)
            )

            skill_score = (
                len(matched_skills)
                / len(career_skills)
            )

        else:

            skill_score = 0

        skill_scores.append(skill_score)

    # -------------------------------------------------
    # Calculate final score
    # -------------------------------------------------

    final_scores = []

    for i in range(len(df)):

        tfidf_score = tfidf_scores[i]

        skill_score = skill_scores[i]

        final_score = (
            (tfidf_score * 0.50)
            +
            (skill_score * 0.50)
        )

        final_scores.append(final_score)

    # -------------------------------------------------
    # Create result dataframe
    # -------------------------------------------------

    results = df.copy()

    results["skill_score"] = skill_scores

    results["final_score"] = final_scores

    # Sort by highest score
    results = results.sort_values(
        by="final_score",
        ascending=False
    )

    # Return top 5 careers
    return results.head(5)
