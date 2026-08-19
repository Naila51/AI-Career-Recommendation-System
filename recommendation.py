import pandas as pd
from scikitlearn.feature_extraction.text import TfidfVectorizer
from scikitlearn.metrics.pairwise import cosine_similarity
streamlit
pandas
numpy
scikit-learn


def load_dataset():

    df = pd.read_csv("data/careers.csv")

    df = df.drop_duplicates()

    df = df.fillna("")

    return df


def create_profiles(df):

    df["profile"] = (
        df["skills"] + " " +
        df["interests"] + " " +
        df["education"] + " " +
        df["experience"] + " " +
        df["description"]
    )

    return df


def create_tfidf_model(df):

    vectorizer = TfidfVectorizer()

    career_vectors = vectorizer.fit_transform(
        df["profile"]
    )

    return vectorizer, career_vectors


def recommend_careers(
    df,
    vectorizer,
    career_vectors,
    skills,
    interests,
    education,
    experience
):

    user_profile = (
        " ".join(skills)
        + " "
        + " ".join(interests)
        + " "
        + education
        + " "
        + experience
    )

    user_vector = vectorizer.transform(
        [user_profile]
    )

    tfidf_scores = cosine_similarity(
        user_vector,
        career_vectors
    ).flatten()

    skill_scores = []

    user_skills = {
        skill.lower().strip()
        for skill in skills
    }

    for _, row in df.iterrows():

        career_skills = {
            skill.strip().lower()
            for skill in row["skills"].split(",")
        }

        if len(career_skills) > 0:

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

    results = df.copy()

    results["skill_score"] = skill_scores

    results["final_score"] = final_scores

    results = results.sort_values(
        by="final_score",
        ascending=False
    )

    return results.head(5)
