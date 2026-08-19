import streamlit as st
from google import genai
import time


# =====================================================
# GEMINI API
# =====================================================

api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


# =====================================================
# AVAILABLE MODELS
# =====================================================

MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]


# =====================================================
# CHATBOT FUNCTION
# =====================================================

def get_chatbot_response(question):

    prompt = f"""
You are an AI Career Assistant for an
AI Career Recommendation System.

Your job is to help students understand:

- Career options
- Required skills
- Learning roadmaps
- AI and technology careers
- Machine Learning
- Data Science
- Software Development
- Cybersecurity
- Cloud Computing
- Web Development
- AI Engineering

Give simple, practical and student-friendly answers.

Do not give unrelated answers.

If the question is about a career, explain:
1. What the career is
2. Required skills
3. Beginner learning path
4. Useful tools
5. Possible projects

Keep answers clear and easy to understand.

Student's question:
{question}
"""

    last_error = None

    for model_name in MODELS:

        for attempt in range(2):

            try:

                chat = client.chats.create(
                    model=model_name
                )

                response = chat.send_message(prompt)

                return response.text

            except Exception as e:

                last_error = e
                time.sleep(2)

    return (
        "Sorry, the AI service is temporarily busy. "
        "Please try again in a few seconds."
    )