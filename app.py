import streamlit as st

from recommendation import (
    load_dataset,
    create_profiles,
    create_tfidf_model,
    recommend_careers
)

from chatbot import get_chatbot_response


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666666;
    font-size: 17px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
}

.career-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-bottom: 10px;
}

.chat-header {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

df = load_dataset()

df = create_profiles(df)

vectorizer, career_vectors = create_tfidf_model(df)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🎓 AI Career Assistant")

    st.write(
        "Your personal AI-powered career guidance system."
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🎯 Career Finder",
            "💬 AI Career Chat"
        ]
    )

    st.divider()

    st.caption(
        "AI Career Recommendation System"
    )


# =====================================================
# CAREER FINDER PAGE
# =====================================================

if page == "🎯 Career Finder":

    st.markdown(
        '<div class="main-title">'
        '🎓 AI Career Recommendation System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Find career paths that match your skills, interests, '
        'education, and experience.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =================================================
    # STUDENT INFORMATION
    # =================================================

    st.markdown(
        '<div class="section-title">'
        '👤 Student Information'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input(
            "Your Name",
            placeholder="Enter your name"
        )

    with col2:

        education = st.selectbox(
            "Education",
            [
                "Artificial Intelligence",
                "Computer Science",
                "Software Engineering",
                "Data Science",
                "Cybersecurity",
                "Information Technology",
                "Business",
                "Other"
            ]
        )


    experience = st.selectbox(
        "Experience Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )


    # =================================================
    # SKILLS
    # =================================================

    st.markdown(
        '<div class="section-title">'
        '💻 Your Skills'
        '</div>',
        unsafe_allow_html=True
    )

    skills = st.multiselect(
        "Select the skills you currently have",
        [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "SQL",
            "Statistics",
            "Pandas",
            "TensorFlow",
            "PyTorch",
            "Java",
            "C++",
            "HTML",
            "CSS",
            "JavaScript",
            "React",
            "Django",
            "Flask",
            "Networking",
            "Linux",
            "Cybersecurity",
            "AWS",
            "Azure",
            "Docker",
            "Kubernetes",
            "Git",
            "Data Analysis",
            "Figma"
        ]
    )


    # =================================================
    # INTERESTS
    # =================================================

    st.markdown(
        '<div class="section-title">'
        '❤️ Your Interests'
        '</div>',
        unsafe_allow_html=True
    )

    interests = st.multiselect(
        "Select the areas you are interested in",
        [
            "AI",
            "Coding",
            "Research",
            "Data",
            "Mathematics",
            "Web",
            "Design",
            "Cybersecurity",
            "Cloud",
            "Automation",
            "Business",
            "Robotics",
            "Technology",
            "Creativity",
            "Problem Solving"
        ]
    )


    st.divider()


    # =================================================
    # RECOMMENDATION BUTTON
    # =================================================

    if st.button(
        "🚀 Get My Career Recommendations",
        use_container_width=True
    ):

        if not name:

            st.warning(
                "Please enter your name."
            )

        elif not skills:

            st.warning(
                "Please select at least one skill."
            )

        elif not interests:

            st.warning(
                "Please select at least one interest."
            )

        else:

            with st.spinner(
                "Analyzing your profile..."
            ):

                top_5 = recommend_careers(
                    df=df,
                    vectorizer=vectorizer,
                    career_vectors=career_vectors,
                    skills=skills,
                    interests=interests,
                    education=education,
                    experience=experience
                )


            st.success(
                f"Great {name}! Your career recommendations are ready."
            )

            st.markdown(
                "## 🏆 Your Top 5 Career Matches"
            )

            st.write(
                "Recommendations are generated using "
                "TF-IDF similarity and weighted skill matching."
            )


            # =============================================
            # DISPLAY RESULTS
            # =============================================

            for rank, (_, row) in enumerate(
                top_5.iterrows(),
                start=1
            ):

                score = round(
                    row["final_score"] * 100,
                    2
                )

                score = max(
                    0,
                    min(score, 100)
                )


                st.markdown(
                    f"""
                    <div class="career-card">
                        <h3>#{rank} {row['career']}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                st.progress(
                    score / 100
                )

                st.write(
                    f"### 🎯 Match Score: {score}%"
                )

                st.write(
                    f"**💻 Required Skills:** "
                    f"{row['skills']}"
                )

                st.write(
                    f"**🎓 Relevant Education:** "
                    f"{row['education']}"
                )

                st.write(
                    f"**📈 Experience Level:** "
                    f"{row['experience']}"
                )

                st.write(
                    f"**📝 Career Description:** "
                    f"{row['description']}"
                )

                st.divider()


# =====================================================
# AI CAREER CHAT PAGE
# =====================================================

else:

    st.markdown(
        '<div class="main-title">'
        '🤖 AI Career Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about careers, skills, roadmaps, '
        'projects, and learning paths.'
        '</div>',
        unsafe_allow_html=True
    )


    # =================================================
    # CHAT HISTORY
    # =================================================

    if "messages" not in st.session_state:

        st.session_state.messages = []


    # =================================================
    # WELCOME MESSAGE
    # =================================================

    if len(st.session_state.messages) == 0:

        st.info(
            "👋 Hello! I am your AI Career Assistant. "
            "Ask me anything about your career."
        )


    # =================================================
    # DISPLAY OLD MESSAGES
    # =================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # =================================================
    # CHAT INPUT
    # =================================================

    question = st.chat_input(
        "Ask something about your career..."
    )


    # =================================================
    # PROCESS QUESTION
    # =================================================

    if question:

        # User message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        with st.chat_message("user"):

            st.write(question)


        # AI response

        with st.chat_message("assistant"):

            with st.spinner(
                "AI Career Assistant is thinking..."
            ):

                try:

                    answer = get_chatbot_response(
                        question
                    )

                    st.write(answer)


                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


                except Exception as e:

                    st.error(
                        "Sorry, the AI assistant is "
                        "temporarily unavailable."
                    )

                    st.caption(
                        f"Error: {e}"
                    )


    # =================================================
    # SUGGESTED QUESTIONS
    # =================================================

    st.divider()

    st.markdown(
        "### 💡 You can ask:"
    )

    st.write(
        "• What should I learn to become a Machine Learning Engineer?"
    )

    st.write(
        "• Give me a roadmap for becoming an AI Engineer."
    )

    st.write(
        "• What projects should I build for Data Science?"
    )

    st.write(
        "• What skills are required for a Python Developer?"
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Career Assistant | "
    "TF-IDF + Cosine Similarity + Weighted Skill Matching + Gemini AI"
)
# =====================================================
# AI CAREER ASSISTANT CHATBOT
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 AI Career Assistant</div>',
    unsafe_allow_html=True
)

st.write(
    "Ask me anything about careers, skills, learning paths, "
    "or your career recommendations."
)

from chatbot import get_chatbot_response

question = st.chat_input(
    "Ask your career question..."
)

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = get_chatbot_response(question)

        st.write(answer)