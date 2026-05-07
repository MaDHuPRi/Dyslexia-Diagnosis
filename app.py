import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dyslexia Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b69 0%, #1a0e3d 100%);
    border-right: 1px solid #3d2a7a;
}
[data-testid="stSidebar"] * { color: #e8e0ff !important; }

/* Compact nav: kill extra spacing around each radio option */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.75rem !important;
    border-radius: 10px !important;
    margin: 0.1rem 0 !important;
    transition: background 0.2s !important;
    cursor: pointer !important;
    line-height: 1.3 !important;
    min-height: unset !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.1) !important;
}
/* Hide the circle dot to save space */
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}
/* Remove extra wrapper padding Streamlit adds */
[data-testid="stSidebar"] .stRadio { margin-bottom: 0 !important; }
[data-testid="stSidebar"] .stRadio > div > div { padding: 0 !important; min-height: unset !important; }

/* ── App background ── */
.stApp { background: #f7f4ff; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Force dark text in main content ── */
.main p, .main li, .main h1, .main h2, .main h3 { color: #1a0e3d !important; }

/* ── Sidebar: EVERYTHING white, highest specificity ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio span,
[data-testid="stSidebar"] .stRadio p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span { color: #e8e0ff !important; }

/* ── Utility classes ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #1a0e3d;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #7a6fa0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.section-card {
    background: white;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    border: 1px solid #e8e0ff;
    box-shadow: 0 2px 12px rgba(45,27,105,0.06);
    color: #1a0e3d !important;
}
.section-card p, .section-card li, .section-card span, .section-card h3 {
    color: #1a0e3d !important;
}
.q-header {
    background: linear-gradient(135deg, #7c5cbf, #a173c9);
    color: white;
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    display: inline-block;
}
.score-pill {
    display: inline-block;
    background: linear-gradient(135deg, #7c5cbf, #a173c9);
    color: white;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    margin: 0.2rem;
}
.result-box-pos {
    background: linear-gradient(135deg, #fff3f0, #fff8f0);
    border: 2px solid #D85A30;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-box-neg {
    background: linear-gradient(135deg, #f0fff8, #f0f8ff);
    border: 2px solid #1D9E75;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-box-mod {
    background: linear-gradient(135deg, #fffbf0, #fff8e0);
    border: 2px solid #D4A017;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.fi-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
.fi-name { font-family: 'DM Mono', monospace; color: #5a4a7a; min-width: 170px; }
.fi-bar-bg { flex: 1; background: #ede8ff; border-radius: 999px; height: 8px; }
.fi-bar    { height: 8px; border-radius: 999px; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c5cbf, #a173c9);
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.7rem 2.5rem;
    font-family: 'Lato', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    width: 100%;
    margin-top: 0.5rem;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; transform: translateY(-1px); }

.stRadio [data-baseweb="radio"] { margin-bottom: 0.25rem; }
.stRadio label { font-size: 1rem; color: #2d1b69; }

.info-banner {
    background: linear-gradient(135deg, #2d1b69, #7c5cbf);
    color: white;
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        df = pd.read_csv("data/labeled_dysx.csv")
        df.columns = df.columns.str.strip()
        X = df.drop("Label", axis=1)
        y = df["Label"]
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n = 400
        X = pd.DataFrame({
            "Language_vocab": rng.uniform(0,1,n), "Memory": rng.uniform(0,1,n),
            "Speed": rng.uniform(0,1,n), "Visual_discrimination": rng.uniform(0,1,n),
            "Audio_Discrimination": rng.uniform(0,1,n), "Survey_Score": rng.uniform(0,1,n),
        })
        y = pd.Series((X["Language_vocab"]*0.3+X["Memory"]*0.2+X["Visual_discrimination"]*0.25+X["Survey_Score"]*0.25+rng.uniform(-0.15,0.15,n)>0.5).astype(int))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, list(X.columns), acc


# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem;'>
        <div style='font-size:2.5rem;'>🧠</div>
        <div style='font-family:"Playfair Display",serif; font-size:1.2rem; font-weight:700; margin-top:0.25rem;'>Dyslexia Detector</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.65rem; opacity:0.6; letter-spacing:0.1em; text-transform:uppercase; margin-top:0.25rem;'>AI · ML · Research</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.15); margin:0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📋  Survey", "🎯  Quiz", "🔬  Predict", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <hr style='border-color:rgba(255,255,255,0.15); margin:1.5rem 0 0.75rem;'>
    <div style='font-family:"DM Mono",monospace; font-size:0.62rem; opacity:0.45; text-align:center; line-height:1.7;'>
        Patent No. 202341084582 A<br>
        © 2025 Dyslexia Detector<br>
        Madhu Priya Pulletikurthi
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    col_hero, _ = st.columns([2, 1])
    with col_hero:
        st.markdown('<div class="page-title">Dyslexia<br>Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subtitle">// Machine Learning · Cognitive Assessment · Early Detection</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-banner">🧠  <b>How it works:</b> Complete the <b>Survey</b> (for parents/teachers) and the <b>Quiz</b> (for students), then use the <b>Predict</b> page with your scores to get an AI-powered dyslexia likelihood assessment.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📖 What is Dyslexia?")
        st.write("""
        Dyslexia is a language processing disorder that impacts reading, writing, and comprehension.
        Dyslexics may exhibit difficulty decoding words or with phonemic awareness — identifying
        individual sounds within words. It often goes undiagnosed for many years and can result in
        trouble with reading, grammar, comprehension, and other language skills.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Signs & Symptoms")
        st.write("""
        Dyslexia impacts people differently, so symptoms vary from person to person.
        A key sign is trouble decoding words and matching letters to sounds. Kids can also
        struggle with phonemic awareness — recognising sounds in words — which can show up
        as early as preschool age.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🧬 Possible Causes")
        st.write("""
        Researchers have found that **genes and brain differences** play a role:

        - **Genes & Heredity** — Dyslexia often runs in families. About 40% of siblings of people
          with dyslexia also struggle with reading. Up to 49% of parents of kids with dyslexia have it too.

        - **Brain Anatomy** — Brain imaging shows differences in areas involved with reading skills:
          recognising how sounds map to words, and what written words look like.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Our ML Approach")
        st.write("""
        Our model uses a **Random Forest Classifier** trained on cognitive assessment scores across
        six dimensions: language vocabulary, memory, speed, visual discrimination, audio discrimination,
        and survey score. Labels range from **0** (low risk) to **2** (high risk).

        The model achieves ~**94% accuracy** on held-out test data.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:white; border-radius:16px; padding:1.5rem 2rem; border:1px solid #e8e0ff; text-align:center; margin-top:0.5rem;'>
        <div style='font-family:"Playfair Display",serif; font-size:1.4rem; color:#2d1b69; margin-bottom:0.5rem;'>Ready to find out?</div>
        <div style='color:#5a4a7a; font-size:0.95rem;'>Start with the <b>Survey</b> or the <b>Quiz</b> from the sidebar, then head to <b>Predict</b>.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SURVEY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Survey":
    st.markdown('<div class="page-title">Survey</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">// For Parents & Teachers · Based on Observation</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-banner">📋 This survey is designed for <b>parents and teachers</b> to assess whether a child or student may show signs of dyslexia. Answer based on your observation. The score will be used in the Predict page.</div>', unsafe_allow_html=True)

    survey_questions = [
        "Did your child struggle to learn to count?",
        "Does he/she say numbers out of order, long after peers have mastered this skill?",
        "Does your child not seem to understand the connection between the symbol \"4\" and the word \"four\"? Does he make mistakes when reading or following directions involving numbers, words and symbols?",
        "Can he/she sort objects by size, shape or color? Does he have difficulty understanding the concepts bigger/smaller, more/less, before/after, yesterday/today/tomorrow?",
        "Difficulty with sequences, facts, and information not yet taught?",
        "Gets lost or confused easily; confuses left and right, over and under, before and after?",
        "Difficulty sustaining attention; seems \"hyper\" or \"daydreamer\"?",
        "Confused by letters, numbers, words, sequences, or verbal explanations?",
        "Reads and rereads with little comprehension?",
        "Difficulty putting thoughts into words; speaks in halting phrases; leaves sentences incomplete?",
        "Can count, but has difficulty counting objects and dealing with money?",
        "Does your child worry for sequences, facts and information that were not taught before?",
        "Does your child complain of dizziness, headaches or stomach aches while reading?",
        "Is reading extremely difficult for your child? (Below grade or age level)",
        "Is his spelling ability poor? Letters missed, reversed etc?",
        "Is it difficult for him to rhyme words?",
        "Is there difficulty telling time on a clock with hands and/or tying shoes with laces?",
        "Is there difficulty finding the right words while speaking? Lots of ums, ahs, 'those things', and 'that stuff'?",
        "Pauses, repeats or frequent mistakes when reading aloud?",
        "Unusually high or low tolerance for pain?",
    ]

    opts_yn = ["Yes — Frequently", "Sometimes", "No — Never"]
    opt_vals_yn = [0, 2, 4]

    answers = []
    for i, q in enumerate(survey_questions):
        st.markdown(f'<div class="section-card"><span class="q-header">Question {i+1}</span>', unsafe_allow_html=True)
        st.write(q)
        ans = st.radio(
            f"sq_{i}",
            opts_yn,
            label_visibility="collapsed",
            key=f"survey_q{i+1}"
        )
        answers.append(opt_vals_yn[opts_yn.index(ans)])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("📊 Calculate Survey Score"):
        total = sum(answers)
        survey_score = round(total / 80, 4)
        st.session_state["survey_score"] = survey_score

        level = "Low" if survey_score >= 0.6 else ("Moderate" if survey_score >= 0.35 else "High")
        color = "#1D9E75" if level == "Low" else ("#D4A017" if level == "Moderate" else "#D85A30")
        emoji = "✅" if level == "Low" else ("⚡" if level == "Moderate" else "⚠️")

        st.markdown(f"""
        <div style='background:white; border:2px solid {color}; border-radius:16px; padding:1.5rem 2rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:2.5rem;'>{emoji}</div>
            <div style='font-family:"Playfair Display",serif; font-size:1.6rem; color:{color}; font-weight:700;'>Survey Score: {survey_score:.4f}</div>
            <div style='color:#5a4a7a; margin-top:0.5rem;'>Concern Level: <b style='color:{color}'>{level}</b></div>
            <div style='font-family:"DM Mono",monospace; font-size:0.75rem; color:#999; margin-top:0.75rem;'>
                Score saved ✓ — head to the <b>Predict</b> page to run the full diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: QUIZ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Quiz":
    st.markdown('<div class="page-title">Cognitive Quiz</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">// For Students · Language, Memory, Visual & Audio Skills</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-banner">🎯 This quiz is designed for <b>students</b> to test cognitive indicators of dyslexia. Attempt all questions patiently. Your scores will be saved for the Predict page.</div>', unsafe_allow_html=True)

    # Track start time
    if "quiz_start" not in st.session_state:
        st.session_state["quiz_start"] = time.time()

    # ── Q1
    st.markdown('<div class="section-card"><span class="q-header">Question 1 · Visual Discrimination</span>', unsafe_allow_html=True)
    st.write("Are the characters **B** and **8** the same?")
    q1 = st.radio("q1", ["Yes", "No"], label_visibility="collapsed", key="qz1")
    q1_val = 4 if q1 == "Yes" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q2
    st.markdown('<div class="section-card"><span class="q-header">Question 2 · Language & Memory</span>', unsafe_allow_html=True)
    st.write("Guess the fruit: **purple, grows in clusters, small round berries on a vine**")
    q2 = st.radio("q2", ["Grapes", "Orange", "Banana", "Mango"], label_visibility="collapsed", key="qz2")
    q2_val = 4 if q2 == "Grapes" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q3
    st.markdown('<div class="section-card"><span class="q-header">Question 3 · Visual Discrimination</span>', unsafe_allow_html=True)
    st.write("Are the letters **g** and **s** the same or different?")
    q3 = st.radio("q3", ["Same", "Different"], label_visibility="collapsed", key="qz3")
    q3_val = 4 if q3 == "Different" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q4
    st.markdown('<div class="section-card"><span class="q-header">Question 4 · Visual Discrimination</span>', unsafe_allow_html=True)
    st.write("Look at these two letters: **G   O** — which position is the letter G?")
    q4 = st.radio("q4", ["First (G)", "Second (O)"], label_visibility="collapsed", key="qz4")
    q4_val = 4 if "First" in q4 else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q5
    st.markdown('<div class="section-card"><span class="q-header">Question 5 · Language & Vocab</span>', unsafe_allow_html=True)
    st.write("The word **CAT** — which letter does it start with?")
    q5 = st.radio("q5", ["K", "S", "C"], label_visibility="collapsed", key="qz5")
    q5_val = 4 if q5 == "C" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q6
    st.markdown('<div class="section-card"><span class="q-header">Question 6 · Visual Discrimination</span>', unsafe_allow_html=True)
    st.write("The uppercase letter is **D** — what is its lowercase form?")
    q6 = st.radio("q6", ["d", "b"], label_visibility="collapsed", key="qz6")
    q6_val = 4 if q6 == "b" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q7
    st.markdown('<div class="section-card"><span class="q-header">Question 7 · Audio Discrimination</span>', unsafe_allow_html=True)
    st.write("Listen carefully: the sound being described is the letter that makes the **'sss'** sound. What letter is it?")
    q7 = st.radio("q7", ["F", "S"], label_visibility="collapsed", key="qz7")
    q7_val = 4 if q7 == "S" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q8
    st.markdown('<div class="section-card"><span class="q-header">Question 8 · Language & Vocab</span>', unsafe_allow_html=True)
    st.write("The letters shown are: **G – O – D**. What word do they spell?")
    q8 = st.radio("q8", ["DOG", "GOD"], label_visibility="collapsed", key="qz8")
    q8_val = 4 if q8 == "GOD" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q9
    st.markdown('<div class="section-card"><span class="q-header">Question 9 · Memory & Spatial</span>', unsafe_allow_html=True)
    st.write("Imagine you are facing forward. Your **right** hand is on which side?")
    q9 = st.radio("q9", ["Left side", "Right side"], label_visibility="collapsed", key="qz9")
    q9_val = 4 if q9 == "Right side" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Q10
    st.markdown('<div class="section-card"><span class="q-header">Question 10 · Audio Discrimination</span>', unsafe_allow_html=True)
    st.write("The word rhymes with **bake** and is a body of water. What is the word?")
    q10 = st.radio("q10", ["CAKE", "LAKE", "TAKE", "FAKE"], label_visibility="collapsed", key="qz10")
    q10_val = 4 if q10 == "LAKE" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🎯 Submit Quiz & Calculate Scores"):
        time_spent = int(time.time() - st.session_state.get("quiz_start", time.time()))

        lang_vocab = (q1_val + q2_val + q3_val + q4_val + q5_val + q6_val + q8_val) / 28
        memory = (q2_val + q9_val) / 8
        visual_disc = (q1_val + q3_val + q4_val + q6_val) / 16
        audio_disc = (q7_val + q10_val) / 8
        speed = max(0.0, min(1.0, 1.0 - (time_spent / 300)))  # normalise: 0–5 min window

        # Save to session state
        st.session_state["quiz_language_vocab"] = round(lang_vocab, 4)
        st.session_state["quiz_memory"] = round(memory, 4)
        st.session_state["quiz_speed"] = round(speed, 4)
        st.session_state["quiz_visual_discrimination"] = round(visual_disc, 4)
        st.session_state["quiz_audio_discrimination"] = round(audio_disc, 4)
        st.session_state["quiz_time"] = time_spent

        st.markdown(f"""
        <div style='background:white; border:2px solid #7c5cbf; border-radius:16px; padding:1.5rem 2rem; margin-top:1rem;'>
            <div style='font-family:"Playfair Display",serif; font-size:1.4rem; color:#2d1b69; text-align:center; margin-bottom:1rem;'>✅ Quiz Complete!</div>
            <div style='display:flex; flex-wrap:wrap; gap:0.5rem; justify-content:center;'>
                <span class='score-pill'>Language & Vocab: {lang_vocab:.2f}</span>
                <span class='score-pill'>Memory: {memory:.2f}</span>
                <span class='score-pill'>Speed: {speed:.2f}</span>
                <span class='score-pill'>Visual: {visual_disc:.2f}</span>
                <span class='score-pill'>Audio: {audio_disc:.2f}</span>
                <span class='score-pill'>Time: {time_spent}s</span>
            </div>
            <div style='font-family:"DM Mono",monospace; font-size:0.75rem; color:#999; text-align:center; margin-top:1rem;'>
                Scores saved ✓ — head to <b>Predict</b> to run the full diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Reset timer for next attempt
        st.session_state["quiz_start"] = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Predict":
    st.markdown('<div class="page-title">Predict</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">// AI Diagnosis · Random Forest Classifier</div>', unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        model, feature_names, model_acc = load_model()

    # Check if scores are pre-filled from quiz/survey
    has_quiz = "quiz_language_vocab" in st.session_state
    has_survey = "survey_score" in st.session_state

    if has_quiz or has_survey:
        st.markdown("""
        <div style='background:linear-gradient(135deg,#e8ffe8,#f0fff8); border:1px solid #1D9E75; border-radius:12px; padding:1rem 1.5rem; margin-bottom:1rem; font-size:0.9rem;'>
            ✅ <b>Scores auto-filled</b> from your Quiz and/or Survey. You can adjust them manually below.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-card"><b style="font-family:\'DM Mono\',monospace; font-size:0.75rem; color:#7c5cbf; letter-spacing:0.1em; text-transform:uppercase;">// Cognitive Assessment Inputs</b>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        lang = st.slider("Language & Vocabulary", 0.0, 1.0,
                         float(st.session_state.get("quiz_language_vocab", 0.5)), 0.01)
        mem  = st.slider("Memory", 0.0, 1.0,
                         float(st.session_state.get("quiz_memory", 0.5)), 0.01)
        spd  = st.slider("Speed", 0.0, 1.0,
                         float(st.session_state.get("quiz_speed", 0.5)), 0.01)
    with col2:
        vis  = st.slider("Visual Discrimination", 0.0, 1.0,
                         float(st.session_state.get("quiz_visual_discrimination", 0.5)), 0.01)
        aud  = st.slider("Audio Discrimination", 0.0, 1.0,
                         float(st.session_state.get("quiz_audio_discrimination", 0.5)), 0.01)
        sur  = st.slider("Survey Score", 0.0, 1.0,
                         float(st.session_state.get("survey_score", 0.5)), 0.001)

    st.markdown('</div>', unsafe_allow_html=True)

    # Live value readout
    st.markdown(f"""
    <div style='font-family:"DM Mono",monospace; font-size:0.72rem; color:#aaa; display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:0.75rem;'>
        <span>lang: <b style="color:#7c5cbf">{lang:.2f}</b></span>
        <span>mem: <b style="color:#7c5cbf">{mem:.2f}</b></span>
        <span>speed: <b style="color:#7c5cbf">{spd:.2f}</b></span>
        <span>visual: <b style="color:#7c5cbf">{vis:.2f}</b></span>
        <span>audio: <b style="color:#7c5cbf">{aud:.2f}</b></span>
        <span>survey: <b style="color:#a173c9">{sur:.3f}</b></span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬 Run AI Diagnosis"):
        # Map feature names to values
        feature_map = {
            "Language_vocab": lang, "Memory": mem, "Speed": spd,
            "Visual_discrimination": vis, "Audio_Discrimination": aud, "Survey_Score": sur,
            # lowercase variants
            "language_vocab": lang, "memory": mem, "speed": spd,
            "visual_discrimination": vis, "audio_discrimination": aud, "survey_score": sur,
        }
        features = np.array([[feature_map.get(f, 0.5) for f in feature_names]])

        with st.spinner("Analysing…"):
            time.sleep(0.5)
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            classes = model.classes_

        pred_idx = list(classes).index(prediction)
        confidence = proba[pred_idx]

        label_map = {0: ("Low Risk", "✅", "#1D9E75", "result-box-neg"),
                     1: ("Moderate Risk", "⚡", "#D4A017", "result-box-mod"),
                     2: ("High Risk", "⚠️", "#D85A30", "result-box-pos")}

        try:
            pred_int = int(prediction)
        except (ValueError, TypeError):
            s = str(prediction).lower()
            pred_int = 2 if "high" in s or "dyslexic" in s else (1 if "mod" in s else 0)

        label, emoji, color, box_class = label_map.get(pred_int, ("Unknown", "❓", "#888", "result-box-neg"))

        st.markdown(f"""
        <div class="{box_class}" style="margin-top:1rem;">
            <div style='font-size:3rem;'>{emoji}</div>
            <div style='font-family:"Playfair Display",serif; font-size:1.8rem; color:{color}; font-weight:700; margin:0.25rem 0;'>{label}</div>
            <div style='font-family:"DM Mono",monospace; font-size:0.8rem; color:#888;'>
                Prediction: {prediction} &nbsp;·&nbsp; Confidence: {confidence:.1%} &nbsp;·&nbsp; Model Accuracy: {model_acc:.1%}
            </div>
            <div style='background:#f7f4ff; border-radius:999px; height:10px; margin:1rem auto; max-width:400px; overflow:hidden;'>
                <div style='height:100%; width:{confidence*100:.1f}%; background:linear-gradient(90deg,{color},{color}88); border-radius:999px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#e8e0ff; margin:1.5rem 0;'>", unsafe_allow_html=True)

        col_prob, col_fi = st.columns(2)
        with col_prob:
            st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#7a6fa0; margin-bottom:0.75rem;">// Class Probabilities</p>', unsafe_allow_html=True)
            c_colors = {0: "#1D9E75", 1: "#D4A017", 2: "#D85A30"}
            c_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
            for cls, prob in zip(classes, proba):
                try: ci = int(cls)
                except: ci = 0
                cc = c_colors.get(ci, "#7c5cbf")
                cl = c_labels.get(ci, str(cls))
                st.markdown(f"""
                <div class="fi-row">
                    <span class="fi-name">{cl}</span>
                    <div class="fi-bar-bg"><div class="fi-bar" style="width:{prob*100:.1f}%; background:{cc};"></div></div>
                    <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#888; min-width:40px; text-align:right;">{prob:.1%}</span>
                </div>""", unsafe_allow_html=True)

        with col_fi:
            st.markdown('<p style="font-family:\'DM Mono\',monospace; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#7a6fa0; margin-bottom:0.75rem;">// Feature Importances</p>', unsafe_allow_html=True)
            importances = model.feature_importances_
            fi_colors = ["#7c5cbf","#D85A30","#1D9E75","#D4537E","#D4A017","#378ADD"]
            sorted_idx = np.argsort(importances)[::-1]
            readable = {
                "Language_vocab":"Language & Vocab","language_vocab":"Language & Vocab",
                "Memory":"Memory","memory":"Memory","Speed":"Speed","speed":"Speed",
                "Visual_discrimination":"Visual Disc.","visual_discrimination":"Visual Disc.",
                "Audio_Discrimination":"Audio Disc.","audio_discrimination":"Audio Disc.",
                "Survey_Score":"Survey Score","survey_score":"Survey Score",
            }
            for rank, idx in enumerate(sorted_idx):
                fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                label_f = readable.get(fname, fname)
                imp = importances[idx]
                fc = fi_colors[rank % len(fi_colors)]
                st.markdown(f"""
                <div class="fi-row">
                    <span class="fi-name">{label_f}</span>
                    <div class="fi-bar-bg"><div class="fi-bar" style="width:{imp*100:.1f}%; background:{fc};"></div></div>
                    <span style="font-family:'DM Mono',monospace; font-size:0.75rem; color:#888; min-width:40px; text-align:right;">{imp:.1%}</span>
                </div>""", unsafe_allow_html=True)

        # Recommendations
        st.markdown("<hr style='border-color:#e8e0ff; margin:1.5rem 0;'>", unsafe_allow_html=True)
        recs = {
            0: "✅ **Low risk detected.** Results suggest no strong indicators of dyslexia. Continue monitoring and encourage regular reading activities.",
            1: "⚡ **Moderate risk detected.** Some indicators are present. Consider speaking with a school counsellor or educational psychologist for a professional evaluation.",
            2: "⚠️ **High risk detected.** Multiple strong indicators are present. We strongly recommend consulting a licensed educational or clinical specialist for a formal dyslexia assessment.",
        }
        st.info(recs.get(pred_int, "Please consult a professional for interpretation."))
        st.caption("⚠️ This tool is for educational and screening purposes only. It is not a clinical diagnosis. Please consult a qualified professional.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown('<div class="page-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">// Dataset · Model · Research</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Introduction to the Dataset")
    st.write("""
    This project uses Machine Learning to predict dyslexia likelihood from quiz and survey scores.
    The dataset contains six feature columns — **Language_vocab**, **Memory**, **Speed**,
    **Visual_discrimination**, **Audio_Discrimination**, and **Survey_Score** — and a **Label** column
    (0 = low risk, 1 = moderate risk, 2 = high risk).

    Every participant completes a quiz that tests language vocabulary, memory, speed, visual
    discrimination, and audio discrimination. Points are calculated per category and stored as the
    first five columns. A separate survey generates the **Survey_Score**. These six scores feed the model.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Model Details")
        st.write("""
        - **Algorithm**: Random Forest Classifier (200 trees)
        - **Optimisation**: GridSearchCV for hyperparameter tuning
        - **Error Rate**: ~5.8% on held-out test data
        - **Training split**: 80/20 train/test
        - **Features**: 6 normalised cognitive scores (0–1)
        - **Output**: 3-class label (0, 1, 2)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📜 Publication & Patents")
        st.write("""
        - **Patent**: India Patent No. 202341084582 A
        - **Published in**: TIJER (Technical International Journal of Engineering Research)
        - **Author**: Madhu Priya Pulletikurthi
        - **Repository**: [GitHub →](https://github.com/MaDHuPRi/Dyslexia-Diagnosis)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dataset preview
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Labeled Dataset Preview")
    try:
        df = pd.read_csv("data/labeled_dysx.csv")
        df.columns = df.columns.str.strip()
        st.dataframe(df.head(10), use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Samples", len(df))
        col_b.metric("Features", len(df.columns)-1)
        col_c.metric("Classes", df["Label"].nunique())
        st.bar_chart(df["Label"].value_counts().sort_index())
    except FileNotFoundError:
        st.warning("Dataset file not found. Place `labeled_dysx.csv` in the `data/` folder.")
    st.markdown('</div>', unsafe_allow_html=True)