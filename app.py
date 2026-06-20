import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import re

# Load NLP model
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ─────────────────────────────────────────────
# Page Config + Styling
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🧑‍💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #00FFD1, #00c6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #a0b4c0;
    margin-bottom: 40px;
    font-weight: 300;
}

.card {
    padding: 24px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.score-badge {
    display: inline-block;
    padding: 12px 28px;
    border-radius: 50px;
    font-size: 28px;
    font-weight: 700;
    text-align: center;
}

.score-high  { background: linear-gradient(90deg,#00c27c,#00ffa3); color:#002d1a; }
.score-mid   { background: linear-gradient(90deg,#f0b429,#f7d070); color:#3d2a00; }
.score-low   { background: linear-gradient(90deg,#e53e3e,#fc8181); color:#fff; }

.skill-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    margin: 3px;
    font-weight: 500;
}

.skill-match   { background: rgba(0,255,209,0.15); border:1px solid #00ffd1; color:#00ffd1; }
.skill-missing { background: rgba(255,80,80,0.15); border:1px solid #ff5050;  color:#ff8080; }

[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.06);
    padding: 10px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}

textarea {
    background-color: rgba(255,255,255,0.06) !important;
    color: white !important;
    border-radius: 10px !important;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 54px;
    font-size: 18px;
    font-weight: 600;
    border: none;
    width: 100%;
    transition: all 0.2s;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    transform: scale(1.02);
}

h1, h2, h3 { color: #00FFD1; }

.stAlert { border-radius: 12px; }

div[role="radiogroup"] { color: white; }

.section-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7ec8e3;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Skills Database (expanded)
# ─────────────────────────────────────────────
SKILLS_DB = [
    # Programming
    "python", "java", "c", "c++", "javascript", "typescript", "r", "scala", "go", "rust",
    "sql", "nosql", "bash", "shell",
    # ML/DL
    "machine learning", "deep learning", "neural networks", "reinforcement learning",
    "computer vision", "natural language processing", "nlp",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    # Data
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data engineering", "feature engineering", "etl",
    "spark", "hadoop", "kafka", "airflow",
    # LLM / AI
    "langchain", "openai", "llm", "rag", "embeddings", "faiss", "vector search",
    "prompt engineering", "huggingface", "transformers", "fine-tuning",
    # Cloud
    "aws", "azure", "gcp", "docker", "kubernetes",
    # Tools
    "git", "github", "streamlit", "flask", "fastapi", "django",
    "power bi", "tableau", "excel", "mlflow",
    # Soft
    "communication", "teamwork", "leadership", "agile", "scrum",
]


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()


def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in SKILLS_DB:
        # Use word-boundary style matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)


def get_score_class(score):
    if score >= 60:
        return "score-high"
    elif score >= 35:
        return "score-mid"
    return "score-low"


def get_score_label(score):
    if score >= 60:
        return "Strong Match ✅"
    elif score >= 35:
        return "Partial Match ⚡"
    return "Low Match ⚠️"


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🧑‍💼 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your resume · paste a job description · get an instant match score & skill gap report</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-label">📄 Resume</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="section-label">📝 Job Description</div>', unsafe_allow_html=True)
    input_mode = st.radio("Input method", ["Paste Text", "Upload File"], horizontal=True, label_visibility="collapsed")
    job_description = ""
    if input_mode == "Paste Text":
        job_description = st.text_area("Paste the job description here…", height=200, label_visibility="collapsed",
                                       placeholder="Paste job description here…")
    else:
        jd_file = st.file_uploader("Upload JD (PDF / TXT)", type=["pdf", "txt"], label_visibility="collapsed")
        if jd_file:
            if jd_file.type == "application/pdf":
                job_description = extract_text_from_pdf(jd_file)
            else:
                job_description = jd_file.read().decode("utf-8")
            st.success("Job description loaded ✓")

# ─────────────────────────────────────────────
# Analyze Button
# ─────────────────────────────────────────────
st.markdown("---")
analyze = st.button("🚀 Analyze Resume")

if analyze:
    if not resume_file:
        st.error("⚠️ Please upload your resume (PDF or DOCX).")
    elif not job_description.strip():
        st.error("⚠️ Please provide a job description.")
    else:
        with st.spinner("Analyzing…"):
            # Extract text
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)

            # Compute
            score = calculate_similarity(resume_text, job_description)
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_description)
            matched = sorted(set(resume_skills) & set(job_skills))
            missing = sorted(set(job_skills) - set(resume_skills))

        # ── Results ──────────────────────────────────
        st.markdown("## 📊 Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            cls = get_score_class(score)
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:#7ec8e3;margin-bottom:8px;">Match Score</div>
                <div class="score-badge {cls}">{score}%</div>
                <div style="margin-top:8px;color:#ccc;font-size:14px;">{get_score_label(score)}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:#7ec8e3;margin-bottom:8px;">Matched Skills</div>
                <div class="score-badge" style="background:rgba(0,255,209,0.15);border:1px solid #00ffd1;color:#00ffd1;">{len(matched)}</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:13px;letter-spacing:1.5px;text-transform:uppercase;color:#7ec8e3;margin-bottom:8px;">Missing Skills</div>
                <div class="score-badge" style="background:rgba(255,80,80,0.15);border:1px solid #ff5050;color:#ff8080;">{len(missing)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### ✅ Matched Skills")
            if matched:
                chips = " ".join([f'<span class="skill-chip skill-match">{s}</span>' for s in matched])
                st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
            else:
                st.info("No matching skills found in our skill database for this JD.")

        with c2:
            st.markdown("### 🔴 Skills to Add")
            if missing:
                chips = " ".join([f'<span class="skill-chip skill-missing">{s}</span>' for s in missing])
                st.markdown(f'<div>{chips}</div>', unsafe_allow_html=True)
            else:
                st.success("Great news — no critical skill gaps detected!")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Recommendation")
        if score >= 60:
            st.success(f"Your resume is a **strong match** ({score}%). You're well-positioned to apply. Tailor your summary to mirror the JD language for best results.")
        elif score >= 35:
            msg = f"Your resume is a **partial match** ({score}%). "
            if missing:
                msg += f"Consider adding experience or projects involving: **{', '.join(missing[:5])}**."
            st.warning(msg)
        else:
            msg = f"Your resume has a **low match** ({score}%) with this role. "
            if missing:
                msg += f"Key gaps: **{', '.join(missing[:5])}**. Consider upskilling or targeting a closer role."
            st.error(msg)

        # Raw text expander
        with st.expander("🔍 View Extracted Resume Text"):
            st.text_area("Resume Text", resume_text, height=200, disabled=True)
