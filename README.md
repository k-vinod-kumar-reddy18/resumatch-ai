# 🧑‍💼 AI Resume Analyzer & Job Matcher

An end-to-end NLP-powered web application that parses resumes and matches them against job descriptions using **TF-IDF cosine similarity** and **spaCy NER-based skill extraction**.

Built with Python, Streamlit, scikit-learn, and spaCy.

---

## ✨ Features

- 📄 **Resume Parsing** — supports PDF and DOCX uploads
- 📝 **Job Description Input** — paste text or upload a PDF/TXT file
- 🎯 **Match Score** — TF-IDF cosine similarity score (0–100%)
- 🧠 **Skill Extraction** — detects 60+ ML/data/cloud/programming skills via regex + spaCy
- 🔴 **Gap Analysis** — highlights skills present in the JD but missing from the resume
- 💡 **Actionable Recommendations** — tailored advice based on match strength

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/k-vinod-kumar-reddy18/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| DOCX Parsing | python-docx |
| NLP / NER | spaCy (`en_core_web_sm`) |
| Similarity | TF-IDF + Cosine Similarity (scikit-learn) |

---

## 📂 Project Structure

```
ai-resume-analyzer/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 📸 Preview

Upload a resume → paste a job description → get an instant match score, skill gap chart, and recommendations — all in one page.

---

## 👤 Author

**K Vinod Kumar Reddy**  
B.Tech CSE (AI/ML) — Alliance University, Bangalore  
📧 kvinodreddy18@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/vinod-reddy-k-787932300) · [GitHub](https://github.com/k-vinod-kumar-reddy18)

---

## 📜 License

MIT License — free to use and modify.
