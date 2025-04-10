import streamlit as st
import pyrebase
import pdfplumber
import re
import os
import io
import pandas as pd
import concurrent.futures
import math  # ✅ Added for ceiling of total score

# Firebase config (same)
config = {
    "apiKey": "AIzaSyBkQNccb3rGilGorWBH1o4p4th91No0tEY",
    "authDomain": "autopapercorrector.firebaseapp.com",
    "databaseURL": "https://autopapercorrector-default-rtdb.firebaseio.com",
    "projectId": "autopapercorrector",
    "storageBucket": "autopapercorrector.appspot.com",
    "messagingSenderId": "688638687879"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
storage = firebase.storage()

os.makedirs("temp_uploads", exist_ok=True)

# ✅ Cached PDF extraction
@st.cache_data(show_spinner=False)
def cached_extract_pdf_text(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def extract_text_from_pdf(pdf_file):
    return cached_extract_pdf_text(pdf_file.read())

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def parse_qa(text):
    qa_pairs = re.findall(r'Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|$)', text, re.DOTALL)
    return {str(i + 1): {"Q": q.strip(), "A": a.strip()} for i, (q, a) in enumerate(qa_pairs)}

# ✅ UI Title
st.title("Answer Paper Evaluator")

# ✅ Form for grouped upload
with st.form("upload_form"):
    model_file = st.file_uploader("Upload Model Answers (PDF or TXT)", type=["pdf", "txt"])
    submitted = st.form_submit_button("Upload & Process")

if submitted:
    student_files = [f for f in os.listdir("temp_uploads") if f.endswith((".pdf", ".txt"))]

    if student_files and model_file:
        # Model answer process
        model_text = extract_text_from_pdf(model_file) if model_file.type == "application/pdf" else extract_text_from_txt(model_file)
        parsed_model = parse_qa(model_text)
        db.child("model_answers").set(parsed_model)
        st.success("Model answers uploaded.")

        # ✅ Parallel student processing
        def process_student_file(filename):
            student_id = os.path.splitext(filename)[0]
            file_path = os.path.join("temp_uploads", filename)

            with open(file_path, "rb") as f:
                text = extract_text_from_pdf(f) if filename.lower().endswith(".pdf") else extract_text_from_txt(f)
            parsed = parse_qa(text)
            if parsed:
                db.child("answers").child("students").child(student_id).set(parsed)
                return student_id
            return None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(process_student_file, student_files))

        st.success("Student files processed successfully!")
        st.session_state["data_uploaded"] = True
    else:
        st.error("Upload a model answer and ensure temp_uploads has student files.")

# ✅ Evaluate trigger (subprocess remains for now)
if st.session_state.get("data_uploaded"):
    if st.button("Evaluate"):
        import subprocess
        script_path = os.path.abspath(os.path.join("..", "ML", "Final_jury.py"))
        if os.path.exists(script_path):
            result = subprocess.run(["python", script_path], capture_output=True, text=True)
            if result.returncode == 0:
                st.text_area("Evaluation Output:", result.stdout)
            else:
                st.error(f"Script Error:\n{result.stderr}")
        else:
            st.error("Evaluation script not found!")

# ✅ View scores with reduced Firebase reads
if st.button(" View Scores "):
    if "scores_data" not in st.session_state:
        st.session_state["scores_data"] = db.child("answers").child("scores").get().val() or {}
    if "student_answers" not in st.session_state:
        st.session_state["student_answers"] = db.child("answers").child("students").get().val() or {}

    scores_data = st.session_state["scores_data"]
    student_answers = st.session_state["student_answers"]

    if scores_data and student_answers:
        for student_id, score_entry in scores_data.items():
            # ✅ Use ceil value of total score
            total_score = math.ceil(sum(score for q, score in score_entry.items() if q != "totalscore" and isinstance(score, (int, float))))
            with st.expander(f"📄 {student_id} |  Total Score: {total_score}"):
                st.markdown("###  Scores")
                for q, score in score_entry.items():
                    if q != "totalscore":
                        st.markdown(f"- **{q}:** {score}")

                st.markdown("### 📄 Answers")
                student_qa = student_answers.get(student_id, {})

                if isinstance(student_qa, list):
                    student_qa = {str(i + 1): qa for i, qa in enumerate(student_qa)}

                if isinstance(student_qa, dict):
                    for qid, qa in student_qa.items():
                        if isinstance(qa, dict):
                            q = qa.get("Q", "N/A")
                            a = qa.get("A", "N/A")
                            st.markdown(f"**Q{qid}:** {q}")
                            st.markdown(f"**A{qid}:** {a}")
                            st.markdown("---")
                else:
                    st.warning(f"No valid answers found for `{student_id}`")

    # ✅ Streamlit-native CSV download with single click
    scores_data = db.child("answers").child("scores").get().val()
    if scores_data:
        # ✅ Read AI scores
        AI_SCORES_PATH = os.path.join("temp_uploads", "ai_scores.txt")
        ai_scores_map = {}

        if os.path.exists(AI_SCORES_PATH):
            with open(AI_SCORES_PATH, "r") as f:
                for line in f:
                    match = re.match(r'(.+?)\.txt:\s*(\d+)', line.strip())
                    if match:
                        fname, score = match.groups()
                        ai_scores_map[f"{fname}.txt"] = int(score)

        # ✅ Compose CSV rows
        rows = []
        for student_id, score_entry in scores_data.items():
            row = {"File Name": f"{student_id}.txt"}
            total_score = 0.0  # ✅ Use float to avoid early rounding

            for q, score in score_entry.items():
                if q.lower() != "totalscore":
                    row[f"Q{q} Score"] = score
                    try:
                        total_score += score
                    except:
                        pass

            ai_percent = ai_scores_map.get(f"{student_id}.txt", "N/A")
            row["AI %"] = ai_percent
            row["Flag"] = "Most-Likely AI" if isinstance(ai_percent, int) and ai_percent > 50 else ""
            row["Total Score"] = math.ceil(total_score)  # ✅ Ceil applied
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False).encode("utf-8")

        # ✅ Single-click download button
        st.download_button("Download Scores as CSV", data=csv_data, file_name="evaluations.csv", mime="text/csv")
    else:
        st.warning("No scores available to download.")
