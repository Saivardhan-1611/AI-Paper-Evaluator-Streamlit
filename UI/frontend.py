import streamlit as st
import fitz  # PyMuPDF
import requests
import re
import os
import pandas as pd
import io

# Backend API URLs
EVALUATE_API_URL = "https://ai-paper-evaluator.onrender.com/evaluate"
FETCH_EVALUATIONS_API_URL = "https://ai-paper-evaluator.onrender.com/evaluations"

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to extract text from PDF or TXT
def extract_text_from_path(file_path):
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        st.error(f"Error extracting text from {file_path}: {e}")
        return None

# Parse Q/A
def parse_questions_answers(text):
    qa_pairs = []
    qa_pattern = re.findall(r'Q:\s*(.*?)\nA:\s*(.*?)\n', text, re.DOTALL)
    for q, a in qa_pattern:
        qa_pairs.append({"question": q.strip(), "answer": a.strip()})
    return qa_pairs if qa_pairs else None

# Evaluate via backend
def evaluate_with_gemini(files_data):
    payload = {"files": files_data}
    headers = {"Content-Type": "application/json"}
    response = requests.post(EVALUATE_API_URL, json=payload, headers=headers)
    
    print("Server Response:", response.text)
    
    if response.status_code == 200:
        response_data = response.json()
        if "evaluations" in response_data:
            return response_data["evaluations"]
        elif "showResultsButton" in response_data:
            return fetch_previous_evaluations()
        else:
            return {"error": "Unexpected API response format"}
    else:
        return {"error": "Failed to evaluate"}

# Fetch stored results
def fetch_previous_evaluations():
    response = requests.get(FETCH_EVALUATIONS_API_URL)
    if response.status_code == 200:
        return response.json().get("evaluations", [])
    else:
        return {"error": "Failed to fetch previous evaluations"}

# Streamlit UI
st.title("Answer Paper Evaluator")
st.write("Evaluating from files inside `temp_uploads` folder.")

# Directory path for temp uploads
TEMP_UPLOADS_DIR = "temp_uploads"

files_data = []
if os.path.exists(TEMP_UPLOADS_DIR):
    for filename in os.listdir(TEMP_UPLOADS_DIR):
        if filename.endswith(".pdf") or filename.endswith(".txt"):
            file_path = os.path.join(TEMP_UPLOADS_DIR, filename)
            extracted_text = extract_text_from_path(file_path)
            if extracted_text:
                st.subheader(f"Extracted Text from {filename}:")
                st.text_area(f"Extracted Student Answers ({filename})", extracted_text, height=100, label_visibility="collapsed", key=filename)
                
                qa_data = parse_questions_answers(extracted_text)
                if qa_data:
                    files_data.append({"fileName": filename, "qaPairs": qa_data})

st.session_state["files_data"] = files_data

if files_data:
    if st.button("Evaluate with AI"):
        result = evaluate_with_gemini(files_data)
        
        if isinstance(result, list):
            st.subheader("Evaluation Results:")
            for file_result in result:
                st.write(f"### Results for {file_result['fileName']}")
                for idx, evaluation in enumerate(file_result['evaluations']):
                    st.write(f"**Q{idx+1}: {evaluation['question']}**")
                    st.write(f"**Score:** {evaluation['score']}")
                    st.write(f"**Feedback:** {evaluation['feedback']}")
                    st.write("---")
        else:
            st.json(result)

if st.button("View Scores"):
    previous_evaluations = fetch_previous_evaluations()
    
    if isinstance(previous_evaluations, list):
        st.subheader("Past Evaluations")
        for eval_record in previous_evaluations:
            with st.expander(f"{eval_record['fileName']} - Total Score: {eval_record['totalScore']}"):
                for ev in eval_record["evaluations"]:
                    st.write(f"**Question:** {ev['question']}")
                    st.write(f"**Answer:** {ev['answer']}")
                    st.write(f"**Score:** {ev['score']}")
                    st.write(f"**Feedback:** {ev['feedback']}")
                    st.write("---")

        # ✅ Read AI scores
        AI_SCORES_PATH = os.path.join(TEMP_UPLOADS_DIR, "ai_scores.txt")
        ai_scores_map = {}

        if os.path.exists(AI_SCORES_PATH):
            with open(AI_SCORES_PATH, "r") as f:
                for line in f:
                    match = re.match(r'(.+?)\.txt:\s*(\d+)', line.strip())
                    if match:
                        fname, score = match.groups()
                        ai_scores_map[f"{fname}.txt"] = int(score)

        # ✅ Create CSV with AI % and flag
        rows = []
        for record in previous_evaluations:
            file_name = record["fileName"]
            ai_percent = ai_scores_map.get(file_name, "N/A")
            flag = "Most-Likely AI" if isinstance(ai_percent, int) and ai_percent > 50 else ""

            row = {
                "File Name": file_name,
                "Total Score": record["totalScore"],
                "AI %": ai_percent,
                "Flag": flag
            }

            for idx, ev in enumerate(record["evaluations"], start=1):
                row[f"Q{idx} Score"] = ev["score"]
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Scores as CSV", data=csv_data, file_name="evaluations.csv", mime="text/csv")
    else:
        st.error("Failed to retrieve evaluations.")
