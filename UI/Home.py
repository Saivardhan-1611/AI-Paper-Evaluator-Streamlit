import streamlit as st
import subprocess
import sys
import os
import requests
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Use provided API keys
ocr_api_key = "K81173007088957"
gemini_api_key = "AIzaSyCjdRS_DUJhWFDsyHjaqAM4f01GLGokGoc"

# Title and intro
st.title("Welcome")
st.write("Upload student answer files here, then detect AI-generated content. After that, save the corrected text files.")

# Directory setup for saving only corrected text
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "temp_uploads")
os.makedirs(output_dir, exist_ok=True)

# Upload files
student_files = st.file_uploader("Upload Student Answer Files", type=["pdf", "txt"], accept_multiple_files=True)

# -------- Step 1: Extract + Detect AI --------
def extract_text_from_pdf(file):
    response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'file': file},
        data={
            'apikey': ocr_api_key,
            'language': 'eng',
            'OCREngine': 2,
            'isOverlayRequired': False
        }
    )
    result = response.json()
    if not result.get('IsErroredOnProcessing', True):
        return "\n".join([r.get("ParsedText", "") for r in result.get("ParsedResults", [])]).strip()
    return ""

def correct_text_with_gemini(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
    headers = {"Content-Type": "application/json"}
    prompt = f"Correct the spelling mistakes in the following handwritten extracted text. If no spelling mistakes are present, simply return the original given text and nothing else, make sure you give each question starting with Q: answer with A: if not present:\n\n{text.strip()}"
    data = { "contents": [ { "parts": [ { "text": prompt } ] } ] }
    response = requests.post(f"{url}?key={gemini_api_key}", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    return text

@st.cache_resource
def load_ai_detector():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    return tokenizer, model

def detect_ai_generated(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
    return probs[1] * 100  # AI probability %

# -------- Step 1 Button --------
if st.button("Extract and Detect AI"):
    if student_files:
        tokenizer, model = load_ai_detector()
        results = []
        st.session_state["extracted_texts"] = {}

        for file in student_files:
            filename = file.name

            # Extract text
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            else:
                text = file.getvalue().decode("utf-8", errors="ignore")

            if not text.strip():
                continue

            corrected = correct_text_with_gemini(text)
            ai_prob = detect_ai_generated(corrected, tokenizer, model)

            # Save corrected text for next step
            st.session_state["extracted_texts"][filename] = corrected
            results.append((filename, ai_prob))

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        st.session_state["ai_results"] = sorted_results
        st.session_state["extracted_ready"] = True
        st.session_state["uploads_ready"] = False

        # Save AI scores summary
        score_path = os.path.join(output_dir, "ai_scores.txt")
        with open(score_path, "w") as score_file:
            for fname, ai_score in sorted_results:
                score_file.write(f"{fname.replace('.pdf', '.txt')}: {ai_score:.2f}%\n")

# ✅ Always show results if they exist
if "ai_results" in st.session_state:
    st.subheader("AI Detection Results")
    for fname, ai_score in st.session_state["ai_results"]:
        color = "red" if ai_score > 50 else "green"
        st.markdown(
            f"""📄 **{fname}** → AI Probability: 
            <span style='color:{color}; font-size: 20px; font-weight: bold;'>{ai_score:.2f}%</span>""",
            unsafe_allow_html=True
        )
    st.session_state["uploads_ready"] = True

# -------- Step 2: Save Corrected Files --------
if st.session_state.get("extracted_ready"):
    for filename, text in st.session_state["extracted_texts"].items():
        corrected_path = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(text)
    st.success("Corrected text files saved to temp_uploads.")
    st.session_state["uploads_ready"] = True

# -------- Navigation Buttons --------
if st.session_state.get("uploads_ready"):
    if st.button("Proceed with AI"):
        script_path = os.path.join(script_dir, "frontend.py")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])

    if st.button("Proceed with Key"):
        script_path = os.path.join(script_dir, "ml_frontend.py")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
