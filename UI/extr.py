import requests
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Step 1: Extract text from PDF using OCR.space
def extract_text_from_pdf(file_path, ocr_api_key):
    ocr_url = 'https://api.ocr.space/parse/image'

    with open(file_path, 'rb') as f:
        response = requests.post(
            ocr_url,
            files={'file': f},
            data={
                'apikey': ocr_api_key,
                'language': 'eng',
                'OCREngine': 2,
                'isOverlayRequired': False,
                'isTable': False,
                'scale': True,
                'detectOrientation': True
            }
        )

    result = response.json()

    if not result.get('IsErroredOnProcessing', True):
        parsed_text = ""
        for parsed_result in result.get('ParsedResults', []):
            parsed_text += parsed_result.get('ParsedText', '') + "\n"
        return parsed_text.strip()
    else:
        print("❌ OCR Error:", result.get('ErrorMessage'))
        return None

# Step 2: Correct spelling using Gemini
def correct_text_with_gemini(text, gemini_api_key):
    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
    headers = { "Content-Type": "application/json" }

    prompt = f"Correct the spelling mistakes in the following handwritten extracted text:\n\n{text.strip()}"

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        f"{gemini_url}?key={gemini_api_key}",
        headers=headers,
        data=json.dumps(data)
    )

    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        print("❌ Gemini Error:", response.text)
        return None

# Step 3: Detect AI-generated text using Hugging Face model
def detect_ai_generated(text):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    human_prob = probs[0]
    ai_prob = probs[1]

    print("\n🔍 AI Detection Result:")
    print(f"🤖 AI-Generated Probability: {ai_prob * 100:.2f}%")
    print(f"🧑 Human-Written Probability: {human_prob * 100:.2f}%")

    if ai_prob > human_prob:
        print("⚠️ This text is likely AI-generated.")
    else:
        print("✅ This text is likely human-written.")

# -------- SET FILE PATH AND YOUR API KEYS --------
pdf_path = "C:/Users/kella/OneDrive/Desktop/Writing.pdf"
ocr_api_key = "K81173007088957"
gemini_api_key = "AIzaSyCjdRS_DUJhWFDsyHjaqAM4f01GLGokGoc"

# -------- MAIN EXECUTION FLOW --------
extracted_text = extract_text_from_pdf(pdf_path, ocr_api_key)

if extracted_text:
    print("\n--- 📝 Raw OCR Text ---\n")
    print(extracted_text)

    corrected_text = correct_text_with_gemini(extracted_text, gemini_api_key)

    if corrected_text:
        print("\n--- ✨ Corrected Text ---\n")
        print(corrected_text)

        detect_ai_generated(corrected_text)
