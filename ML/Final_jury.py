import requests
from jury import Jury
from jury.metrics import load_metric
import nltk
from nltk.tokenize import word_tokenize
import math

# NLTK setup
# nltk.download('punkt')

# Firebase Database URLs
STUDENT_BASE_URL = "https://autopapercorrector-default-rtdb.firebaseio.com/answers/students.json"
MODEL_URL = "https://autopapercorrector-default-rtdb.firebaseio.com/model_answers.json"
SCORES_BASE_URL = "https://autopapercorrector-default-rtdb.firebaseio.com/answers/scores.json"

# Fetch student answers and model answers
students_response = requests.get(STUDENT_BASE_URL)
model_response = requests.get(MODEL_URL)

if students_response.status_code != 200 or model_response.status_code != 200:
    print("Error fetching data from Firebase.")
    exit()

students_data = students_response.json()
model_answers = model_response.json()

# Clean text helper
def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()

# Jury evaluator
jury = Jury(
    metrics=[
        load_metric("rouge"),
        load_metric("meteor"),
        load_metric("bertscore")
    ],
    run_concurrent=False
)

# Step 1: Collect raw scores
raw_scores = {}
max_question_score = 0.01  # To prevent division by zero
max_total_score = 0.01

for student_id, answers in students_data.items():
    student_scores = {}
    total_score = 0.0

    for idx in range(1, len(answers)):
        ans = answers[idx]
        if not isinstance(ans, dict):
            continue

        q_text = clean_text(ans.get("Q", ""))
        a_text = clean_text(ans.get("A", ""))

        if idx >= len(model_answers) or not isinstance(model_answers[idx], dict):
            continue

        model_ans = clean_text(model_answers[idx].get("A", ""))

        if not a_text or not model_ans:
            continue

        try:
            result = jury.evaluate(predictions=[a_text], references=[model_ans])
        except Exception:
            continue

        rougeL_score = result.get("rouge", {}).get("rougeL", 0)
        meteor_score = result.get("meteor", {}).get("score", 0)
        bert_score = result.get("bertscore", {}).get("f1", [0])[0]

        student_tokens = set(word_tokenize(a_text.lower()))
        model_tokens = set(word_tokenize(model_ans.lower()))
        keyword_match_score = len(student_tokens & model_tokens) / len(model_tokens) if model_tokens else 0
        len_ratio = len(a_text) / len(model_ans) if len(model_ans) > 0 else 1
        length_penalty = abs(1 - len_ratio) * 2.0

        final_score = (
            (4.0 * rougeL_score) +
            (3.5 * meteor_score) +
            (1.5 * bert_score) +
            (1.5 * keyword_match_score) -
            length_penalty
        ) * 10

        final_score = max(0, round(final_score, 2))  # Ensure non-negative
        student_scores[f"Q{idx}"] = final_score
        total_score += final_score

        # Track max for normalization
        if final_score > max_question_score:
            max_question_score = final_score

    student_scores["totalscore"] = round(total_score, 2)
    if total_score > max_total_score:
        max_total_score = total_score

    raw_scores[student_id] = student_scores

# Step 2: Normalize to [0, 10]
normalized_scores = {}
for student_id, scores in raw_scores.items():
    normalized = {}
    for qid, score in scores.items():
        if qid == "totalscore":
            normalized[qid] = round((score / max_total_score) * 10, 2)
        else:
            normalized[qid] = round((score / max_question_score) * 10, 2)
    normalized_scores[student_id] = normalized

# Step 3: Upload to Firebase
score_response = requests.patch(SCORES_BASE_URL, json=normalized_scores)

if score_response.status_code == 200:
    print("\n Successfully updated Firebase with normalized scores.")
else:
    print(f"\n Failed to update Firebase. Status: {score_response.status_code}, Error: {score_response.text}")
