import fitz
import google.generativeai as genai
import pandas as pd
import json
import re
 
# -------------------------
# GEMINI API CONFIG
# -------------------------
genai.configure(api_key="AIzaSyAJjVVD1lOlLzTz6YqVN8XcvEjFeBNYYXg")
 
model = genai.GenerativeModel("gemini-2.5-flash")
 
# -------------------------
# OPEN PDF
# -------------------------
doc = fitz.open("test_pdf.pdf")
 
all_questions = []
 
for page_num, page in enumerate(doc):
 
    text = page.get_text()
 
    prompt = f"""
Extract MCQ questions from the following text.
 
Rules:
1. Preserve mathematical expressions.
2. Convert powers like ² to ^2
3. Convert square roots to sqrt()
4. Return ONLY JSON.
 
Format:
[
{{
"question_name":"",
"choice1_text":"",
"choice2_text":"",
"choice3_text":"",
"choice4_text":""
}}
]
 
TEXT:
{text}
"""
 
    try:
 
        response = model.generate_content(prompt)
 
        raw = response.text
 
        # Remove markdown
        raw = re.sub(r"```json", "", raw)
        raw = re.sub(r"```", "", raw)
 
        data = json.loads(raw)
 
        for q in data:
 
            q["question_type_id"] = 1
            q["tags"] = "Math"
            q["weightage"] = 1
            q["is_active"] = True
 
            q["choice1_isCorrect"] = ""
            q["choice2_isCorrect"] = ""
            q["choice3_isCorrect"] = ""
            q["choice4_isCorrect"] = ""
 
            all_questions.append(q)
 
        print(f"Page {page_num+1} processed")
 
    except Exception as e:
 
        print(f"Error on page {page_num+1}:", e)
 
# -------------------------
# SAVE CSV
# -------------------------
df = pd.DataFrame(all_questions)
 
df = df[
[
"question_name",
"question_type_id",
"tags",
"weightage",
"is_active",
"choice1_text",
"choice1_isCorrect",
"choice2_text",
"choice2_isCorrect",
"choice3_text",
"choice3_isCorrect",
"choice4_text",
"choice4_isCorrect"
]
]
 
df.to_csv("questions.csv", index=False)
 
print("✅ CSV Generated Successfully")