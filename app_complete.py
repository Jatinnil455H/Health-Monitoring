import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json

# Load model
print("Loading model...")
model_name = "shanover/symps_disease_bert_v3_c41"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

id2label = {
    "LABEL_0": "Acne", "LABEL_1": "AIDS", "LABEL_2": "Alcoholic hepatitis",
    "LABEL_3": "Allergy", "LABEL_4": "Arthritis", "LABEL_5": "Asthma",
    "LABEL_6": "Bronchitis", "LABEL_7": "Cervical Spondylosis",
    "LABEL_8": "Chicken pox", "LABEL_9": "Chronic cholestasis",
    "LABEL_10": "Common Cold", "LABEL_11": "COVID-19", "LABEL_12": "Dengue",
    "LABEL_13": "Diabetes", "LABEL_14": "Drug Reaction",
    "LABEL_15": "Fungal Infection", "LABEL_16": "Gastroenteritis",
    "LABEL_17": "GERD", "LABEL_18": "Heart Attack", "LABEL_19": "Hepatitis A",
    "LABEL_20": "Hepatitis B", "LABEL_21": "Hepatitis C",
    "LABEL_22": "Hepatitis D", "LABEL_23": "Hepatitis E",
    "LABEL_24": "Hypertension", "LABEL_25": "Hyperthyroidism",
    "LABEL_26": "Hypoglycemia", "LABEL_27": "Hypothyroidism",
    "LABEL_28": "Impetigo", "LABEL_29": "Jaundice", "LABEL_30": "Malaria",
    "LABEL_31": "Migraine", "LABEL_32": "Osteoarthritis",
    "LABEL_33": "Paralysis (Brain Hemorrhage)",
    "LABEL_34": "Peptic ulcer disease", "LABEL_35": "Pneumonia",
    "LABEL_36": "Psoriasis", "LABEL_37": "Tuberculosis",
    "LABEL_38": "Typhoid", "LABEL_39": "Urinary Tract Infection",
    "LABEL_40": "Varicose veins"
}

def predict_disease(symptoms):
    if not symptoms.strip():
        return "Please enter symptoms!"
    
    results = clf(symptoms)[0]
    
    output = "### 🎯 Top 5 Predictions:\n\n"
    for i, item in enumerate(results[:5], 1):
        disease = id2label[item["label"]]
        confidence = item["score"] * 100
        output += f"**{i}. {disease}**\n"
        output += f"   Confidence: {confidence:.2f}%\n\n"
    
    output += "\n⚠️ **Medical Disclaimer:** This is for informational purposes only. Consult a healthcare professional."
    
    return output

# Example symptoms
examples = [
    ["I have high fever, severe headache, muscle pain, and dry cough"],
    ["Runny nose, sneezing, watery eyes, and sore throat"],
    ["Excessive thirst, frequent urination, fatigue, and blurred vision"],
    ["Severe chest pain, sweating, and shortness of breath"],
    ["Difficulty breathing, wheezing, and chest tightness"]
]

# Create Gradio interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Describe your symptoms in detail...",
        label="Enter Your Symptoms"
    ),
    outputs=gr.Markdown(label="Prediction Results"),
    examples=examples,
    title="🏥 AI Disease Predictor",
    description="Describe your symptoms and get AI-powered disease predictions. The system can identify 41 different conditions.",
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
