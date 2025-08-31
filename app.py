import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

# --- MEDICAL KNOWLEDGE BASE ---
MEDICAL_DATA = [
    {"title": "Hypertension (High Blood Pressure)",
     "content": "Hypertension is a condition in which the force of the blood against the artery walls is too high. Symptoms include severe headaches, nosebleeds, and shortness of breath. Lifestyle changes like a low-sodium diet and regular exercise are key to prevention.",
     "keywords": ["hypertension", "blood pressure", "headaches", "nosebleeds"]},
    {"title": "Diabetes",
     "content": "Diabetes is a chronic disease that affects how your body turns food into energy. Symptoms include frequent urination, increased thirst, and unexplained weight loss. Managing blood sugar levels through diet and medication is crucial for treatment.",
     "keywords": ["diabetes", "blood sugar", "urination", "thirst", "weight loss"]},
    {"title": "Common Cold",
     "content": "The common cold is a viral infection of your nose and throat (upper respiratory tract). Symptoms are a runny nose, sore throat, and sneezing. It is usually harmless and the best way to prevent it is by washing your hands and avoiding contact with sick people.",
     "keywords": ["common cold", "runny nose", "sore throat", "sneezing", "fever"]},
    {"title": "Asthma",
     "content": "Asthma is a condition in which your airways narrow and swell and produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. It is a chronic disease.",
     "keywords": ["asthma", "wheezing", "shortness of breath", "coughing", "breathing"]},
    {"title": "Migraine",
     "content": "A migraine is a type of severe headache that can cause throbbing in the head, often on one side. It can be accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Rest in a dark, quiet room is often helpful for relief.",
     "keywords": ["migraine", "headache", "throbbing", "nausea", "light sensitivity"]}
]

# --- PRE-TRAINED MODEL FOR SYMPTOM EXTRACTION ---
# This is our predictive field. We'll use a pre-trained Named Entity Recognition (NER) model
# to identify symptoms and diseases from the user's input.
# 'dslim/bert-base-NER' is a great general-purpose NER model.
# For a real-world medical application, you would fine-tune a model like BioBERT on a medical dataset.

@st.cache_resource
def get_ner_pipeline():
    """Caches the NER pipeline to avoid reloading the model on every interaction."""
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# --- RAG FUNCTIONALITY ---
def find_relevant_info(query):
    query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
    matches = []
    for data in MEDICAL_DATA:
        match_score = len(query_tokens.intersection(set(data["keywords"])))
        if match_score > 0:
            matches.append({"title": data["title"], "score": match_score, "content": data["content"]})

    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    if not matches:
        return "I'm sorry, I couldn't find relevant information on that topic in my knowledge base."
    
    return matches[0]["content"]

# --- STREAMLIT APP LAYOUT ---
st.title("⚕️ HealthAI Suite")
st.write("Welcome to your personal AI health assistant. You can use this tool to ask questions about health topics or get a preliminary diagnosis based on your symptoms.")

tab1, tab2 = st.tabs(["💬 General Q&A", "🔍 Symptom Analyzer"])

with tab1:
    st.header("General Q&A")
    st.markdown("Ask me anything about common diseases, symptoms, and health tips.")
    
    # Initialize session state for user input
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    
    user_question = st.text_input("Your question:", value=st.session_state.user_question, key="question_input")
    
    # Add suggested questions as interactive buttons
    st.subheader("Or, try these questions:")
    default_questions = [
        "What are the symptoms of hypertension?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "What medications are used for high cholesterol?",
        "What is the relationship between cholesterol and heart disease?"
    ]
    
    # Use st.columns to display buttons in a row
    cols = st.columns(len(default_questions))
    
    for col, question in zip(cols, default_questions):
        with col:
            if st.button(question, key=question):
                st.session_state.user_question = question
                st.experimental_rerun()
                
    if st.session_state.user_question:
        response = find_relevant_info(st.session_state.user_question)
        st.write(response)

with tab2:
    st.header("Symptom Analyzer (Predictive Field)")
    st.markdown("Describe your symptoms, and this module will use a **pre-trained model** to analyze them and suggest a possible diagnosis.")
    
    ner_pipeline = get_ner_pipeline()
    
    symptoms_input = st.text_area("Example: 'I have a sore throat and a runny nose. I am also coughing.'")

    if st.button("Analyze Symptoms"):
        if symptoms_input:
            ner_results = ner_pipeline(symptoms_input)
            
            # Filter for relevant entities (e.g., MISC, ORG, LOC, etc.)
            identified_symptoms = [result['word'].lower() for result in ner_results if result['entity_group'] in ['MISC', 'ORG', 'LOC']]
            
            if not identified_symptoms:
                st.warning("The model did not identify any key symptoms in your description. Please try again with more specific medical terms.")
            else:
                symptom_text = " ".join(identified_symptoms)
                
                diagnosis_scores = {}
                for diagnosis, info in MEDICAL_KNOWLEDGE_BASE.items():
                    score = sum(1 for symptom in info["symptoms"] if symptom in symptom_text)
                    diagnosis_scores[diagnosis] = score

                if not any(diagnosis_scores.values()):
                    st.warning("I couldn't match the extracted symptoms to a specific diagnosis in my knowledge base. Please consult a doctor.")
                else:
                    sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda item: item[1], reverse=True)
                    top_diagnosis, top_score = sorted_diagnoses[0]

                    if top_score > 0:
                        st.subheader("Analysis Results")
                        st.success(f"Based on my analysis, a possible diagnosis is **{top_diagnosis}**.")
                        st.write("---")
                        st.write("Additional Information:")
                        st.info(f"**Prevention:** {MEDICAL_KNOWLEDGE_BASE[top_diagnosis]['prevention']}")
                        
                    else:
                        st.warning("I could not confidently make a diagnosis based on the symptoms provided.")
        else:
            st.error("Please enter your symptoms to begin the analysis.")

        st.warning("Disclaimer: This tool is for informational purposes only and is not a substitute for professional medical advice. Always consult a healthcare professional for diagnosis and treatment.")

MEDICAL_KNOWLEDGE_BASE = {
    "Common Cold": {
        "symptoms": ["runny nose", "sore throat", "cough", "sneezing", "headache", "fever", "body aches"],
        "prevention": "Wash your hands frequently, avoid touching your face, and get adequate rest.",
    },
    "Influenza (Flu)": {
        "symptoms": ["fever", "chills", "muscle aches", "cough", "sore throat", "fatigue", "headache"],
        "prevention": "Get an annual flu vaccine, wash hands often, and avoid close contact with sick people.",
    },
    "Allergies": {
        "symptoms": ["sneezing", "itchy eyes", "runny nose", "rash", "hives", "wheezing", "difficulty breathing"],
        "prevention": "Avoid known allergens, use an air purifier, and keep windows closed during high-pollen seasons.",
    },
    "Stomach Flu (Gastroenteritis)": {
        "symptoms": ["nausea", "vomiting", "diarrhea", "stomach cramps", "fever", "dehydration"],
        "prevention": "Practice good hygiene, especially hand-washing, and avoid contaminated food and water.",
    },
    "Migraine": {
        "symptoms": ["severe headache", "throbbing pain", "nausea", "vomiting", "light sensitivity", "sound sensitivity"],
        "prevention": "Identify and avoid triggers (like certain foods or stress), maintain a regular sleep schedule, and stay hydrated.",
    }
}
