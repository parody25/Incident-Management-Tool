import streamlit as st
import requests
import uuid
import datetime
import re
import time
import openai
import os
import random
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

FASTAPI_URL = "http://localhost:8000/get_solution"

# Department classification keywords
DEPARTMENT_KEYWORDS = {
    "IT Support": ["login", "password", "access", "account", "user", "reset", "permissions", "unlock"],
    "Network": ["internet", "network", "wifi", "connection", "slow", "disconnect", "firewall", "dns"],
    "Security": ["virus", "malware", "hack", "breach", "unauthorized", "phishing", "ransomware"],
    "Database": ["database", "db", "query", "timeout", "corruption", "backup", "restore"],
    "Application": ["app", "application", "software", "bug", "crash", "error", "freeze", "slow response"],
    "Infrastructure": ["server", "disk", "memory", "cpu", "outage", "maintenance", "power"]
}

def classify_department(detailed_description: str) -> str:
    text_lower = detailed_description.lower()
    max_matches = 0
    best_dept = "IT Support"  # default
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > max_matches:
            max_matches = matches
            best_dept = dept
    return best_dept

def generate_short_description(detailed_description: str) -> str:
    # Simple: first sentence, truncated to 100 chars
    sentences = re.split(r'[.!?]+', detailed_description.strip())
    short = sentences[0].strip() if sentences else detailed_description[:50].strip()
    return short[:100]  # limit

# Optionally, use LLM to generate short desc
# def generate_short_description_llm(detailed_description: str) -> str:
#     prompt = f"Summarize this incident description into a short title (max 50 words): {detailed_description}"
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=50
#         )
#         return response.choices[0].message.content.strip()
#     except:
#         return detailed_description[:50]

def main():
    st.set_page_config(
        page_title="Incident Management Chat Bot",
        page_icon="🛠️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.title("🛠️ Incident Management System")
    st.markdown("Welcome! Provide the incident details below and we'll process it automatically.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history using chat_message components
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Incident creation form
    with st.form(key="incident_form"):
        description = st.text_input("Short Description (Summary)", placeholder="Enter a brief summary of the incident")
        detailed_description = st.text_area("Detailed Description", placeholder="Enter a detailed description of the incident")
        department_options = list(DEPARTMENT_KEYWORDS.keys())
        department = st.selectbox("Department", department_options, index=0)

        submit_button = st.form_submit_button(label="Submit Incident")

    if submit_button and detailed_description.strip():
        # Add user message to history
        user_msg = f"**Description:** {description}\n\n**Detailed Description:** {detailed_description}\n\n**Department:** {department}"
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

        # Process the incident
        with st.spinner("🔄 Processing incident..."):
            description = description.strip() or generate_short_description(detailed_description)
            # department is selected from dropdown
            incident_num = f"INC-{int(time.time())}"
            customer_name = f"User{random.randint(10000, 99999)}"
            organization = "CBD"
            reported_date = datetime.datetime.now().isoformat()

            payload = {
                "incident_num": incident_num,
                "customer_name": customer_name,
                "organization": organization,
                "department": department,
                "description": description,
                "detailed_description": detailed_description,
                "reported_date": reported_date
            }

            try:
                response = requests.post(FASTAPI_URL, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                priority = result["priority"]
                solution = result["solution"]

                # Add assistant response to history with enhanced formatting
                assistant_msg = f"✅ **Incident {incident_num}** successfully created!\n\n**Department:** {department}\n\n🔔 **Priority:** {priority}\n\n🔧 **Solution:** {solution}"
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})

            except Exception as e:
                error_msg = f"❌ Error processing incident: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        st.rerun()

if __name__ == "__main__":
    main()
