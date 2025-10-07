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

# Issue type categories for user selection
ISSUE_CATEGORIES = {
    "Mobile App": ["mobile", "app", "ios", "android", "phone", "smartphone", "download", "install"],
    "Desktop Application": ["desktop", "windows", "mac", "linux", "software", "program", "exe", "application crash"],
    "Network": ["internet", "wifi", "connection", "slow", "disconnect", "firewall", "dns", "router", "lan", "wan"],
    "Email": ["outlook", "gmail", "mail", "exchange", "smtp", "attachments", "sent", "draft"],
    "Login/Access": ["login", "password", "reset", "unlock", "access", "account", "permission", "authentication"],
    "Security": ["virus", "malware", "hack", "breach", "unauthorized", "phishing", "ransomware", "firewall"],
    "Hardware": ["computer", "laptop", "printer", "keyboard", "mouse", "monitor", "usb", "disk", "drives"],
    "Database": ["database", "db", "query", "timeout", "corruption", "backup", "restore", "sql"],
    "System/Infrastructure": ["server", "disk", "memory", "cpu", "outage", "maintenance", "system", "infrastructure"]
}

# Department classification keywords - backend uses this for automatic classification
DEPARTMENT_KEYWORDS = {
    "IT Support": ["login", "password", "access", "account", "user", "reset", "permissions", "unlock", "desktop", "windows", "mac", "linux"],
    "Network": ["internet", "network", "wifi", "connection", "slow", "disconnect", "firewall", "dns", "router", "lan"],
    "Security": ["virus", "malware", "hack", "breach", "unauthorized", "phishing", "ransomware"],
    "Database": ["database", "db", "query", "timeout", "corruption", "backup", "restore", "sql"],
    "Application": ["app", "application", "software", "bug", "crash", "error", "freeze", "slow response", "mobile", "ios", "android"],
    "Infrastructure": ["server", "disk", "memory", "cpu", "outage", "maintenance", "power", "system"]
}

def classify_department(detailed_description: str) -> str:
    text_lower = detailed_description.lower()
    max_matches = 0
    best_dept = "IT Support"
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

def classify_department_with_issue_type(selected_issue: str, detailed_description: str) -> str:
    """
    Classifies department based on both selected issue type and description keywords
    """
    # Start with department mapping based on selected issue type
    issue_to_dept_mapping = {
        "Mobile App": "Application",
        "Desktop Application": "Application",
        "Network": "Network",
        "Email": "IT Support",
        "Login/Access": "IT Support",
        "Security": "Security",
        "Hardware": "IT Support",
        "Database": "Database",
        "System/Infrastructure": "Infrastructure"
    }

    # Get base department from issue type
    base_dept = issue_to_dept_mapping.get(selected_issue, "IT Support")

    # Now check keywords in description to potentially override
    text_lower = detailed_description.lower()
    keyword_matches = {}
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_matches[dept] = matches

    # If any department has more keyword matches than the selected issue type's base dept
    # and has at least 2 more matches, override the classification
    base_matches = keyword_matches.get(base_dept, 0)
    best_keyword_dept = max(keyword_matches, key=keyword_matches.get)
    best_matches = keyword_matches[best_keyword_dept]

    if best_matches >= base_matches + 2 and best_matches > 0:
        return best_keyword_dept
    else:
        return base_dept

def login_page():
    """Display login page"""
    st.title("🔐 Login to Incident Management System")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username.strip() and password.strip():
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Please enter both username and password")

def main():
    st.set_page_config(
        page_title="Incident Management Chat Bot",
        page_icon="🛠️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Check if user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
        return

    st.title("🛠️ Incident Management System")
    st.markdown(f"Welcome, **{st.session_state.username}**! Provide the incident details below and we'll process it automatically.")

    # Sidebar for logout
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

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

        # Issue type selection (replaces department selection)
        issue_options = list(ISSUE_CATEGORIES.keys())
        issue_with = st.selectbox("Issue with", issue_options, index=0)

        submit_button = st.form_submit_button(label="Submit Incident")

    if submit_button and detailed_description.strip():
        # Classify department based on both issue type selection and description
        department = classify_department_with_issue_type(issue_with, detailed_description)

        # Add user message to history
        user_msg = f"**Description:** {description}\n\n**Detailed Description:** {detailed_description}\n\n**Issue with:** {issue_with}\n\n**Department (Auto-classified):** {department}"
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

        # Process the incident
        with st.spinner("🔄 Processing incident..."):
            description = description.strip() or generate_short_description(detailed_description)
            incident_num = f"INC-{int(time.time())}"
            customer_name = st.session_state.username  # Use logged-in username
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
