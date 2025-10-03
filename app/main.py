import sqlite3
import openai
import os
import time
import logging
import datetime
import pandas as pd
import json
import numpy as np
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import smtplib
from email.mime.text import MIMEText
import concurrent.futures
from functools import partial

logger = logging.getLogger(__name__)

# ---- Setup ---- #
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="incident_solution_retrieval.log",
    filemode="a"
)

logger.info("Starting Incident Management application")

DB_FILE = "quality_incidents.db"  # Use simplified schema
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
SIMILARITY_THRESHOLD = 0.95  # High threshold for "absolutely similar"

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL")

# ---- DB Setup ---- #
def init_db():
    logger.info("Initializing database")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if we need to recreate the table (old schema)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'")
    exists = cursor.fetchone()
    if exists:
        # Check column count
        cursor.execute("PRAGMA table_info(incidents)")
        columns = cursor.fetchall()
        if len(columns) != 8:  # Simplified schema
            logger.info("Dropping old incidents table due to schema mismatch")
            cursor.execute("DROP TABLE incidents")
            conn.commit()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Incident_Number TEXT,
            Description TEXT,
            Detailed_Description TEXT,
            Resolution TEXT,
            Reported_Date TEXT,
            Computed_Priority INTEGER,
            Description_Embedding TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialization completed")

# ---- Priority Matrix ---- #
# Matrix: Keywords grouped by priority levels, impact, urgency
PRIORITY_MATRIX = {
    'critical_keywords': ["critical", "outage", "breach", "security breach", "system down", "unavailable", "total failure"],
    'high_keywords': ["failure", "error", "crash", "data loss", "exfiltration", "vulnerability"],
    'medium_keywords': ["slow", "timeout", "disconnect", "issue", "problem", "bug"],
    'low_keywords': ["request", "access", "enhancement", "minor", "slow response"],
    'impact_multipliers': {  # Optional weights based on services
        'core': 1.2,  # Critical services
        'network': 1.1,
        'database': 1.1,
        'default': 1.0
    }
}

def calculate_priority(description: str, detailed_description: str, service_type: str = "default") -> int:
    text = (description + " " + detailed_description).lower()
    score = 0
    # Base priority from keywords
    for word in PRIORITY_MATRIX['critical_keywords']:
        if word in text:
            score = max(score, 5)
            break
    if score < 5:
        for word in PRIORITY_MATRIX['high_keywords']:
            if word in text:
                score = max(score, 4)
                break
    if score < 4:
        for word in PRIORITY_MATRIX['medium_keywords']:
            if word in text:
                score = max(score, 3)
                break
    if score == 0:
        for word in PRIORITY_MATRIX['low_keywords']:
            if word in text:
                score = max(score, 2)
                break
    if score == 0:
        score = 1  # Lowest default

    # Apply service multiplier
    multiplier = PRIORITY_MATRIX['impact_multipliers'].get(service_type.lower(), 1.0)
    adjusted_score = min(5, int(score * multiplier))
    return max(1, adjusted_score)  # Ensure at least 1

# ---- Embedding Functions ---- #
def get_embedding(text: str) -> List[float]:
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        return []

# ---- Cosine Similarity ---- #
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---- Solution Retrieval with Embeddings ---- #
def find_solution_vector(new_text: str) -> Optional[Tuple[str, int]]:
    start_time = time.time()
    logger.info(f"Starting vector similarity search at {datetime.datetime.now()}")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT Resolution, Computed_Priority, Description_Embedding FROM incidents WHERE Description_Embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Vector similarity search completed at {datetime.datetime.now()}, duration: {duration:.2f} seconds - no historical data")
        return None

    new_embedding = get_embedding(new_text)
    if not new_embedding:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Vector similarity search failed at {datetime.datetime.now()}, duration: {duration:.2f} seconds - embedding generation failed")
        return None

    best_similarity = -1
    best_solution = None
    best_priority = None
    for resolution, priority, emb_str in rows:
        try:
            emb = json.loads(emb_str)
            sim = cosine_similarity(new_embedding, emb)
            if sim > best_similarity and sim >= SIMILARITY_THRESHOLD:
                best_similarity = sim
                best_solution = resolution
                best_priority = priority
        except Exception as e:
            logger.warning(f"Error processing embedding: {e}")
            continue

    end_time = time.time()
    duration = end_time - start_time
    if best_solution:
        logger.info(f"Vector similarity search completed at {datetime.datetime.now()}, duration: {duration:.2f} seconds - similar incident found with similarity {best_similarity:.2f}, priority {best_priority}")
    else:
        logger.info(f"Vector similarity search completed at {datetime.datetime.now()}, duration: {duration:.2f} seconds - no similar incident found")

    return (best_solution, best_priority) if best_solution else None

# ---- Email Notification ---- #
def send_email_notification(subject: str, body: str):
    if not SMTP_EMAIL or not SMTP_PASSWORD or not NOTIFY_EMAIL:
        logger.warning("Email config not complete, skipping notification.")
        return

    try:
        logger.info(f"Sending email notification: {subject}")
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SMTP_EMAIL
        msg['To'] = NOTIFY_EMAIL

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, NOTIFY_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"Email sent to {NOTIFY_EMAIL}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

# ---- Get Solution from LLM ---- #
def generate_solution(description: str, detailed: str) -> str:
    start_time = time.time()
    logger.info(f"Starting AI solution generation at {datetime.datetime.now()}")

    prompt = f"""
    Incident:
    {description}\n{detailed}

    Provide a concise IT support resolution (step-by-step if needed).
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an IT support assistant that provides practical resolutions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"AI solution generation completed at {datetime.datetime.now()}, duration: {duration:.2f} seconds")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Solution generation failed (attempt {attempt+1}): {e}")
            time.sleep(SLEEP_BETWEEN_RETRIES)

    end_time = time.time()
    duration = end_time - start_time
    logger.warning(f"AI solution generation failed after retries at {datetime.datetime.now()}, duration: {duration:.2f} seconds")
    return "No solution could be generated."

# ---- Process Excel ---- #
def process_excel(file_path: str):
    start_time = time.time()
    logger.info(f"Starting Excel processing for {file_path} at {datetime.datetime.now()}")

    try:
        if not os.path.exists(file_path):
            file_path = os.path.join("app", file_path)  # Assuming run from root
        df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"Failed to load Excel file {file_path}: {e}")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if DB has data
    cursor.execute("SELECT COUNT(*) FROM incidents")
    db_count = cursor.fetchone()[0]
    db_empty = db_count == 0

    rows_with_existing = []
    rows_to_generate = []
    all_rows = []
    all_texts = []

    total = len(df)
    logger.info(f"Processing {total} incidents... Preparing for batch processing...")
    print(f"Processing {total} incidents... Preparing for batch processing...")  # Keep for user feedback

    # Collect all row data
    for idx, row in df.iterrows():
        values = (
            str(row.get("Incident Number", "")),
            str(row.get("Description", "")),
            str(row.get("Detailed Decription", "")),
            str(row.get("Resolution", "")),  # Use existing if available
            str(row.get("Reported Date", "")),
        )
        all_rows.append(values)
        all_texts.append(f"{values[1]} {values[2]}")

        if str(row.get("Resolution", "")).strip():
            rows_with_existing.append((values, idx))
        else:
            rows_to_generate.append((values, idx))

    msg = f"Found {len(rows_with_existing)} with existing resolutions, {len(rows_to_generate)} need generation."
    logger.info(msg)
    print(msg)

    # Generate resolutions in parallel for those without
    resolutions = []
    if rows_to_generate:
        logger.info("Generating resolutions in parallel...")
        print("Generating resolutions in parallel...")
        defs = [(vals[1], vals[2]) for vals, idx in rows_to_generate]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  # Adjust workers
            resolutions = list(executor.map(lambda x: generate_solution(x[0], x[1]), defs))

    # Batch embed all texts
    logger.info("Batch embedding all texts...")
    print("Batch embedding all texts...")
    emb_start = time.time()
    all_embeddings = embeddings.embed_documents(all_texts)
    emb_duration = time.time() - emb_start
    logger.info(f"Batch embedding completed in {emb_duration:.2f} seconds")
    print(f"Batch embedding completed in {emb_duration:.2f} seconds")

    # Assemble final records
    new_records = []
    res_idx = 0
    for i, values in enumerate(all_rows):
        if i < len(rows_with_existing):
            # With existing
            resolution = values[3]
        else:
            # From generated
            resolution = resolutions[res_idx]
            res_idx += 1

        computed_priority = calculate_priority(values[1], values[2])

        new_records.append((None, values[0], values[1], values[2], resolution, values[4], computed_priority, json.dumps(all_embeddings[i])))

    logger.info(f"Prepared {len(new_records)} records for insertion.")
    print(f"Prepared {len(new_records)} records for insertion.")

    # Batch insert
    try:
        cursor.executemany("""
            INSERT INTO incidents
            (id, Incident_Number, Description, Detailed_Description, Resolution, Reported_Date, Computed_Priority, Description_Embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, new_records)

        conn.commit()
        conn.close()
        logger.info("Batch insert to database completed")
    except Exception as e:
        logger.error(f"Failed to insert records into database: {e}")
        conn.rollback()
        conn.close()
        raise

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Excel processing completed for {file_path} at {datetime.datetime.now()}, duration: {duration:.2f} seconds")



# ---- Check and Process if Needed ---- #
def initialize_system(excel_path: str = "Incident Details with REQ and Reason Oct23-Mar24-App-Only.xlsx"):
    logger.info(f"Initializing system with Excel path: {excel_path}")
    init_db()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM incidents")
    count = cursor.fetchone()[0]
    conn.close()
    logger.info(f"Database has {count} incidents")
    if count == 0:
        logger.info("Database empty, loading data from Excel...")
        print("Database empty, loading data from Excel...")
        process_excel(excel_path)
        logger.info("Data loaded successfully.")
        print("Data loaded successfully.")
    else:
        logger.info("Database already has data, skipping load.")
        print("Database already has data, skipping load.")

# ---- Runner ---- #
if __name__ == "__main__":
    initialize_system("Incident Details with REQ and Reason Oct23-Mar24-App-Only.xlsx")
    logger.info("System ready.")
    print("System ready.")
