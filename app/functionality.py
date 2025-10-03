from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import sqlite3
import logging
import json
import datetime
import time

logger = logging.getLogger(__name__)

# Import all helper functions and constants from main.py
from . import main as main_module

# Initialize system on startup
logger.info("Initializing system on startup")
main_module.initialize_system()
logger.info("System initialization completed")

app = FastAPI(title="Incident Solution API")

# ---- Pydantic Model ---- #
class IncidentRequest(BaseModel):
    incident_num: str
    customer_name: str
    organization: str
    department: str
    description: str
    detailed_description: str
    reported_date: str


# ---- DB Helpers (thin wrappers using main.DB_FILE) ---- #
def save_incident_full(incident: IncidentRequest, solution: str, priority: int, embedding_json: str):
    try:
        logger.info(f"Saving incident {incident.incident_num} to database")
        conn = sqlite3.connect(main_module.DB_FILE)
        cursor = conn.cursor()
        text_emb = f"{incident.description} {incident.detailed_description}"
        emb = main_module.get_embedding(text_emb)
        emb_json = json.dumps(emb)
        cursor.execute("""
            INSERT INTO incidents
            (Incident_Number, Description, Detailed_Description, Reported_Date, Resolution, Computed_Priority, Description_Embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            incident.incident_num,
            incident.description,
            incident.detailed_description,
            incident.reported_date,
            solution,
            priority,
            emb_json
        ))
        conn.commit()
        conn.close()
        logger.info(f"Successfully saved incident {incident.incident_num}")
    except Exception as e:
        logger.error(f"Failed to save incident {incident.incident_num}: {e}")
        raise

    if priority >= 4:
        logger.warning(f"High priority incident {incident.incident_num} (priority {priority}) - sending email notification")
        subject = f"HIGH PRIORITY Incident: {incident.incident_num}"
        body = f"""
        Incident Number: {incident.incident_num}
        Description: {incident.description}
        Detailed: {incident.detailed_description}
        Priority: {priority}
        Solution: {solution}
        """
        main_module.send_email_notification(subject, body)


# ---- FastAPI Endpoint ---- #
@app.post("/get_solution")
def get_solution(incident: IncidentRequest):
    request_start = time.time()
    logger.info(f"Started processing incident {incident.incident_num} at {datetime.datetime.now()}")

    try:
        text = f"{incident.description} {incident.detailed_description}"

        # Try to find similar incident
        find_start = time.time()
        similar = main_module.find_solution_vector(text)
        find_duration = time.time() - find_start
        logger.info(f"Vector search completed in {find_duration:.2f} seconds")

        if similar:
            solution = similar[0]
            logger.info("Reused existing solution.")
        else:
            logger.info("No similar solution found, generating new solution")
            gen_start = time.time()
            solution = main_module.generate_solution(incident.description, incident.detailed_description)
            gen_duration = time.time() - gen_start
            logger.info(f"AI generation completed in {gen_duration:.2f} seconds - Generated new solution.")

        # Calculate priority
        pri_start = time.time()
        priority = main_module.calculate_priority(incident.description, incident.detailed_description)
        pri_duration = time.time() - pri_start
        logger.info(f"Priority calculation completed in {pri_duration:.2f} seconds - Priority: {priority}")

        # Save to DB and send email if high priority
        save_start = time.time()
        save_incident_full(incident, solution, priority, "")
        save_duration = time.time() - save_start
        logger.info(f"Incident save completed in {save_duration:.2f} seconds")

        total_duration = time.time() - request_start
        logger.info(f"Request completed for {incident.incident_num} at {datetime.datetime.now()}, total duration: {total_duration:.2f} seconds")
        return {"priority": priority, "solution": solution}
    except Exception as e:
        total_duration = time.time() - request_start
        logger.error(f"Request failed for incident {incident.incident_num} after {total_duration:.2f} seconds: {e}")
        raise
