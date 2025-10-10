from fastapi import FastAPI, HTTPException
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
from .agents import AgentOrchestrator, SimilarityAgent, SolutionAgent, PriorityAgent, DatabaseAgent

# Initialize system on startup
logger.info("Initializing system on startup")
main_module.initialize_system()
logger.info("System initialization completed")

# Initialize agent pipeline
logger.info("Setting up agent orchestrator")
similarity_agent = SimilarityAgent(
    db_file=main_module.DB_FILE,
    embeddings_model=main_module.embeddings,
    similarity_threshold=main_module.SIMILARITY_THRESHOLD
)
solution_agent = SolutionAgent()
priority_agent = PriorityAgent()
database_agent = DatabaseAgent(db_file=main_module.DB_FILE)

agents = {
    'similarity': similarity_agent,
    'solution': solution_agent,
    'priority': priority_agent,
    'database': database_agent
}

orchestrator = AgentOrchestrator(agents)
logger.info("Agent orchestrator initialized")

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

class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    resolution: Optional[str] = None

class Incident(BaseModel):
    id: int
    incident_number: str
    description: str
    detailed_description: str
    resolution: str
    reported_date: str
    computed_priority: int
    user_name: str
    department: str
    status: str




# ---- FastAPI Endpoint ---- #
@app.post("/get_solution")
def get_solution(incident: IncidentRequest):
    request_start = time.time()
    logger.info(f"Started processing incident {incident.incident_num} at {datetime.datetime.now()}")

    try:
        # Run the agent pipeline
        result = orchestrator.run({'incident': incident})

        solution = result['solution']
        priority = result['priority']
        used_similar = result['used_similar']

        log_message = "Reused existing solution" if used_similar else "Generated new solution"
        logger.info(log_message)

        total_duration = time.time() - request_start
        logger.info(f"Request completed for {incident.incident_num} at {datetime.datetime.now()}, total duration: {total_duration:.2f} seconds")
        return {"priority": priority, "solution": solution}
    except Exception as e:
        total_duration = time.time() - request_start
        logger.error(f"Request failed for incident {incident.incident_num} after {total_duration:.2f} seconds: {e}")
        raise


@app.get("/incidents", response_model=list[Incident])
def get_incidents(
    department: Optional[str] = None,
    status: Optional[str] = None,
    user_name: Optional[str] = None,
    priority: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    conn = sqlite3.connect(main_module.DB_FILE)
    query = "SELECT id, Incident_Number, Description, Detailed_Description, Resolution, Reported_Date, Computed_Priority, User_Name, Department, Status FROM incidents WHERE 1=1"
    params = []

    if department:
        query += " AND Department = ?"
        params.append(department)
    if status:
        query += " AND Status = ?"
        params.append(status)
    if user_name:
        query += " AND User_Name = ?"
        params.append(user_name)
    if priority:
        query += " AND Computed_Priority = ?"
        params.append(priority)
    if date_from:
        query += " AND Reported_Date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND Reported_Date <= ?"
        params.append(date_to)

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    incidents = []
    for row in rows:
        incidents.append(Incident(
            id=row[0],
            incident_number=row[1],
            description=row[2],
            detailed_description=row[3],
            resolution=row[4],
            reported_date=row[5],
            computed_priority=row[6],
            user_name=row[7],
            department=row[8],
            status=row[9]
        ))
    return incidents


@app.get("/incidents/{incident_num}", response_model=Incident)
def get_incident(incident_num: str):
    conn = sqlite3.connect(main_module.DB_FILE)
    cursor = conn.execute("SELECT id, Incident_Number, Description, Detailed_Description, Resolution, Reported_Date, Computed_Priority, User_Name, Department, Status FROM incidents WHERE Incident_Number = ?", (incident_num,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Incident not found")

    return Incident(
        id=row[0],
        incident_number=row[1],
        description=row[2],
        detailed_description=row[3],
        resolution=row[4],
        reported_date=row[5],
        computed_priority=row[6],
        user_name=row[7],
        department=row[8],
        status=row[9]
    )


@app.put("/incidents/{incident_num}")
def update_incident(incident_num: str, update: IncidentUpdate):
    conn = sqlite3.connect(main_module.DB_FILE)
    updates = []
    params = []

    if update.status is not None:
        updates.append("Status = ?")
        params.append(update.status)
    if update.resolution is not None:
        updates.append("Resolution = ?")
        params.append(update.resolution)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    query = f"UPDATE incidents SET {', '.join(updates)} WHERE Incident_Number = ?"
    params.append(incident_num)

    cursor = conn.execute(query, params)
    conn.commit()
    conn.close()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Incident not found")

    return {"message": "Incident updated"}
