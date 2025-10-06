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
