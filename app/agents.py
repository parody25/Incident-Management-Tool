from typing import Dict, Any, Optional, List
import json
import sqlite3
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents in the pipeline."""

    def __init__(self, name: str):
        self.name = name

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent's logic. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run method")

    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]):
        """Validate that required inputs are present."""
        missing = [key for key in required_keys if key not in inputs]
        if missing:
            raise ValueError(f"Agent {self.name} missing required inputs: {missing}")


class SimilarityAgent(BaseAgent):
    """Agent responsible for finding similar incidents based on vector similarity."""

    def __init__(self, db_file: str, embeddings_model, similarity_threshold: float = 0.95):
        super().__init__("SimilarityAgent")
        self.db_file = db_file
        self.embeddings = embeddings_model
        self.similarity_threshold = similarity_threshold

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_inputs(inputs, ['text'])
        text = inputs['text']

        logger.info(f"SimilarityAgent: Starting similarity search for text")

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT Resolution, Computed_Priority, Description_Embedding FROM incidents WHERE Description_Embedding IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.info("SimilarityAgent: No historical data found")
            return {'similar_incident': None}

        # Import get_embedding and cosine_similarity from main
        from .main import get_embedding, cosine_similarity

        new_embedding = get_embedding(text)
        if not new_embedding:
            logger.error("SimilarityAgent: Failed to generate embedding")
            return {'similar_incident': None}

        best_similarity = -1
        best_incident = None

        for resolution, priority, emb_str in rows:
            try:
                emb = json.loads(emb_str)
                sim = cosine_similarity(new_embedding, emb)
                if sim > best_similarity and sim >= self.similarity_threshold:
                    best_similarity = sim
                    best_incident = {'resolution': resolution, 'priority': priority, 'similarity': sim}
            except Exception as e:
                logger.warning(f"SimilarityAgent: Error processing embedding: {e}")
                continue

        result = {'similar_incident': best_incident, 'embedding': new_embedding}
        logger.info(f"SimilarityAgent: Search completed, found similar: {best_incident is not None}")
        return result


class SolutionAgent(BaseAgent):
    """Agent responsible for generating new solutions using LLM."""

    def __init__(self):
        super().__init__("SolutionAgent")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_inputs(inputs, ['description', 'detailed_description'])
        description = inputs['description']
        detailed_description = inputs['detailed_description']

        logger.info("SolutionAgent: Generating new solution")

        # Import generate_solution from main
        from .main import generate_solution

        solution = generate_solution(description, detailed_description)

        logger.info("SolutionAgent: Solution generated")
        return {'solution': solution}


class PriorityAgent(BaseAgent):
    """Agent responsible for calculating incident priority."""

    def __init__(self):
        super().__init__("PriorityAgent")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_inputs(inputs, ['description', 'detailed_description'])
        description = inputs['description']
        detailed_description = inputs['detailed_description']

        logger.info("PriorityAgent: Calculating priority")

        # Import calculate_priority from main
        from .main import calculate_priority

        priority = calculate_priority(description, detailed_description)

        logger.info(f"PriorityAgent: Priority calculated as {priority}")
        return {'priority': priority}


class DatabaseAgent(BaseAgent):
    """Agent responsible for saving incidents to database and handling notifications."""

    def __init__(self, db_file: str):
        super().__init__("DatabaseAgent")
        self.db_file = db_file

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_inputs(inputs, ['incident', 'solution', 'priority', 'embedding'])
        incident = inputs['incident']
        solution = inputs['solution']
        priority = inputs['priority']
        embedding = inputs.get('embedding', [])

        logger.info(f"DatabaseAgent: Saving incident {incident.incident_num}")

        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            emb_json = json.dumps(embedding)
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

            # Check for high priority and notify
            if priority >= 4:
                logger.warning(f"DatabaseAgent: High priority incident {incident.incident_num}")
                # Import send_email_notification from main
                from .main import send_email_notification
                subject = f"HIGH PRIORITY Incident: {incident.incident_num}"
                body = f"""
                Incident Number: {incident.incident_num}
                Description: {incident.description}
                Detailed: {incident.detailed_description}
                Priority: {priority}
                Solution: {solution}
                """
                send_email_notification(subject, body)

            logger.info("DatabaseAgent: Incident saved successfully")
            return {'saved': True}

        except Exception as e:
            logger.error(f"DatabaseAgent: Failed to save incident: {e}")
            raise


class AgentOrchestrator(BaseAgent):
    """Orchestrator that coordinates the agent pipeline."""

    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__("AgentOrchestrator")
        self.agents = agents

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_inputs(inputs, ['incident'])
        incident = inputs['incident']

        # Prepare initial data
        text = f"{incident.description} {incident.detailed_description}"

        # Step 1: Similarity Search
        similarity_result = self.agents['similarity'].run({'text': text})
        similar_incident = similarity_result.get('similar_incident')

        # Step 2: Get Solution - either reuse or generate
        if similar_incident:
            solution = similar_incident['resolution']
            logger.info("Orchestrator: Using existing solution")
        else:
            solution_result = self.agents['solution'].run({
                'description': incident.description,
                'detailed_description': incident.detailed_description
            })
            solution = solution_result['solution']
            logger.info("Orchestrator: Generated new solution")

        # Step 3: Calculate Priority
        priority_result = self.agents['priority'].run({
            'description': incident.description,
            'detailed_description': incident.detailed_description
        })
        priority = priority_result['priority']

        # Step 4: Save to Database
        db_result = self.agents['database'].run({
            'incident': incident,
            'solution': solution,
            'priority': priority,
            'embedding': similarity_result.get('embedding', [])
        })

        # Return final result
        return {
            'solution': solution,
            'priority': priority,
            'saved': db_result.get('saved', False),
            'used_similar': similar_incident is not None
        }
