# Incident Management Tool

## Overview

An AI-powered incident management system that uses machine learning to automatically classify incidents, assign priorities, and provide resolution suggestions. The application consists of a FastAPI backend and a Streamlit frontend for creating and managing IT support tickets.

## Features

- **Automated Incident Classification**: Automatically categorizes incidents into departments (IT Support, Network, Security, Database, Application, Infrastructure)
- **AI-Generated Solutions**: Uses GPT-4 to generate intelligent resolution suggestions
- **Similarity Matching**: Detects similar historical incidents to reuse proven solutions
- **Priority Calculation**: Automatically assigns priority levels (1-5) based on incident content analysis
- **Email Notifications**: Sends alerts for high-priority incidents
- **Embedding-based Search**: Uses vector embeddings for efficient incident similarity matching

## Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) SMTP server credentials for email notifications

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Incident-Management-Tool
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key_here

# Optional email configuration for notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFY_EMAIL=alerts@yourcompany.com
```

## Database Setup

The application uses SQLite for data storage. On first run, the system will automatically:

- Initialize the database schema
- Load historical incident data from `app/Incident Details with REQ and Reason Oct23-Mar24-App-Only.xlsx`
- Compute embeddings for similarity matching

## Running the Application

### Step 1: Start the FastAPI Backend

First, ensure the backend API server is running, as the Streamlit frontend depends on it:

```bash
python run_server.py
```

The FastAPI server will start on `http://localhost:8000`. Keep this terminal running.

### Step 2: Start the Streamlit Frontend

In a separate terminal (with the virtual environment activated):

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app will be available at `http://localhost:8501`.

## Usage

1. **Access the Application**: Open your browser and navigate to `http://localhost:8501`

2. **Login**: Use any username and password to log in (authentication is currently for demo purposes)

3. **Create Incident**: Fill out the incident form with description and detailed description, then select an issue category

4. **Automatic Processing**: The system will:
   - Classify the department automatically
   - Assign a priority level
   - Generate or find a similar solution
   - Create the incident record in the database

## Architecture

- **FastAPI Backend** (`app/functionality.py`): Handles API endpoints and incident processing logic
- **Streamlit Frontend** (`app/streamlit_app.py`): Web interface for incident creation
- **Agent System** (`app/agents.py`): Orchestrates AI agents for similarity, solution generation, priority calculation, and database operations
- **Database** (`quality_incidents.db`): SQLite database storing incident records with embeddings

## Database Schema

The incidents table includes the following columns:

- `id`: Primary key
- `Incident_Number`: Unique incident identifier
- `Description`: Short description
- `Detailed_Description`: Detailed incident description
- `Resolution`: Solution text
- `Reported_Date`: Date when incident was reported
- `Computed_Priority`: Calculated priority (1-5)
- `Description_Embedding`: Vector embedding for similarity search
- `User_Name`: Name of the user who reported the incident
- `Department`: Categorized department

## API Endpoints

- `POST /get_solution`: Process an incident and return priority and solution

## Configuration

- **Similarity Threshold**: Set to 0.95 for high-confidence matching
- **Embedding Model**: Uses `text-embedding-3-small`
- **LLM Model**: Uses `gpt-4o` for solution generation
- **Max Retries**: 3 attempts for API calls with 2-second delays

## Troubleshooting

1. **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **OpenAI API Issues**: Verify your API key in the `.env` file and check your OpenAI account quota

3. **Database Errors**: Ensure the application has write permissions in the directory

4. **Port Conflicts**: If ports 8000 or 8501 are in use, modify the run scripts accordingly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest app/test.py`
5. Submit a pull request

## License

TBD
