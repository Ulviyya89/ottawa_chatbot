# Ottawa Newcomer Support Chatbot

A Lambton College group project to help newcomers find information about housing, healthcare, jobs, education, transportation, and community services in Ottawa.

## Group project credit

This is a copy of the local group-project files. Related group repository: https://github.com/gouthamrp576-stack/ottawa_chatbot1

This project was developed collaboratively; this copy does not claim sole authorship. The relationship and file differences with the related repository have not yet been verified.

## Technology

Python, FastAPI, Streamlit, OpenAI, PDF document retrieval, and SQLite vector storage. Embedding and answer generation require an OpenAI API key and an internet connection.

## Setup

From the project root:

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and enter your OpenAI API key.
4. Build the local document database: `python scripts/ingest_local_sqlite.py`
5. Start the backend: `python backend/run_server.py`
6. In a second terminal, start the interface: `streamlit run frontend/streamlit_app.py`

Source PDFs are in `data/raw_sources/`. Ingestion makes OpenAI embedding requests, which may incur charges. The backend runs at http://localhost:8000. API documentation is available at http://localhost:8000/docs.

## Included material

- `backend/`: API, retrieval, generation, and evaluation code
- `frontend/`: Streamlit interface and alternative HTML frontend
- `scripts/`: data preparation, ingestion, and evaluation scripts
- `data/`: source documents and metadata
- `docs/`: reference document and draft project documentation; the proposal PDF, presentation, and architecture image are placeholder files, not finished deliverables
- `deployment/`: placeholder deployment configuration; hosted deployment has not been configured or tested

## Private and generated files

API keys, virtual environments, local databases, and saved chat sessions are excluded from version control. The original local folder is unchanged.

## Verification

Run `python scripts/run_tests.py` for 11 offline regression tests. These cover API health, message validation, missing configuration, missing documents, simulated AI answers and failures, feedback compatibility, category mapping, and SQLite retrieval. No API key or paid requests are required.

The Streamlit interface was also started using its app testing framework without exceptions. All 37 source PDFs contain extractable text, and Python and JSON syntax checks pass. Live OpenAI requests and real-answer quality have not yet been verified. Feedback is acknowledged by the backend but is not persisted. Older evaluation scripts and deployment files still contain placeholders.
