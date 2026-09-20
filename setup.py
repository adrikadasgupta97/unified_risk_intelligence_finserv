"""
One-time setup: initialise the database and ingest the knowledge base.
Run: python setup.py
"""
from src.data.database import init_db
from src.rag.kb_ingestion import ingest_knowledge_base

print("Initialising database...")
init_db()
print("Database ready.")

print("Ingesting knowledge base into ChromaDB...")
n = ingest_knowledge_base()
print(f"Ingested {n} articles.")

print("\nSetup complete. Run the app with:\n  streamlit run app.py")
