# Medical Entity & Relation Annotation Tool

A lightweight web app for medical subject matter experts to tag entities and relationships in clinical notes. The backend is a FastAPI service with SQLite storage; the frontend is a Vite/React interface.

## Requirements

- Python 3.10+
- Node.js 18+
- (Optional) Local HuggingFace/transformers weights for the suggestion provider

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m backend.init_db  # seeds default project and sample document
uvicorn backend.main:app --reload --port 8000
```


### LLM Suggestions

The default provider expects a locally available llm model (meta-llama/Llama-3.2-3B-Instruct) downloaded from Hugging Face. Configure `backend/config/local_llm.json` with:

```json
{
  "model_path": "../models/your-model",
  "max_new_tokens": 512,
  "device_map": "auto"
}
```

Weights are loaded with `transformers` and run entirely offline. Adjust the config or extend `backend/suggestion_providers.py` for alternative providers.

## Frontend Setup

```bash
cd frontend
npm install
cp .env.template .env.local  # adjust VITE_API_URL if the backend host changes
npm run dev
```

The app runs at http://localhost:5173 and expects the backend on http://localhost:8000 by default.

## Using the App

1. **Select or create a document** – use the seeded sample or paste new clinical text. Titles and annotator names can be edited inline.
2. **Annotate entities** – highlight spans in the read-only note and choose a type. Enable “Auto-label repeats” to tag matching spans across the document automatically.
3. **Manage relations** – link existing entities, with schema validation and cached LLM suggestions for repeated runs.
4. **Leverage suggestions** – click *Generate Suggestions (Local)* to fetch cached or refreshed entity and relationship proposals.
5. **Export** – download the active document or the entire project as JSON directly from the create/select panel.

## Sample Data

The repository ships with `backend/sample_documents/ten_paragraph_note.txt`, ten sample notes (~100 words each) for thorough experimentation. The `backend.init_db` seeding step loads it automatically when the database is empty.

## Project Structure

```
backend/   FastAPI service, ORM models, LLM helper, prompts
frontend/  React UI, Vite config, styling
models/    (optional) local transformer weights
```


