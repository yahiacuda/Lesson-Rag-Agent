# RAG-Enhanced Quiz and Assessment Generator

**CMPE 682/683/782/783 — Assignment 2 | Track A: RAG Implementation**
**Authors:** Walid Ben Ali, Yahia Boray

Portfolio note: this fork is maintained under Yahia Boray's GitHub account as
implementation evidence for the RAG lesson-generation, quiz-generation, and
evaluation work. The original coursework collaboration is credited above.

A Retrieval-Augmented Generation (RAG) system that generates curriculum-grounded quiz questions from lesson objectives, grounded in a corpus of 16 OpenStax textbooks (22,066 indexed chunks). Built with FastAPI, Ollama, Qdrant, and Streamlit.

---

## Architecture

```
Teacher Input (Lesson Objectives)
           |
           v
  [Subject Inference]  ← keyword matching
           |
     Subject in corpus?
     /              \
   Yes               No
    |                 |
    v                 v
[Qdrant Retrieval]  [Fallback Mode]
(dense or hybrid     (prompt-only,
 dense+BM25+RRF)      no citations)
    |
    v
[Prompt Construction]
  [Source 1]...[Source N] injected
  CoT + Input Quality Gate
  Citation rule + Bloom's Taxonomy
    |
    v
[Ollama Generation]  ← qwen3.5:27b (local)
    |
    v
[Citation Extraction + References Footer]
    |
    v
Grounded Quiz with [Source N] inline citations
```

### Component Map

```
app/
├── main.py                        # FastAPI: /quiz/start, /quiz/status, /quiz/generate
├── schemas.py                     # QuizRequest / QuizResponse Pydantic models
└── services/
    ├── quiz_pipeline.py           # Full RAG quiz pipeline (retrieve → prompt → generate)
    ├── quiz_prompt_builder.py     # Grounded + fallback prompt templates (CoT + Input Gate)
    ├── retriever.py               # Dense + hybrid BM25+RRF retrieval with metadata filters
    ├── bm25_index.py              # Lazy-loaded BM25 singleton (persisted to disk)
    ├── ollama_client.py           # Ollama API client (generate + embed)
    └── qdrant_client.py           # Qdrant connection and collection management
streamlit_quiz_app.py              # Streamlit frontend with conversational clarification
```

---

## Features

- **RAG-Grounded Questions** — Questions drawn from retrieved OpenStax textbook passages with inline `[Source N]` citations
- **Hybrid Retrieval** — Dense vector search + BM25 lexical search merged with Reciprocal Rank Fusion (RRF, k=60)
- **Bloom's Taxonomy Tagging** — Each question maps to a cognitive level (Remember → Create)
- **Conversational Clarification** — If input is vague, the app enters chat mode so the teacher can refine the request
- **Fallback Mode** — Degrades gracefully to prompt-only generation when the topic is outside the corpus
- **Three Question Types** — MCQ (with distractors), Short Answer, Open-Ended
- **Download** — Export generated quiz as `.txt`

---

## Tech Stack

| Component | Technology |
| --- | --- |
| Backend API | FastAPI (async job pattern) |
| LLM | Ollama — `qwen3.5:27b` (local, no API cost) |
| Embeddings | Ollama — `nomic-embed-text` |
| Vector Database | Qdrant Cloud |
| Lexical Search | BM25 (`rank-bm25`) merged via RRF |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- [Qdrant Cloud](https://qdrant.tech/) account (free tier works)

### 1. Clone the Repository

```bash
git clone https://github.com/yahiacuda/Lesson-Rag-Agent.git
cd Lesson-Rag-Agent
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama Models

```bash
ollama pull qwen3.5:27b        # generation model
ollama pull nomic-embed-text   # embedding model
```

### 5. Configure Environment

Create a `.env` file (copy from `.env.example`):

```env
OLLAMA_BASE_URL=http://localhost:11434/api
OLLAMA_GENERATION_MODEL=qwen3.5:27b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key_here
QDRANT_COLLECTION=lesson_docs

CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100
TOPIC_DEFAULT=general
LANGUAGE_DEFAULT=en
EMBEDDING_BATCH_SIZE=64
QDRANT_UPSERT_BATCH_SIZE=128
```

### 6. (Optional) Ingest Your Own Documents

Place PDFs in `data/raw/`, add entries to `data/raw/metadata.json`, then run:

```bash
python -m app.ingestion.run_ingestion
```

The pre-built corpus (16 OpenStax textbooks, 22,066 chunks) is already indexed in Qdrant Cloud and does not need to be re-ingested.

### 7. Start the Backend

```bash
uvicorn app.main:app --reload
```

API runs at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 8. Start the Quiz App

```bash
streamlit run streamlit_quiz_app.py
```

Runs at `http://localhost:8501`. Requires the FastAPI backend to be running.

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/quiz/start` | Submit async quiz job — returns `{"job_id": "..."}` |
| `GET` | `/quiz/status/{job_id}` | Poll job status (`processing` / `done` / `error`) |
| `POST` | `/quiz/generate` | Synchronous quiz generation (small requests) |
| `GET` | `/health` | Health check |

### Example

```bash
# Submit job
curl -X POST http://localhost:8000/quiz/start \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Students will understand atomic structure: protons, neutrons, electrons, atomic number, mass number, and isotopes.",
    "subject": "chemistry",
    "grade_level": "High School",
    "num_questions": 5,
    "difficulty": "Mixed",
    "question_types": "MCQ, Short Answer, Open-Ended",
    "retrieval_method": "hybrid"
  }'
# Returns: {"job_id": "...", "status": "processing"}

# Poll result
curl http://localhost:8000/quiz/status/<job_id>
```

### Request Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | string | required | Lesson objectives or content description |
| `subject` | string | auto-detect | Subject override |
| `grade_level` | string | null | Grade level override |
| `num_questions` | int | 5 | Number of questions (3–15) |
| `difficulty` | string | `"Mixed"` | `"Easy"`, `"Medium"`, `"Hard"`, `"Mixed"` |
| `question_types` | string | `"MCQ, Short Answer, Open-Ended"` | Comma-separated list |
| `retrieval_limit` | int | 5 | Top-k chunks to retrieve |
| `retrieval_mode` | string | `"auto"` | `"auto"`, `"filtered"`, `"all"` |
| `retrieval_method` | string | `"dense"` | `"dense"` or `"hybrid"` |

---

## Document Corpus

| Subject | Books | Chunks |
| --- | --- | --- |
| Mathematics | Precalculus 2e, Algebra 1, College Algebra 2e, Algebra & Trigonometry 2e, Calculus Vol. 1–3 | 5,078 |
| Physics | College Physics 2e, Physics (OpenStax) | 4,151 |
| Biology | Biology 2e, Concepts of Biology | 3,773 |
| Chemistry | Chemistry 2e, Chemistry: Atoms First 2e | 3,593 |
| Computer Science | Intro to CS, Intro to Python | 1,927 |
| Data Science | Principles of Data Science | 802 |
| **Total** | **16 books** | **22,066 chunks** |

---

## Key Design Decisions

1. **Local model (no API cost)** — Both baseline (A1) and RAG-enhanced (A2) systems use `qwen3.5:27b` via Ollama. This isolates retrieval as the sole experimental variable and eliminates API costs and rate limits.

2. **Hybrid retrieval with RRF** — Dense vector search captures semantic similarity; BM25 captures exact lexical matches (e.g. "atomic number", "photosynthesis"). Reciprocal Rank Fusion (k=60) merges both ranked lists without score normalisation.

3. **Auto subject inference** — The pipeline infers subject from content keywords and checks whether that subject exists in the Qdrant corpus before retrieval. If not found, it skips retrieval and falls back to prompt-only generation rather than retrieving irrelevant chunks.

4. **Conversational clarification** — When the model requests more detail (detected by regex on the output), the Streamlit app switches to a chat-input mode. The user's reply is merged with the original content and re-submitted.

5. **Async job pattern** — `/quiz/start` returns a `job_id` immediately; the frontend polls `/quiz/status/{job_id}` every 2 seconds. This prevents timeout errors on long generation requests.

---

## Evaluation Results (25 Test Cases)

| Criterion | A1 (no RAG) | A2 (RAG) | Δ |
| --- | :---: |:---:| :---: |
| Objective Alignment | 4.56 | 4.40 | −0.16 |
| Question Quality | 4.12 | 4.36 | +0.24 |
| Difficulty Appropriateness | 4.52 | 4.52 | 0.00 |
| **Groundedness** | **1.00** | **4.28** | **+3.28** |
| **Citation Accuracy** | **1.00** | **4.16** | **+3.16** |

All 10 RAG-specific test cases (TC16–TC25) achieved perfect Groundedness (5.00/5.00) with A2.
See [`A2_BenAli_notebook.ipynb`](A2_BenAli_notebook.ipynb) for full evaluation and [`A2_Technical_Report.md`](A2_Technical_Report.md) for the IEEE-format report.

---

## Repository Structure

```
Lesson-Rag-Agent/
├── app/
│   ├── main.py                      # FastAPI endpoints
│   ├── config.py                    # Environment settings
│   ├── schemas.py                   # Pydantic models (QuizRequest, QuizResponse)
│   ├── services/
│   │   ├── quiz_pipeline.py         # RAG quiz pipeline
│   │   ├── quiz_prompt_builder.py   # Prompt templates (grounded + fallback)
│   │   ├── retriever.py             # Dense + hybrid retrieval
│   │   ├── bm25_index.py            # BM25 index (lazy-loaded, persisted)
│   │   ├── ollama_client.py         # Ollama client
│   │   └── qdrant_client.py         # Qdrant client
│   └── ingestion/                   # PDF ingestion pipeline
├── streamlit_quiz_app.py            # Quiz Streamlit frontend
├── A2_BenAli_notebook.ipynb         # Assignment notebook (proposal + evaluation)
├── A2_Technical_Report.md           # IEEE-format technical report
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── .gitignore
└── README.md
```
