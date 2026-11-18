# BSM Hospital RAG Retrieval Studio

A high-performance multi-strategy document retrieval system built with **FastAPI** and a clean, responsive single-page web frontend. The system indexes the official **BSM Hospital Institutional Knowledge Base & Operations Handbook PDF** using **Recursive Character Chunking** and provides instant switching and side-by-side comparison between **BM25 (Sparse)**, **Dense (Semantic)**, and **Hybrid Fusion (RRF & Weighted Combination)** retrieval strategies.

---

## 🎯 Key Features

1. **Document Ingestion & Recursive Chunking**:
   - Parses `BSM_Hospital_RAG_Knowledge_Base.pdf` (15 pages).
   - Strips repeating page banners/headers to maintain high chunk quality.
   - Uses `RecursiveCharacterTextSplitter` with natural text boundaries (`["\n\n", "\n", ". ", " ", ""]`).
   - Generates 69 clean document chunks with tracked metadata (`page`, `section`, `chunk_id`, `char_count`).

2. **Four Retrieval Strategies**:
   - ⚡ **BM25 Sparse Retrieval**: Okapi BM25 keyword matching for exact medical terms, acronyms, and codes.
   - 🧠 **Dense Semantic Retrieval**: Vector embeddings using `all-MiniLM-L6-v2` with Cosine Similarity.
   - 🔀 **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF Score}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{Dense}}(d)}$$
     *(Configurable smoothing constant $k$, default = 60)*
   - ⚖️ **Weighted Score Combination**:
     $$\text{Score}_{\text{weighted}}(d) = \alpha \cdot \text{Dense}_{\text{norm}}(d) + (1 - \alpha) \cdot \text{BM25}_{\text{norm}}(d)$$
     *(Configurable dense weight $\alpha \in [0.0, 1.0]$, default = 0.50)*
   - 📊 **Compare All 4 Side-by-Side**: Live multi-column view showing how BM25, Dense, RRF, and Weighted rank chunks for the same query.

3. **Simple & Intuitive Frontend**:
   - Instant search with keyboard `Enter` support.
   - Curated sample query chips from hospital operations handbook.
   - Dynamic Top-$K$ slider ($1 - 15$).
   - Dynamic $\alpha$ slider with real-time percentage breakdown for Weighted Fusion.
   - Clean result cards displaying rank, primary score, breakdown pills (individual ranks and normalized scores), page number, and section name.

4. **FastAPI Backend**:
   - Fully asynchronous with lifespan startup pre-indexing (queries execute in $<15\text{ ms}$).
   - RESTful endpoints for stats, examples, single-mode search, side-by-side comparison, and dynamic reindexing.

---

## 📂 Project Structure

```
e:\LLM eng\
├── app.py                           # FastAPI application & REST endpoints
├── retrieval_engine.py              # PDF chunker, BM25, Dense embeddings, RRF & Weighted Fusion
├── test_retrieval.py                # Comprehensive test suite (9 unit & integration tests)
├── requirements.txt                 # Dependencies
├── BSM_Hospital_RAG_Knowledge_Base.pdf # Source hospital knowledge base
└── static/
    ├── index.html                   # Clean, modern single-page frontend
    ├── style.css                    # Responsive CSS styles & themes
    └── script.js                    # Interactive search, mode switching & comparison
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Or via Uvicorn:
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 3. Open Web UI
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web frontend UI |
| `GET` | `/api/stats` | Returns total chunks, pages, embedding model name, and ready status |
| `GET` | `/api/examples` | Curated sample queries targeting specific handbook sections |
| `POST` | `/api/search` | Performs retrieval (`bm25`, `dense`, `rrf`, `weighted`, or `compare`) |
| `POST` | `/api/reindex` | Dynamically re-splits and re-indexes with new `chunk_size` and `chunk_overlap` |

### Example Search Payload:
```json
{
  "query": "What should staff do if a laboratory specimen is mislabeled?",
  "mode": "weighted",
  "top_k": 5,
  "alpha": 0.6,
  "rrf_k": 60
}
```

---

## 🧪 Running Tests

A complete automated test suite is included:
```bash
python test_retrieval.py
```
Verifies chunking, sparse search, dense similarity, both fusion algorithms, and FastAPI API routes.
