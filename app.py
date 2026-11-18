import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retrieval_engine import RetrievalEngine

# Path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "BSM_Hospital_RAG_Knowledge_Base.pdf")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Global retrieval engine instance
engine = RetrievalEngine(
    pdf_path=PDF_PATH,
    model_name="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=80,
)


def ensure_engine_ready():
    """Ensure engine is initialized even outside lifespan (e.g. during test runs)."""
    if not engine.is_ready:
        print("Initializing RAG Retrieval Engine...")
        t0 = time.time()
        engine.initialize()
        print(f"Retrieval Engine ready in {time.time() - t0:.2f} seconds.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_engine_ready()
    yield
    print("Shutting down Retrieval Engine...")


app = FastAPI(
    title="BSM Hospital RAG Retrieval Engine",
    description="Multi-mode Retrieval Engine supporting BM25, Dense Retrieval, and Hybrid Fusion (RRF & Weighted Combination)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    mode: Literal["bm25", "dense", "rrf", "weighted", "compare"] = Field(
        "bm25", description="Retrieval mode: bm25, dense, rrf, weighted, or compare"
    )
    top_k: int = Field(5, ge=1, le=50, description="Number of results to retrieve")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for Dense in Weighted Fusion")
    rrf_k: int = Field(60, ge=1, le=200, description="Smoothing constant k for RRF Fusion")


class ReindexRequest(BaseModel):
    chunk_size: int = Field(500, ge=100, le=3000)
    chunk_overlap: int = Field(80, ge=0, le=1000)


@app.get("/api/stats")
async def get_stats():
    """Returns status and corpus statistics."""
    ensure_engine_ready()
    return engine.get_stats()


@app.get("/api/examples")
async def get_examples():
    """Curated list of hospital-specific RAG test queries."""
    return [
        {
            "category": "Emergency & Triage",
            "query": "What are the emergency department triage priority levels?",
            "target": "Section 3.2 – Emergency triage overview",
        },
        {
            "category": "Laboratory & Specimens",
            "query": "What should staff do if a laboratory specimen is mislabeled?",
            "target": "Section 4.2 – Sample collection guidance",
        },
        {
            "category": "Appointments & Scheduling",
            "query": "What is the policy for late arrival and cancellation of appointments?",
            "target": "Section 2.4 – Cancellation and late arrival",
        },
        {
            "category": "Staff & Operations",
            "query": "Who is responsible for managing pharmacy operations and staffing?",
            "target": "Section 7.3 – Staff roles and directory",
        },
        {
            "category": "Information Security & IT",
            "query": "How does the hospital handle suspected patient data privacy incidents and IT downtime?",
            "target": "Sections 7.5 & 7.6 – IT data privacy and downtime",
        },
        {
            "category": "Billing & Financial",
            "query": "What is the procedure for billing disputes, refunds, and payment corrections?",
            "target": "Section 6.5 – Refund and correction policy",
        },
    ]


@app.post("/api/search")
async def search(req: SearchRequest):
    """Executes search with the chosen retrieval strategy."""
    ensure_engine_ready()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t0 = time.time()
    mode = req.mode

    if mode == "bm25":
        results = engine.search_bm25(query=query, top_k=req.top_k)
    elif mode == "dense":
        results = engine.search_dense(query=query, top_k=req.top_k)
    elif mode == "rrf":
        results = engine.search_rrf(query=query, top_k=req.top_k, rrf_k=req.rrf_k)
    elif mode == "weighted":
        results = engine.search_weighted(query=query, top_k=req.top_k, alpha=req.alpha)
    elif mode == "compare":
        comparison = engine.search_all_modes(
            query=query, top_k=req.top_k, alpha=req.alpha, rrf_k=req.rrf_k
        )
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "query": query,
            "mode": "compare",
            "latency_ms": latency_ms,
            "top_k": req.top_k,
            "comparison": comparison,
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    latency_ms = round((time.time() - t0) * 1000, 2)
    return {
        "query": query,
        "mode": mode,
        "latency_ms": latency_ms,
        "top_k": req.top_k,
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/reindex")
async def reindex(req: ReindexRequest):
    """Dynamically re-chunks and re-indexes the document."""
    t0 = time.time()
    engine.reindex(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
    duration = round(time.time() - t0, 2)
    return {
        "status": "success",
        "duration_seconds": duration,
        "stats": engine.get_stats(),
    }


# Mount static directory
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serves the single page frontend application."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "BSM Hospital Retrieval API is running. Place index.html into the static directory."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
