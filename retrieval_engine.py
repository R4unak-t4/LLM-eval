import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


def clean_page_text(raw_text: str) -> str:
    """Remove repeating PDF header and footer banners to keep chunks pure."""
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Filter out repeating header banners and page indicators
        if re.match(r"^BSM Hospital\s*\|\s*Synthetic RAG Knowledge Base", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Page\s+\d+$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        page: int,
        section: str,
        char_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.page = page
        self.section = section
        self.char_count = char_count
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "page": self.page,
            "section": self.section,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


class RetrievalEngine:
    def __init__(
        self,
        pdf_path: str = "BSM_Hospital_RAG_Knowledge_Base.pdf",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 80,
    ):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: List[DocumentChunk] = []
        self.total_pages = 0
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.dense_model: Optional[SentenceTransformer] = None
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.is_ready = False

    def initialize(self):
        """Extract text from PDF, perform recursive chunking, and build BM25 & Dense indexes."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        print(f"Reading PDF: {self.pdf_path}")
        reader = pypdf.PdfReader(self.pdf_path)
        self.total_pages = len(reader.pages)

        # Recursive character text splitter with multiple natural boundary separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        extracted_chunks: List[DocumentChunk] = []
        chunk_counter = 0
        current_section = "General Information"

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            cleaned = clean_page_text(raw_text)
            if not cleaned:
                continue

            # Update section context if a section header is detected on this page
            section_matches = re.findall(r"^(\d+\.[\d\.]*\s+[^\n]+)", cleaned, re.MULTILINE)
            if section_matches:
                current_section = section_matches[0].strip()

            split_texts = splitter.split_text(cleaned)
            for text_piece in split_texts:
                clean_piece = text_piece.strip()
                if not clean_piece:
                    continue

                # Check if this specific chunk contains a section header
                piece_section = current_section
                inner_section = re.search(r"^(\d+\.[\d\.]*\s+[^\n]+)", clean_piece, re.MULTILINE)
                if inner_section:
                    piece_section = inner_section.group(1).strip()
                    current_section = piece_section

                cid = f"chunk_{chunk_counter:03d}"
                meta = {
                    "source": os.path.basename(self.pdf_path),
                    "page": page_num,
                    "section": piece_section,
                }
                extracted_chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        content=clean_piece,
                        page=page_num,
                        section=piece_section,
                        char_count=len(clean_piece),
                        metadata=meta,
                    )
                )
                chunk_counter += 1

        self.chunks = extracted_chunks
        print(f"Generated {len(self.chunks)} recursive chunks across {self.total_pages} pages.")

        # Build BM25 Sparse Index
        print("Building BM25 index...")
        self.tokenized_corpus = [tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build Dense Index
        print(f"Loading Dense Embedding model '{self.model_name}'...")
        self.dense_model = SentenceTransformer(self.model_name)
        texts = [c.content for c in self.chunks]
        print(f"Computing embeddings for {len(texts)} chunks...")
        self.chunk_embeddings = self.dense_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.is_ready = True
        print("RetrievalEngine fully initialized and ready.")

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using BM25 Okapi sparse keyword search."""
        if not self.is_ready or self.bm25 is None:
            raise RuntimeError("Engine not ready.")

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        raw_scores = np.array(self.bm25.get_scores(q_tokens))
        sorted_indices = np.argsort(raw_scores)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(float(raw_scores[idx]), 4),
                "bm25_score": round(float(raw_scores[idx]), 4),
                "bm25_rank": rank,
                "method": "bm25",
            })
        return results

    def search_dense(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using Dense vector cosine similarity."""
        if not self.is_ready or self.dense_model is None or self.chunk_embeddings is None:
            raise RuntimeError("Engine not ready.")

        q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
        cosine_sims = np.dot(self.chunk_embeddings, q_emb)
        sorted_indices = np.argsort(cosine_sims)[::-1]

        results = []
        for rank, idx in enumerate(sorted_indices[:top_k], start=1):
            chunk = self.chunks[idx]
            sim = float(cosine_sims[idx])
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(sim, 4),
                "dense_score": round(sim, 4),
                "dense_rank": rank,
                "method": "dense",
            })
        return results

    def search_rrf(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_dense(d))
        Where k is the smoothing constant (default: 60).
        """
        if not self.is_ready or self.bm25 is None or self.dense_model is None or self.chunk_embeddings is None:
            raise RuntimeError("Engine not ready.")

        n_docs = len(self.chunks)
        q_tokens = tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens)) if q_tokens else np.zeros(n_docs)

        q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.chunk_embeddings, q_emb)

        # 1-based ranks
        bm25_order = np.argsort(bm25_scores)[::-1]
        dense_order = np.argsort(dense_scores)[::-1]

        bm25_ranks = np.zeros(n_docs, dtype=int)
        dense_ranks = np.zeros(n_docs, dtype=int)

        for pos, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = pos

        for pos, idx in enumerate(dense_order, start=1):
            dense_ranks[idx] = pos

        # Compute RRF score
        rrf_scores = (1.0 / (rrf_k + bm25_ranks)) + (1.0 / (rrf_k + dense_ranks))
        top_indices = np.argsort(rrf_scores)[::-1]

        results = []
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(float(rrf_scores[idx]), 5),
                "rrf_score": round(float(rrf_scores[idx]), 5),
                "bm25_score": round(float(bm25_scores[idx]), 4),
                "bm25_rank": int(bm25_ranks[idx]),
                "dense_score": round(float(dense_scores[idx]), 4),
                "dense_rank": int(dense_ranks[idx]),
                "rrf_k": rrf_k,
                "method": "rrf",
            })
        return results

    def search_weighted(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Weighted Score Fusion:
        Score = alpha * Dense_norm + (1 - alpha) * BM25_norm
        alpha: weight for dense retrieval in [0.0, 1.0]
        """
        if not self.is_ready or self.bm25 is None or self.dense_model is None or self.chunk_embeddings is None:
            raise RuntimeError("Engine not ready.")

        n_docs = len(self.chunks)
        q_tokens = tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens)) if q_tokens else np.zeros(n_docs)

        q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.chunk_embeddings, q_emb)

        # Min-Max Normalization for BM25
        b_min, b_max = float(np.min(bm25_scores)), float(np.max(bm25_scores))
        bm25_norm = (bm25_scores - b_min) / (b_max - b_min) if b_max > b_min else np.zeros(n_docs)

        # Min-Max Normalization for Dense
        d_min, d_max = float(np.min(dense_scores)), float(np.max(dense_scores))
        dense_norm = (dense_scores - d_min) / (d_max - d_min) if d_max > d_min else np.zeros(n_docs)

        # Combine
        alpha_val = max(0.0, min(1.0, float(alpha)))
        weighted_scores = (alpha_val * dense_norm) + ((1.0 - alpha_val) * bm25_norm)

        # Ranks
        bm25_order = np.argsort(bm25_scores)[::-1]
        dense_order = np.argsort(dense_scores)[::-1]
        bm25_ranks = np.zeros(n_docs, dtype=int)
        dense_ranks = np.zeros(n_docs, dtype=int)
        for pos, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = pos
        for pos, idx in enumerate(dense_order, start=1):
            dense_ranks[idx] = pos

        top_indices = np.argsort(weighted_scores)[::-1]

        results = []
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            chunk = self.chunks[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(float(weighted_scores[idx]), 4),
                "weighted_score": round(float(weighted_scores[idx]), 4),
                "dense_norm": round(float(dense_norm[idx]), 4),
                "bm25_norm": round(float(bm25_norm[idx]), 4),
                "dense_score": round(float(dense_scores[idx]), 4),
                "dense_rank": int(dense_ranks[idx]),
                "bm25_score": round(float(bm25_scores[idx]), 4),
                "bm25_rank": int(bm25_ranks[idx]),
                "alpha": alpha_val,
                "method": "weighted",
            })
        return results

    def search_all_modes(self, query: str, top_k: int = 5, alpha: float = 0.5, rrf_k: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        """Runs BM25, Dense, RRF, and Weighted Fusion simultaneously."""
        return {
            "bm25": self.search_bm25(query, top_k),
            "dense": self.search_dense(query, top_k),
            "rrf": self.search_rrf(query, top_k, rrf_k),
            "weighted": self.search_weighted(query, top_k, alpha),
        }

    def reindex(self, chunk_size: int, chunk_overlap: int):
        """Re-runs recursive chunking and rebuilds both BM25 and Dense indexes."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.is_ready = False
        self.initialize()

    def get_stats(self) -> Dict[str, Any]:
        """Return engine metadata and corpus statistics."""
        avg_len = sum(len(c.content) for c in self.chunks) / max(len(self.chunks), 1)
        return {
            "document": os.path.basename(self.pdf_path),
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "avg_chunk_chars": round(avg_len, 1),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.model_name,
            "is_ready": self.is_ready,
        }
