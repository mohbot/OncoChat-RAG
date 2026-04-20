"""
Cancer Drugs RAG Chat System

A RAG system for answering questions about FDA-approved cancer drugs.
Uses PDF drug labels from the FDA, sectioned by canonical headers,
embedded with sentence-transformers, stored in FAISS, and answered
via a local Ollama LLM.

Components:
  - PDF ingestion & chunking  : pypdf
  - Embeddings                : sentence-transformers (all-MiniLM-L6-v2)
  - Vector store              : FAISS (faiss-cpu)
  - LLM                       : Ollama (local, default gemma3:4b)
"""

import json
import logging
import os
import pickle
import re
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CANONICAL_HEADERS = [
    "Adverse Reactions",
    "Use in Specific Populations",
    "Indications and Usage",
    "Dosage and Administration",
    "Dosage Forms and Strengths",
    "Contraindications",
    "Warnings and Precautions",
    "Overdosage",
    "Description",
    "Clinical Pharmacology",
    "Nonclinical Toxicology",
    "Clinical Studies",
    "How Supplied/Storage and Handling",
    "Patient Counseling Information",
    "Drug Interactions",
]

# FDA label section numbers mapped to canonical headers
SECTION_NUMBER_MAP: Dict[str, int] = {
    "Indications and Usage": 1,
    "Dosage and Administration": 2,
    "Dosage Forms and Strengths": 3,
    "Contraindications": 4,
    "Warnings and Precautions": 5,
    "Adverse Reactions": 6,
    "Drug Interactions": 7,
    "Use in Specific Populations": 8,
    "Overdosage": 10,
    "Description": 11,
    "Clinical Pharmacology": 12,
    "Nonclinical Toxicology": 13,
    "Clinical Studies": 14,
    "How Supplied/Storage and Handling": 16,
    "Patient Counseling Information": 17,
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_CHUNK_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"
TOP_K = 8
PDF_DIR = "drug_reports"
INDEX_DIR = "index_store"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunks_metadata.pkl"
MANIFEST_FILE = "index_manifest.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    drug_name: str
    section_name: str
    source_file: str
    chunk_index: int


# ===================================================================
# Section 2: PDF Parsing & Section Detection
# ===================================================================

def extract_drug_name(filename: str) -> str:
    """Extract the drug name from an FDA label filename.

    Typical patterns:
      osimertinib_2022_208065s025lbl.pdf
      darzalex faspro_2022_761145s012lbl.pdf
      bicalutamide_2009_079185s000lbl.pdf
      busulfan_pre96_09386slr013_tabloid_lbl.pdf
    """
    stem = Path(filename).stem  # drop .pdf
    # Match up to the first underscore followed by a year or 'pre'
    m = re.match(r"^(.+?)_(?:(?:20|19)\d{2}|pre\d+)_", stem)
    if m:
        return m.group(1).strip().title()
    # Fallback: use everything before the first underscore
    return stem.split("_")[0].strip().title()


def extract_pdf_text(filepath: str) -> Optional[str]:
    """Extract all text from a PDF file, returning None on failure."""
    try:
        reader = PdfReader(filepath)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            log.warning("No text extracted from %s", filepath)
            return None
        return "\n".join(pages)
    except Exception as e:
        log.warning("Failed to read PDF %s: %s", filepath, e)
        return None


def _build_section_patterns() -> List[Tuple[str, re.Pattern]]:
    """Build regex patterns for detecting canonical header sections.

    Handles three FDA label formats:
      Modern numbered:   '5 WARNINGS AND PRECAUTIONS'
      Dot numbered:      '5. WARNINGS AND PRECAUTIONS'
      Unnumbered:        'WARNINGS AND PRECAUTIONS' (on its own line)
    """
    patterns = []
    for header in CANONICAL_HEADERS:
        header_upper = re.escape(header.upper())
        num = SECTION_NUMBER_MAP.get(header)
        if num is not None:
            # Match numbered (with or without dot) or unnumbered
            pat = re.compile(
                rf"(?:^|\n)"                      # start of line
                rf"[ \t]*"                         # optional leading whitespace
                rf"(?:{num}\.?\s+)?"               # optional section number
                rf"{header_upper}"                 # the header text
                rf"(?:\s+\d+)?"                    # optional trailing page number
                rf"[ \t]*(?:\n|$)",                # end of line
                re.IGNORECASE,
            )
        else:
            pat = re.compile(
                rf"(?:^|\n)[ \t]*{header_upper}(?:\s+\d+)?[ \t]*(?:\n|$)",
                re.IGNORECASE,
            )
        patterns.append((header, pat))
    return patterns


_SECTION_PATTERNS = _build_section_patterns()


def _find_body_start(text: str) -> int:
    """Find where the actual prescribing information body begins.

    FDA labels have: highlights -> TOC ("FULL PRESCRIBING INFORMATION: CONTENTS")
    -> body ("FULL PRESCRIBING INFORMATION"). We want the LAST standalone
    "FULL PRESCRIBING INFORMATION" (not followed by CONTENTS) as the body start.
    Returns the position of the newline before the body marker so that section 1
    headers immediately following the marker are still discoverable.
    """
    body_start = 0
    for m in re.finditer(
        r"(?:^|\n)\s*FULL PRESCRIBING INFORMATION\s*(?:\n|$)",
        text,
        re.IGNORECASE,
    ):
        body_start = m.start()
    return body_start


def detect_sections(text: str) -> List[Tuple[int, str]]:
    """Detect canonical sections in FDA label text.

    Returns a sorted list of (char_position, header_name).
    Filters out cross-references (e.g., 'see Adverse Reactions (6.1)').
    Uses the LAST valid match per header to skip TOC entries and find
    the actual section content.
    """
    body_start = _find_body_start(text)

    sections: List[Tuple[int, str]] = []
    for header, pattern in _SECTION_PATTERNS:
        best_match = None
        for match in pattern.finditer(text, pos=body_start):
            # Filter cross-references: text right after header starts with ( or [
            after = text[match.end(): match.end() + 30]
            if re.match(r"\s*[\(\[]", after):
                continue
            best_match = match
        if best_match is not None:
            sections.append((best_match.start(), header))

    sections.sort(key=lambda x: x[0])
    return sections


# ===================================================================
# Section 3: Chunking Engine
# ===================================================================

def _split_text_into_chunks(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If a single paragraph exceeds max_chars, split on sentences
        if len(para) > max_chars:
            sentences = re.split(r"(?<=\.)\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_chars:
                    current = f"{current} {sent}".strip() if current else sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            continue

        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:]
                current = f"{overlap_text} {para}".strip()
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    text: str,
    drug_name: str,
    source_file: str,
    sections: List[Tuple[int, str]],
) -> List[Chunk]:
    """Chunk a document using detected canonical sections."""
    all_chunks: List[Chunk] = []

    if not sections:
        # Fallback: blind chunking
        raw_chunks = _split_text_into_chunks(text)
        for i, chunk_text in enumerate(raw_chunks):
            prefixed = f"[Drug: {drug_name} | Section: General]\n{chunk_text}"
            all_chunks.append(
                Chunk(
                    text=prefixed,
                    drug_name=drug_name,
                    section_name="General",
                    source_file=source_file,
                    chunk_index=i,
                )
            )
        return all_chunks

    for idx, (pos, header) in enumerate(sections):
        # Section text runs from this header to the next (or end of doc)
        end_pos = sections[idx + 1][0] if idx + 1 < len(sections) else len(text)
        section_text = text[pos:end_pos].strip()

        # Remove the header line itself from the section body
        first_newline = section_text.find("\n")
        if first_newline > 0:
            section_text = section_text[first_newline:].strip()

        if not section_text:
            continue

        raw_chunks = _split_text_into_chunks(section_text)
        for i, chunk_text in enumerate(raw_chunks):
            prefixed = f"[Drug: {drug_name} | Section: {header}]\n{chunk_text}"
            all_chunks.append(
                Chunk(
                    text=prefixed,
                    drug_name=drug_name,
                    section_name=header,
                    source_file=source_file,
                    chunk_index=i,
                )
            )

    return all_chunks


def process_single_pdf(filepath: str) -> List[Chunk]:
    """Process a single FDA drug label PDF into chunks."""
    filename = os.path.basename(filepath)
    drug_name = extract_drug_name(filename)
    text = extract_pdf_text(filepath)
    if text is None:
        return []

    sections = detect_sections(text)
    return chunk_document(text, drug_name, filename, sections)


def process_all_pdfs(pdf_dir: str) -> List[Chunk]:
    """Process all PDF files in a directory."""
    pdf_path = Path(pdf_dir)
    pdf_files = sorted(pdf_path.glob("*.pdf"))
    total = len(pdf_files)
    log.info("Found %d PDF files in %s", total, pdf_dir)

    all_chunks: List[Chunk] = []
    failed = 0

    for i, fpath in enumerate(pdf_files, 1):
        if i % 25 == 0 or i == total:
            log.info("Processing PDF %d/%d: %s", i, total, fpath.name)
        chunks = process_single_pdf(str(fpath))
        if not chunks:
            failed += 1
        all_chunks.extend(chunks)

    log.info(
        "Done: %d PDFs processed, %d failed, %d total chunks (avg %.1f chunks/doc)",
        total - failed,
        failed,
        len(all_chunks),
        len(all_chunks) / max(total - failed, 1),
    )
    return all_chunks


# ===================================================================
# Section 4: Embedding & FAISS Vector Store
# ===================================================================

class VectorStore:
    """Manages embeddings and FAISS index with disk persistence."""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []
        self._embedder: Optional[SentenceTransformer] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            log.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def build_index(self, chunks: List[Chunk], batch_size: int = 256) -> None:
        """Embed all chunks and build a FAISS index."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        log.info("Embedding %d chunks (batch_size=%d)...", len(texts), batch_size)

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        log.info("FAISS index built with %d vectors", self.index.ntotal)

    def save(self) -> None:
        """Persist index, metadata, and manifest to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_dir / INDEX_FILE))
        with open(self.index_dir / METADATA_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

        manifest = {
            "version": 1,
            "num_chunks": len(self.chunks),
            "build_timestamp": time.time(),
        }
        with open(self.index_dir / MANIFEST_FILE, "w") as f:
            json.dump(manifest, f, indent=2)

        log.info("Index saved to %s/", self.index_dir)

    def load(self) -> bool:
        """Load a previously saved index. Returns True on success."""
        idx_path = self.index_dir / INDEX_FILE
        meta_path = self.index_dir / METADATA_FILE

        if not idx_path.exists() or not meta_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)

            if self.index.ntotal != len(self.chunks):
                log.warning("Index/metadata mismatch — rebuild required")
                return False

            log.info(
                "Loaded index: %d vectors, %d chunks",
                self.index.ntotal,
                len(self.chunks),
            )
            return True
        except Exception as e:
            log.warning("Failed to load index: %s", e)
            return False

    def search(
        self, query: str, top_k: int = TOP_K
    ) -> List[Tuple[Chunk, float]]:
        """Search the index for chunks most relevant to the query."""
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype(
            np.float32
        )
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


# ===================================================================
# Section 5: LLM Generation via Ollama
# ===================================================================

class OllamaClient:
    """Communicates with a local Ollama server via HTTP."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
    ):
        self.base_url = base_url
        self.model = model

    def check_connection(self) -> bool:
        """Verify Ollama is running and the model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            model_names = [m.get("name", "") for m in data.get("models", [])]
            # Check if our model (with or without :latest tag) exists
            if not any(self.model in name for name in model_names):
                log.warning(
                    "Model '%s' not found. Available: %s. "
                    "Pull it with: ollama pull %s",
                    self.model,
                    ", ".join(model_names) or "(none)",
                    self.model,
                )
                return False
            return True
        except Exception as e:
            log.error(
                "Cannot connect to Ollama at %s: %s\n"
                "Start it with: ollama serve",
                self.base_url,
                e,
            )
            return False

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream a response from Ollama's /api/generate endpoint."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.URLError as e:
            yield f"\n[Error communicating with Ollama: {e}]"
        except Exception as e:
            yield f"\n[Unexpected error: {e}]"


# ===================================================================
# Section 6: RAG Orchestrator
# ===================================================================

SYSTEM_PROMPT = """\
You are a knowledgeable pharmacology assistant specializing in FDA-approved cancer drugs.
Use ONLY the provided context from FDA drug labels to answer the question.
Always cite the drug name and label section your answer comes from.
If the context does not contain enough information to answer, say so clearly.
Be precise and concise.\
"""


def format_context(results: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    parts = []
    for chunk, score in results:
        parts.append(
            f"--- Source: {chunk.drug_name} | Section: {chunk.section_name} "
            f"(relevance: {score:.3f}) ---\n{chunk.text}"
        )
    return "\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """Build the full prompt for the LLM."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )


class CancerDrugRAG:
    """Main orchestrator for the cancer drugs RAG system."""

    def __init__(
        self,
        pdf_dir: str = PDF_DIR,
        index_dir: str = INDEX_DIR,
        ollama_model: str = OLLAMA_MODEL,
        top_k: int = TOP_K,
    ):
        self.pdf_dir = pdf_dir
        self.index_dir = index_dir
        self.top_k = top_k
        self.store = VectorStore(index_dir)
        self.llm = OllamaClient(model=ollama_model)
        self.last_results: List[Tuple[Chunk, float]] = []

    def initialize(self, force_rebuild: bool = False) -> None:
        """Load or build the vector index."""
        if not force_rebuild and self.store.load():
            print(
                f"Loaded existing index: {self.store.index.ntotal} chunks "
                f"from {self.index_dir}/",
                file=sys.stderr,
            )
        else:
            if force_rebuild:
                log.info("Forcing index rebuild...")
            else:
                log.info("No existing index found — building from scratch...")

            chunks = process_all_pdfs(self.pdf_dir)
            if not chunks:
                log.error("No chunks produced. Check the PDF directory: %s", self.pdf_dir)
                sys.exit(1)

            self.store.build_index(chunks)
            self.store.save()

        # Check Ollama
        if not self.llm.check_connection():
            print(
                "\nWarning: Ollama is not available. You can still search "
                "the index, but generation will fail.\n",
                file=sys.stderr,
            )

    def ask(self, question: str) -> Generator[str, None, None]:
        """Answer a question using RAG: retrieve, augment, generate."""
        # Retrieve
        results = self.store.search(question, top_k=self.top_k)
        self.last_results = results

        if not results:
            yield "No relevant information found in the drug labels."
            return

        # Augment
        context = format_context(results)
        prompt = build_prompt(question, context)

        # Generate (stream)
        yield from self.llm.generate_stream(prompt)

    def show_sources(self) -> str:
        """Return a formatted string of the last query's sources."""
        if not self.last_results:
            return "No previous query results."

        lines = ["Last query sources:"]
        for i, (chunk, score) in enumerate(self.last_results, 1):
            lines.append(
                f"  {i}. {chunk.drug_name} — {chunk.section_name} "
                f"(score: {score:.3f}, file: {chunk.source_file})"
            )
        return "\n".join(lines)


# ===================================================================
# Section 7: Interactive Chat Loop & CLI
# ===================================================================


BANNER = """
========================================================
         Cancer Drugs RAG Chat System
  Ask questions about FDA-approved cancer drugs
========================================================
  Commands:
    sources  — show sources for last answer
    rebuild  — rebuild the index from PDFs
    help     — show this message
    quit     — exit
========================================================
"""


def run_chat(rag: CancerDrugRAG) -> None:
    """Run the interactive chat loop."""
    print(BANNER)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        cmd = question.lower()
        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        elif cmd == "help":
            print(BANNER)
            continue
        elif cmd == "sources":
            print(rag.show_sources())
            continue
        elif cmd == "rebuild":
            print("Rebuilding index...")
            rag.initialize(force_rebuild=True)
            print("Index rebuilt successfully.")
            continue

        # Stream the answer
        print("\nAssistant: ", end="", flush=True)
        for token in rag.ask(question):
            print(token, end="", flush=True)
        print()  # newline after streamed response


def main() -> None:
    """Entry point with CLI argument handling."""
    # Simple arg parsing (no argparse to keep it lightweight)
    pdf_dir = PDF_DIR
    index_dir = INDEX_DIR
    model = OLLAMA_MODEL
    top_k = TOP_K
    force_rebuild = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--pdf-dir" and i + 1 < len(args):
            pdf_dir = args[i + 1]
            i += 2
        elif args[i] == "--index-dir" and i + 1 < len(args):
            index_dir = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--top-k" and i + 1 < len(args):
            top_k = int(args[i + 1])
            i += 2
        elif args[i] == "--rebuild":
            force_rebuild = True
            i += 1
        elif args[i] in ("--help", "-h"):
            print(
                "Usage: python cancer_drugs_chat_RAG.py [OPTIONS]\n\n"
                "Options:\n"
                "  --pdf-dir DIR     Path to PDF directory (default: drug_reports)\n"
                "  --index-dir DIR   Path to index storage (default: index_store)\n"
                "  --model MODEL     Ollama model name (default: gemma3:4b)\n"
                "  --top-k N         Number of chunks to retrieve (default: 8)\n"
                "  --rebuild         Force rebuild the index\n"
                "  -h, --help        Show this help message\n"
            )
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            sys.exit(1)

    rag = CancerDrugRAG(
        pdf_dir=pdf_dir,
        index_dir=index_dir,
        ollama_model=model,
        top_k=top_k,
    )
    rag.initialize(force_rebuild=force_rebuild)
    run_chat(rag)


if __name__ == "__main__":
    main()
