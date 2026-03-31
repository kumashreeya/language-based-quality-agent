"""
rag_guidelines.py — The Rulebook Library

This file manages a searchable database of coding style guides (PDFs like PEP8, Google Java, BARR-C, etc.).
It reads PDF documents, splits them into small chunks, converts them into numerical vectors (embeddings),
and stores them in a ChromaDB vector database so we can quickly look up relevant rules later.

Think of it as a librarian: it reads textbooks, remembers key passages, and finds answers when asked.
"""

import os, json, hashlib, re, glob
from typing import Optional

# --- Optional dependencies (won't crash if not installed yet) ---

# ChromaDB: a vector database that stores text chunks and finds similar ones
try:
    import chromadb
except ImportError:
    chromadb = None

# SentenceTransformer: converts text into numerical vectors (embeddings) so we can compare meaning
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# PyPDF2: reads text from PDF files
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None


# ============================================================
# CONFIGURATION
# ============================================================

RAG_PERSIST_DIR = os.environ.get("RAG_PERSIST_DIR", "./rag_data")  # Where the database is saved on disk
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast model that converts text to numbers
CHUNK_SIZE = 500       # Max words per chunk when splitting documents
CHUNK_OVERLAP = 100    # Overlapping words between chunks (so we don't cut sentences in half)
TOP_K = 5              # Default number of search results to return
COLLECTION_NAME = "guidelines"  # Name of the database collection


# ============================================================
# EMBEDDING FUNCTION
# Converts text into a list of numbers (a "vector") so the
# database can compare how similar two pieces of text are.
# Similar text = vectors close together, different text = far apart.
# ============================================================

class EmbeddingFunction:
    def __init__(self, model_name=EMBEDDING_MODEL):
        """Load the embedding model. Raises error if sentence-transformers isn't installed."""
        if SentenceTransformer is None: raise ImportError("pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def __call__(self, input):
        """Main method: takes a list of text strings, returns a list of number vectors."""
        return self._model.encode(input, show_progress_bar=False).tolist()

    def name(self):
        """Returns the model name (required by ChromaDB)."""
        return self._model_name

    def embed_documents(self, input):
        """Convert document text to vectors (ChromaDB compatibility method)."""
        return self._model.encode(input, show_progress_bar=False).tolist()

    def embed_query(self, input):
        """Convert a search query to a vector (ChromaDB compatibility method)."""
        return self._model.encode(input, show_progress_bar=False).tolist()


# ============================================================
# TEXT CHUNKING
# Splits a long document into smaller overlapping pieces.
# Example: a 2000-word doc becomes ~5 chunks of 500 words each,
# with 100 words overlapping between consecutive chunks.
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of `chunk_size` words."""
    words = text.split()
    # If the text is short enough, return it as a single chunk
    if len(words) <= chunk_size: return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+chunk_size]))
        start += chunk_size - overlap  # Move forward, but keep some overlap
    return chunks


# ============================================================
# DOCUMENT READER
# Opens a file and extracts its text content.
# Supports PDFs and plain text files (txt, md).
# ============================================================

def read_document(path):
    """Read a PDF or text file and return its text content as a string."""
    if path.lower().endswith(".pdf"):
        # PDF file: use PyPDF2 to extract text from each page
        if PdfReader is None: raise ImportError("pip install PyPDF2")
        reader = PdfReader(path)
        return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    # Plain text file: just read it directly
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ============================================================
# LANGUAGE DETECTION FROM FILENAME
# Guesses what programming language a guideline document is about
# by looking at its filename. For example:
#   "python_pep8.pdf" → "python"
#   "c_barr.pdf" → "c"
#   "java_google.pdf" → "java"
# Checks longer names first so "javascript" matches before "java".
# ============================================================

def detect_language_from_path(path, title=""):
    """Detect the programming language from a guideline file's path and title."""
    lower = (path + " " + title).lower()

    # Check multi-word / longer names first to avoid false matches (e.g. "c" in "csharp")
    ordered_checks = [
        ("javascript", "javascript"), ("typescript", "typescript"),
        ("visual basic", "visualbasic"), ("visualbasic", "visualbasic"),
        ("csharp", "csharp"), ("c#", "csharp"),
        ("cpp", "cpp"), ("c++", "cpp"),
        ("python", "python"), ("java", "java"),
        ("go", "go"), ("rust", "rust"), ("kotlin", "kotlin"),
        ("swift", "swift"), ("ruby", "ruby"), ("php", "php"),
        ("sql", "sql"), ("delphi", "delphi"),
        ("r_", "r"),  # e.g. r_google.pdf
    ]
    for pattern, lang in ordered_checks:
        if pattern in lower:
            return lang

    # Check for C last — use word-boundary-like matching to avoid false positives
    # (the letter "c" appears in many words, so we check it's standalone like "c_barr" or "/c/")
    import re
    if re.search(r'(?:^|[_/\\. ])c(?:[_/\\. ]|$)', lower):
        return "c"

    return "general"


# ============================================================
# GUIDELINES RAG CLASS — THE LIBRARIAN
# This is the main class that manages the vector database.
# It can:
#   - Ingest (read and store) guideline documents
#   - Search for relevant passages by meaning
#   - List what's in the library
#   - Delete documents
# ============================================================

class GuidelinesRAG:

    def __init__(self, persist_dir=RAG_PERSIST_DIR):
        """Open (or create) the vector database. Like opening the library for the first time."""
        if chromadb is None: raise ImportError("pip install chromadb")
        os.makedirs(persist_dir, exist_ok=True)
        # Create a persistent ChromaDB client (data survives restarts)
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Load the embedding model
        self._embed_fn = EmbeddingFunction()
        # Get or create the "guidelines" collection (like a table in a database)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity for comparing vectors
        )

    @property
    def count(self):
        """How many text chunks are stored in the database."""
        return self._collection.count()

    def ingest(self, path, doc_id=None, language=None, title=None):
        """
        Read a document and add it to the library.
        1. Reads the file (PDF or text)
        2. Figures out a name, title, and language
        3. Splits into chunks
        4. Stores each chunk in ChromaDB with metadata labels
        """
        text = read_document(path)

        # Auto-generate doc_id from filename (e.g. "python_pep8" from "python_pep8.pdf")
        if not doc_id: doc_id = os.path.splitext(os.path.basename(path))[0]

        # Auto-generate title from first line of the document
        if not title:
            first_line = text.strip().split("\n")[0][:200]
            title = first_line if len(first_line) < 150 else doc_id

        # Auto-detect language from the file path
        if not language: language = detect_language_from_path(path, title)

        # Split the document into chunks
        chunks = chunk_text(text)

        # Create unique IDs for each chunk (e.g. "python_pep8_chunk_0", "python_pep8_chunk_1", ...)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        # Attach metadata to each chunk (language, title, position)
        metadatas = [{"doc_id":doc_id,"title":title,"language":language,"chunk_index":i,"total_chunks":len(chunks)} for i in range(len(chunks))]

        # Store in database (upsert = insert or update if already exists)
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return {"doc_id":doc_id,"title":title,"language":language,"chunks":len(chunks)}

    def ingest_directory(self, directory):
        """Read ALL guideline files (PDF, txt, md) in a folder and add them to the library."""
        results = []
        for pattern in ["*.pdf","*.txt","*.md"]:
            for path in glob.glob(os.path.join(directory, pattern)):
                try:
                    info = self.ingest(path)
                    print(f"  [OK] {info['title']} ({info['language']}, {info['chunks']} chunks)")
                    results.append(info)
                except Exception as e:
                    print(f"  [FAIL] {path} - {e}")
                    results.append({"doc_id":path,"error":str(e)})
        return results

    def search(self, query, top_k=TOP_K, language=None):
        """
        Find the most relevant guideline passages for a query.
        Example: search("naming conventions", language="python")
        Returns the top_k closest matches from the database.
        Optionally filters by language so Python rules don't show up for C code.
        """
        kwargs = {"query_texts":[query],"n_results":min(top_k,max(self.count,1)),"include":["documents","metadatas","distances"]}

        # If a language filter is provided, only search within that language
        if language: kwargs["where"] = {"language":language}
        try:
            results = self._collection.query(**kwargs)
        except Exception:
            # If filtered search fails (e.g. no docs for that language), try without filter
            kwargs.pop("where",None)
            results = self._collection.query(**kwargs)

        # Format results as a list of dicts with text, metadata, and distance (similarity score)
        return [{"text":results["documents"][0][i],"metadata":results["metadatas"][0][i],"distance":results["distances"][0][i]} for i in range(len(results["ids"][0]))]

    def list_docs(self):
        """List all documents in the library. Returns {doc_id: language} dict."""
        all_meta = self._collection.get(include=["metadatas"])
        docs = {}
        for m in all_meta["metadatas"]: docs[m.get("doc_id","unknown")] = m.get("language","?")
        return docs

    def get_languages(self):
        """Get a sorted list of all languages that have guidelines in the library."""
        all_meta = self._collection.get(include=["metadatas"])
        return sorted(set(m.get("language","general") for m in all_meta["metadatas"]))

    def delete(self, doc_id):
        """Remove a document and all its chunks from the library."""
        all_data = self._collection.get(include=["metadatas"])
        ids = [all_data["ids"][i] for i,m in enumerate(all_data["metadatas"]) if m.get("doc_id")==doc_id]
        if ids: self._collection.delete(ids=ids)
        return len(ids)


# ============================================================
# SINGLETON HELPER
# Creates the librarian only once, then reuses it.
# Like opening the library door — you only do it the first time.
# ============================================================

_rag = None
def _get_rag():
    """Get the shared GuidelinesRAG instance (created once, reused after)."""
    global _rag
    if _rag is None: _rag = GuidelinesRAG()
    return _rag


# ============================================================
# PUBLIC API — These are called by quality_agent.py
# ============================================================

def get_guidelines_for_language(language, top_k=8):
    """
    Fetch coding guidelines for a specific language from the library.
    Example: get_guidelines_for_language("python") → returns PEP8 rules as text.
    Used in Step 2 of the pipeline.
    """
    try:
        rag = _get_rag()
        if rag.count == 0: return ""  # Library is empty, nothing to return
        hits = rag.search(f"{language} coding style guidelines rules best practices", top_k, language)
        if not hits: return ""
        # Format the results as readable text
        parts = [f"Official {language} coding guidelines:"]
        for hit in hits:
            parts.append(f"  [{hit['metadata'].get('title','')}] {hit['text'][:400].replace(chr(10),' ')}")
        return "\n".join(parts)
    except Exception: return ""

def get_rules_for_evaluation(language, query, top_k=5):
    """
    Search for specific rules relevant to a particular evaluation question.
    Example: get_rules_for_evaluation("python", "naming conventions") → returns specific naming rules.
    Used in Step 5 when the AI is grading the code and needs rules to reference.
    """
    try:
        rag = _get_rag()
        if rag.count == 0: return ""  # Library is empty
        hits = rag.search(query, top_k, language)
        if not hits: return ""
        parts = [f"Applicable {language} rules:"]
        for hit in hits:
            parts.append(f"  - [{hit['metadata'].get('title','')}] {hit['text'][:300].replace(chr(10),' ')}")
        return "\n".join(parts)
    except Exception: return ""


# ============================================================
# COMMAND-LINE INTERFACE
# Run this file directly to manage the library from the terminal:
#   python rag_guidelines.py ingest guidelines/     → Load all PDFs
#   python rag_guidelines.py list                    → See what's loaded
#   python rag_guidelines.py search python "naming"  → Search for rules
#   python rag_guidelines.py status                  → Summary stats
#   python rag_guidelines.py delete python_pep8      → Remove a document
# ============================================================

def main():
    import sys
    usage = "\n  python rag_guidelines.py ingest <dir>\n  python rag_guidelines.py ingest-file <file> [lang]\n  python rag_guidelines.py search <lang> <query>\n  python rag_guidelines.py languages\n  python rag_guidelines.py list\n  python rag_guidelines.py status\n  python rag_guidelines.py delete <doc_id>\n"
    if len(sys.argv) < 2: print(usage); return

    cmd = sys.argv[1]
    rag = GuidelinesRAG()

    if cmd == "ingest":
        # Load all guideline files from a directory
        if len(sys.argv)<3: print("Error: specify dir."); return
        results = rag.ingest_directory(sys.argv[2])
        print(f"\nIngested {len(results)} docs. Chunks: {rag.count}")
    elif cmd == "ingest-file":
        # Load a single file
        if len(sys.argv)<3: print("Error: specify file."); return
        result = rag.ingest(sys.argv[2], language=sys.argv[3] if len(sys.argv)>3 else None)
        print(f"Ingested: {result}")
    elif cmd == "search":
        # Search for guidelines by language and query
        if len(sys.argv)<4: print("Usage: search <lang> <query>"); return
        for hit in rag.search(" ".join(sys.argv[3:]),TOP_K,sys.argv[2]):
            print(f"\n({hit['distance']:.3f}) {hit['metadata'].get('title','?')}\n  {hit['text'][:200]}...")
    elif cmd == "languages":
        # List all languages with guidelines
        print(f"Languages: {', '.join(rag.get_languages()) or 'none'}")
    elif cmd == "list":
        # List all documents
        for d,l in rag.list_docs().items(): print(f"  {d} ({l})")
        print(f"Total: {len(rag.list_docs())} docs, {rag.count} chunks")
    elif cmd == "status":
        # Show summary stats
        print(f"Documents: {len(rag.list_docs())}\nChunks: {rag.count}\nLanguages: {', '.join(rag.get_languages()) or 'none'}")
    elif cmd == "delete":
        # Delete a document by its ID
        if len(sys.argv)<3: print("Error: specify doc_id."); return
        print(f"Deleted {rag.delete(sys.argv[2])} chunks.")
    else:
        print(f"Unknown: {cmd}"); print(usage)

if __name__ == "__main__":
    main()
