import os, json, hashlib, re, glob
from typing import Optional
try:
    import chromadb
except ImportError:
    chromadb = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

RAG_PERSIST_DIR = os.environ.get("RAG_PERSIST_DIR", "./rag_data")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
COLLECTION_NAME = "guidelines"

class EmbeddingFunction:
    def __init__(self, model_name=EMBEDDING_MODEL):
        if SentenceTransformer is None: raise ImportError("pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
    def __call__(self, input):
        return self._model.encode(input, show_progress_bar=False).tolist()
    def name(self):
        return self._model_name
    def embed_documents(self, input):
        return self._model.encode(input, show_progress_bar=False).tolist()
    def embed_query(self, input):
        return self._model.encode(input, show_progress_bar=False).tolist()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size: return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+chunk_size]))
        start += chunk_size - overlap
    return chunks

def read_document(path):
    if path.lower().endswith(".pdf"):
        if PdfReader is None: raise ImportError("pip install PyPDF2")
        reader = PdfReader(path)
        return "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def detect_language_from_path(path, title=""):
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
    import re
    if re.search(r'(?:^|[_/\\. ])c(?:[_/\\. ]|$)', lower):
        return "c"
    return "general"

class GuidelinesRAG:
    def __init__(self, persist_dir=RAG_PERSIST_DIR):
        if chromadb is None: raise ImportError("pip install chromadb")
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embed_fn = EmbeddingFunction()
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=self._embed_fn, metadata={"hnsw:space": "cosine"})
    @property
    def count(self):
        return self._collection.count()
    def ingest(self, path, doc_id=None, language=None, title=None):
        text = read_document(path)
        if not doc_id: doc_id = os.path.splitext(os.path.basename(path))[0]
        if not title:
            first_line = text.strip().split("\n")[0][:200]
            title = first_line if len(first_line) < 150 else doc_id
        if not language: language = detect_language_from_path(path, title)
        chunks = chunk_text(text)
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id":doc_id,"title":title,"language":language,"chunk_index":i,"total_chunks":len(chunks)} for i in range(len(chunks))]
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return {"doc_id":doc_id,"title":title,"language":language,"chunks":len(chunks)}
    def ingest_directory(self, directory):
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
        kwargs = {"query_texts":[query],"n_results":min(top_k,max(self.count,1)),"include":["documents","metadatas","distances"]}
        if language: kwargs["where"] = {"language":language}
        try:
            results = self._collection.query(**kwargs)
        except Exception:
            kwargs.pop("where",None)
            results = self._collection.query(**kwargs)
        return [{"text":results["documents"][0][i],"metadata":results["metadatas"][0][i],"distance":results["distances"][0][i]} for i in range(len(results["ids"][0]))]
    def list_docs(self):
        all_meta = self._collection.get(include=["metadatas"])
        docs = {}
        for m in all_meta["metadatas"]: docs[m.get("doc_id","unknown")] = m.get("language","?")
        return docs
    def get_languages(self):
        all_meta = self._collection.get(include=["metadatas"])
        return sorted(set(m.get("language","general") for m in all_meta["metadatas"]))
    def delete(self, doc_id):
        all_data = self._collection.get(include=["metadatas"])
        ids = [all_data["ids"][i] for i,m in enumerate(all_data["metadatas"]) if m.get("doc_id")==doc_id]
        if ids: self._collection.delete(ids=ids)
        return len(ids)

_rag = None
def _get_rag():
    global _rag
    if _rag is None: _rag = GuidelinesRAG()
    return _rag

def get_guidelines_for_language(language, top_k=8):
    try:
        rag = _get_rag()
        if rag.count == 0: return ""
        hits = rag.search(f"{language} coding style guidelines rules best practices", top_k, language)
        if not hits: return ""
        parts = [f"Official {language} coding guidelines:"]
        for hit in hits:
            parts.append(f"  [{hit['metadata'].get('title','')}] {hit['text'][:400].replace(chr(10),' ')}")
        return "\n".join(parts)
    except Exception: return ""

def get_rules_for_evaluation(language, query, top_k=5):
    try:
        rag = _get_rag()
        if rag.count == 0: return ""
        hits = rag.search(query, top_k, language)
        if not hits: return ""
        parts = [f"Applicable {language} rules:"]
        for hit in hits:
            parts.append(f"  - [{hit['metadata'].get('title','')}] {hit['text'][:300].replace(chr(10),' ')}")
        return "\n".join(parts)
    except Exception: return ""

def main():
    import sys
    usage = "\n  python rag_guidelines.py ingest <dir>\n  python rag_guidelines.py ingest-file <file> [lang]\n  python rag_guidelines.py search <lang> <query>\n  python rag_guidelines.py languages\n  python rag_guidelines.py list\n  python rag_guidelines.py status\n  python rag_guidelines.py delete <doc_id>\n"
    if len(sys.argv) < 2: print(usage); return
    cmd = sys.argv[1]; rag = GuidelinesRAG()
    if cmd == "ingest":
        if len(sys.argv)<3: print("Error: specify dir."); return
        results = rag.ingest_directory(sys.argv[2])
        print(f"\nIngested {len(results)} docs. Chunks: {rag.count}")
    elif cmd == "ingest-file":
        if len(sys.argv)<3: print("Error: specify file."); return
        result = rag.ingest(sys.argv[2], language=sys.argv[3] if len(sys.argv)>3 else None)
        print(f"Ingested: {result}")
    elif cmd == "search":
        if len(sys.argv)<4: print("Usage: search <lang> <query>"); return
        for hit in rag.search(" ".join(sys.argv[3:]),TOP_K,sys.argv[2]):
            print(f"\n({hit['distance']:.3f}) {hit['metadata'].get('title','?')}\n  {hit['text'][:200]}...")
    elif cmd == "languages": print(f"Languages: {', '.join(rag.get_languages()) or 'none'}")
    elif cmd == "list":
        for d,l in rag.list_docs().items(): print(f"  {d} ({l})")
        print(f"Total: {len(rag.list_docs())} docs, {rag.count} chunks")
    elif cmd == "status":
        print(f"Documents: {len(rag.list_docs())}\nChunks: {rag.count}\nLanguages: {', '.join(rag.get_languages()) or 'none'}")
    elif cmd == "delete":
        if len(sys.argv)<3: print("Error: specify doc_id."); return
        print(f"Deleted {rag.delete(sys.argv[2])} chunks.")
    else: print(f"Unknown: {cmd}"); print(usage)

if __name__ == "__main__":
    main()
