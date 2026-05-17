import os                          # for file path operations
import faiss                       # Facebook's vector search library
import numpy as np                 # for array operations (FAISS works with numpy arrays)
import pickle                      # for saving/loading Python objects to disk
from typing import List, Dict, Any # type hints to make code readable and safe
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits long docs into chunks
from app.services.embeddings import get_embedding_model  # our shared embedding model from Step 4
from app.core.config import get_settings                 # our centralized config from Step 2


class VectorStore:
    def __init__(self):
        settings = get_settings()                      # load all settings from .env
        self.persist_dir = settings.FAISS_STORE_PATH   # folder where index files are saved (faiss_store/)
        self.embedder = get_embedding_model()          # load the HuggingFace embedding model (once, cached)
        self.index = None                              # FAISS index object — None until built or loaded
        self.metadata: List[Dict] = []                 # list of dicts storing text, source, page per chunk
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # each chunk is max 1000 characters
            chunk_overlap=200,  # 200 characters overlap between chunks to avoid cutting sentences
        )
        os.makedirs(self.persist_dir, exist_ok=True)   # create faiss_store/ folder if it doesn't exist

    def _embed(self, texts: List[str]) -> np.ndarray:
        # convert a list of text strings into a 2D numpy array of float32 vectors
        vectors = self.embedder.embed_documents(texts)              # returns list of lists (one vector per text)
        return np.array(vectors, dtype="float32")                   # FAISS requires float32 numpy array

    def build_from_documents(self, documents: List[Any]):
        chunks = self.splitter.split_documents(documents)           # break all documents into smaller chunks
        print(f"[INFO] Split into {len(chunks)} chunks")

        texts = [chunk.page_content for chunk in chunks]            # extract raw text from each chunk
        metas = [
            {
                "text": chunk.page_content,                         # the actual text content of the chunk
                "source": chunk.metadata.get("source", "unknown"),  # original filename (e.g. DBMS_Notes.pdf)
                "page": chunk.metadata.get("page", 0),              # page number in the original document
            }
            for chunk in chunks
        ]

        embeddings = self._embed(texts)                             # convert all chunk texts to vectors
        dim = embeddings.shape[1]                                   # get vector dimension (384 for MiniLM)
        self.index = faiss.IndexFlatIP(dim)                         # IP = Inner Product (cosine similarity on normalized vectors)
        self.index.add(embeddings)                                  # add all vectors to the FAISS index
        self.metadata = metas                                       # store the metadata list
        self.save()                                                 # persist index + metadata to disk
        print(f"[INFO] Vector store built with {len(chunks)} chunks")

    def add_documents(self, documents: List[Any]):
        # adds NEW documents to an already existing vector store (used when user uploads files)
        if self.index is None:
            self.load()                                             # load existing index first if not in memory

        chunks = self.splitter.split_documents(documents)          # split new documents into chunks
        texts = [chunk.page_content for chunk in chunks]
        metas = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
            }
            for chunk in chunks
        ]

        embeddings = self._embed(texts)
        self.index.add(embeddings)                                  # append new vectors to existing index
        self.metadata.extend(metas)                                 # append new metadata to existing list
        self.save()                                                 # save updated index to disk
        print(f"[INFO] Added {len(chunks)} new chunks to vector store")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            self.load()                                             # auto-load index if not already loaded

        query_vec = np.array(
            [self.embedder.embed_query(query_text)], dtype="float32"  # embed the user's question into a vector
        )
        scores, indices = self.index.search(query_vec, top_k)      # find top_k most similar vectors in the index

        results = []
        for score, idx in zip(scores[0], indices[0]):              # loop through each result
            if idx < len(self.metadata):                           # safety check — idx must be valid
                results.append({
                    "score": float(score),                         # similarity score (closer to 1.0 = more relevant)
                    "text": self.metadata[idx]["text"],            # the actual chunk text
                    "source": self.metadata[idx]["source"],        # which file it came from
                    "page": self.metadata[idx]["page"],            # which page it came from
                })
        return results                                             # list of dicts with score, text, source, page

    def save(self):
        # write FAISS index and metadata to disk so they survive app restarts
        faiss.write_index(self.index, os.path.join(self.persist_dir, "faiss.index"))
        with open(os.path.join(self.persist_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)                          # pickle serializes the Python list to bytes

    def load(self):
        index_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError("Vector store not found. Run build first.")  # clear error if store missing
        self.index = faiss.read_index(index_path)                  # load FAISS index from disk into memory
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)                         # deserialize metadata list from disk
        print(f"[INFO] Vector store loaded ({self.index.ntotal} vectors)")
