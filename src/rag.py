"""
RAG engine: vector store, retrieval, and prompt formatting.
"""
import os
from pathlib import Path
import pandas as pd


RAG_PROMPT = """You are a professional perfumer. Given the main accords of a fragrance, predict the most likely Top, Middle, and Base notes.

Use the reference perfumes below (which share similar accords) as hints. The notes you predict should be plausible for the given accords — they do NOT need to exactly match the references.

Main Accords: {accords}

Reference perfumes with similar accords:
{references}

Reply ONLY with a JSON object (no markdown, no explanation):
{{"top_notes": ["note1", "note2", "note3", "note4", "note5"], "middle_notes": ["note1", "note2", "note3", "note4", "note5"], "base_notes": ["note1", "note2", "note3", "note4", "note5"]}}

Include exactly 5 specific notes per layer. Use standardized note names in lowercase."""


class FragranceVectorStore:
    """
    Embed each perfume (accords + notes) with sentence-transformers,
    store in ChromaDB for semantic retrieval.
    """

    def __init__(self, persist_dir="./chroma_fragrances"):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="fragrances",
            metadata={"hnsw:space": "cosine"},
        )

    def build(self, df):
        """Index all perfumes into the vector store. Skips if already built."""
        if self.collection.count() > 0:
            print(f"  Vector store already has {self.collection.count()} entries, skipping build.")
            return

        print(f"  Building vector store for {len(df):,} perfumes...")
        documents = []
        metadatas = []
        ids = []

        for i, (_, row) in enumerate(df.iterrows()):
            accords_text = ", ".join(row['accords'])
            notes_text = (
                f"Top: {', '.join(sorted(row['Top_clean']))} | "
                f"Middle: {', '.join(sorted(row['Middle_clean']))} | "
                f"Base: {', '.join(sorted(row['Base_clean']))}"
            )
            doc = f"Accords: {accords_text}\nNotes: {notes_text}"

            documents.append(doc)
            metadatas.append({
                "name": str(row.get("Perfume", "")),
                "brand": str(row.get("Brand", "")),
                "accords": accords_text,
                "notes": notes_text,
                "rating_value": float(row.get("Rating Value", 0) or 0),
                "rating_count": int(row.get("Rating Count", 0) or 0),
                "top_notes": ", ".join(sorted(row['Top_clean'])),
                "middle_notes": ", ".join(sorted(row['Middle_clean'])),
                "base_notes": ", ".join(sorted(row['Base_clean'])),
                "idx": i,
            })
            ids.append(f"perfume_{i}")

            if (i + 1) % 5000 == 0:
                print(f"    ... {i+1}/{len(df)} documents prepared")

        batch_size = 1000
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]
            batch_meta = metadatas[start:end]

            embeddings = self.embedder.encode(batch_docs, show_progress_bar=False).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            print(f"    ... indexed {end}/{len(documents)}")

        print(f"  Vector store built: {self.collection.count()} entries.")

    def retrieve(self, accords, top_k=5):
        """Given a list of accords, retrieve top-K similar perfumes."""
        query = f"Accords: {', '.join(accords)}"
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        references = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            similarity = max(0.0, 1.0 - dist)
            references.append({
                "name": meta.get("name", "unknown"),
                "brand": meta.get("brand", "unknown"),
                "accords": meta.get("accords", ""),
                "notes": meta.get("notes", ""),
                "rating_value": meta.get("rating_value", 0),
                "rating_count": meta.get("rating_count", 0),
                "similarity": round(similarity, 4),
            })

        return references

    def retrieve_for_recommendation(self, accords, top_k=20):
        """Retrieve top-K similar perfumes with full per-layer metadata for recommendations."""
        query = f"Accords: {', '.join(accords)}"
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        references = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            similarity = max(0.0, 1.0 - dist)
            references.append({
                "name": meta.get("name", "unknown"),
                "brand": meta.get("brand", "unknown"),
                "accords": meta.get("accords", ""),
                "top_notes": meta.get("top_notes", ""),
                "middle_notes": meta.get("middle_notes", ""),
                "base_notes": meta.get("base_notes", ""),
                "rating_value": meta.get("rating_value", 0),
                "rating_count": meta.get("rating_count", 0),
                "similarity": round(similarity, 4),
            })
        return references


def format_references(refs):
    """Format retrieved references for insertion into the RAG prompt."""
    parts = []
    for i, ref in enumerate(refs, 1):
        parts.append(
            f"[{i}] {ref['name']} (similarity: {ref['similarity']:.2f})\n"
            f"    {ref['notes']}"
        )
    return "\n".join(parts)
