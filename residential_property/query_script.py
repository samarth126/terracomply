# query_script.py
import requests
import argparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_db(index_dir: str, model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    return db, embeddings


def query_ollama(db: FAISS, question: str, k: int = 3, model: str = "qwen2.5:3b") -> str:
    # retrieve context
    docs = db.similarity_search(question, k=k)
    context_blocks = []
    for d in docs:
        src = d.metadata.get("file") or d.metadata.get("source") or "unknown"
        context_blocks.append(f"[{src}]\n{d.page_content}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "No response from model.")


def main():
    parser = argparse.ArgumentParser(description="Query FAISS + Ollama for property_data/")
    parser.add_argument("question", nargs="*", help="Your question (leave empty for REPL)")
    parser.add_argument("--index-dir", default="faiss_index", help="Path to FAISS index (default: faiss_index)")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HF embedding model name")
    parser.add_argument("--ollama-model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("-k", type=int, default=3, help="Top-k retrieval")
    args = parser.parse_args()

    db, _ = load_db(args.index_dir, args.embed_model)

    if args.question:
        q = " ".join(args.question).strip()
        print(query_ollama(db, q, k=args.k, model=args.ollama_model))
    else:
        # Simple REPL
        try:
            while True:
                q = input("\n❓ Question (Ctrl+C to exit): ").strip()
                if not q:
                    continue
                print("\n🧠 Answer:\n" + query_ollama(db, q, k=args.k, model=args.ollama_model))
        except KeyboardInterrupt:
            print("\nbye!")


if __name__ == "__main__":
    main()
