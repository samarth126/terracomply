# build_index.py
import argparse
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def collect_pdfs(root: Path, numeric_only: bool = False) -> list[Path]:
    """
    Find PDFs under `root` (recursive).
    If numeric_only=True, only use files named like 1.pdf, 2.pdf, ...
    """
    pdfs = []
    for p in root.rglob("*.pdf"):
        if numeric_only:
            name = p.name.lower()
            if name[:-4].isdigit() and name.endswith(".pdf"):
                pdfs.append(p)
        else:
            pdfs.append(p)
    # stable, natural-ish sort
    return sorted(pdfs, key=lambda x: (x.parent.as_posix(), x.name))


def load_all_docs(pdf_paths: list[Path]):
    """Load every page from each PDF as LangChain Documents."""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        # ensure helpful metadata for later debugging/answers
        for d in docs:
            d.metadata.setdefault("file", path.name)
            d.metadata.setdefault("source", str(path))
        all_docs.extend(docs)
    return all_docs


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs in tenant_data/")
    parser.add_argument("--data-dir", default="data", help="Folder containing PDFs (default: tenant_data)")
    parser.add_argument("--index-dir", default="faiss_index", help="Where to save the FAISS index")
    parser.add_argument("--chunk-size", type=int, default=200, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Chunk overlap for splitting")
    parser.add_argument("--numeric-only", action="store_true", help="Only index PDFs named like 1.pdf, 2.pdf, ...")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HF embedding model name")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    pdf_paths = collect_pdfs(data_dir, numeric_only=args.numeric_only)
    if not pdf_paths:
        flag_msg = " (numeric-only enabled)" if args.numeric_only else ""
        raise SystemExit(f"No PDFs found under {data_dir}{flag_msg}.")

    print(f"📄 Found {len(pdf_paths)} PDF file(s) under {data_dir}")
    docs = load_all_docs(pdf_paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    print(f"✂️  Split into {len(split_docs)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=args.model)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(args.index_dir)
    print(f"✅ FAISS index saved to: ./{args.index_dir}")


if __name__ == "__main__":
    main()
