import argparse
import json
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from nvidia_embeddings import NvidiaNIMEmbeddings


def load_chunks(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(
                Document(
                    page_content=item["text"],
                    metadata=item.get("metadata", {}),
                )
            )
    if not docs:
        raise ValueError("Chunk file is empty.")
    return docs


def build_embeddings(provider: str, model_name: str):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url)

    if provider == "nvidia":
        return NvidiaNIMEmbeddings(
            model=model_name,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )

    if provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError("Unsupported embedding provider. Use 'openai', 'nvidia', or 'huggingface'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunked KB data.")
    parser.add_argument("--chunks", default="data/chunks.jsonl", help="Path to chunked JSONL")
    parser.add_argument("--index-dir", default="vectorstore/faiss_index", help="Output directory for FAISS index")
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("EMBEDDING_PROVIDER", "openai"),
        choices=["openai", "nvidia", "huggingface"],
        help="Embedding backend",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_chunks(Path(args.chunks))
    embeddings = build_embeddings(args.embedding_provider, args.embedding_model)

    index = FAISS.from_documents(docs, embeddings)
    out_dir = Path(args.index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(out_dir))

    print(f"Indexed chunks: {len(docs)}")
    print(f"Saved FAISS index to: {out_dir}")


if __name__ == "__main__":
    main()
