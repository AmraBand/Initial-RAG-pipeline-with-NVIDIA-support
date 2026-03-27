import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from nvidia_embeddings import NvidiaNIMEmbeddings


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def load_chunks(path: Path) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(Document(page_content=item["text"], metadata=item.get("metadata", {})))
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


def build_llm(provider: str, model_name: str, temperature: float):
    from langchain_openai import ChatOpenAI

    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM generation.")
        base_url = os.getenv("OPENAI_BASE_URL")
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)

    if provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required for NVIDIA LLM generation.")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key, base_url=base_url)

    raise ValueError("Unsupported LLM provider. Use 'openai' or 'nvidia'.")


def vector_retrieve(vstore: FAISS, query: str, k: int) -> List[Document]:
    return vstore.similarity_search(query, k=k)


def bm25_retrieve(docs: Sequence[Document], query: str, k: int) -> List[Document]:
    from rank_bm25 import BM25Okapi

    tokenized_docs = [simple_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(simple_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in ranked_idx]


def reciprocal_rank_fusion(lists: Sequence[Sequence[Document]], k: int, rrf_k: int = 60) -> List[Document]:
    scores: Dict[str, float] = {}
    selected: Dict[str, Document] = {}

    for ranking in lists:
        for rank, doc in enumerate(ranking, start=1):
            chunk_id = doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            selected[chunk_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [selected[chunk_id] for chunk_id, _ in ranked[:k]]


def build_context(docs: Sequence[Document], max_chars: int) -> Tuple[str, List[str]]:
    context_blocks = []
    citations: List[str] = []
    total = 0

    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip()
        block = f"[{chunk_id}] source={source}\n{snippet}"

        if total + len(block) > max_chars:
            break

        context_blocks.append(block)
        citations.append(chunk_id)
        total += len(block)

    return "\n\n".join(context_blocks), citations


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a grounded QA assistant. Use ONLY the provided context to answer.\n"
        "If the answer is not in the context, say you do not have enough information.\n"
        "Cite evidence inline with [chunk-id] markers from the context.\n"
        "End with 'Confidence: <low|medium|high> - <one short reason>'.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


@dataclass
class Pipeline:
    vstore: FAISS
    all_docs: List[Document]
    llm: object

    def retrieve(self, query: str, k: int, strategy: str) -> List[Document]:
        if strategy == "vector":
            return vector_retrieve(self.vstore, query, k)
        if strategy == "hybrid":
            dense = vector_retrieve(self.vstore, query, k)
            sparse = bm25_retrieve(self.all_docs, query, k)
            return reciprocal_rank_fusion([dense, sparse], k=k)
        raise ValueError("strategy must be 'vector' or 'hybrid'")

    def answer(self, query: str, k: int, strategy: str, max_context_chars: int) -> Dict[str, object]:
        docs = self.retrieve(query, k=k, strategy=strategy)
        context, citation_ids = build_context(docs, max_chars=max_context_chars)

        if not context:
            return {
                "answer": "I do not have enough information from the retrieved context.",
                "citations": [],
                "retrieved": [],
            }

        prompt = build_prompt(query, context)
        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        retrieved = [
            {
                "chunk_id": d.metadata.get("chunk_id", "unknown"),
                "source": d.metadata.get("source", "unknown"),
                "title": d.metadata.get("title", ""),
            }
            for d in docs
        ]

        return {
            "answer": answer_text,
            "citations": citation_ids,
            "retrieved": retrieved,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks and generate a citation-aware answer.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--chunks", default="data/chunks.jsonl", help="Chunk JSONL file")
    parser.add_argument("--index-dir", default="vectorstore/faiss_index", help="FAISS index directory")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--strategy", choices=["vector", "hybrid"], default="vector", help="Retrieval strategy")
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
    parser.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER", "openai"), choices=["openai", "nvidia"], help="LLM backend")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"), help="Chat model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max-context-chars", type=int, default=12000, help="Max context length sent to LLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embeddings = build_embeddings(args.embedding_provider, args.embedding_model)
    index_path = Path(args.index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"Index directory not found: {index_path}. Run index.py first.")

    vstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    all_docs = load_chunks(Path(args.chunks))
    llm = build_llm(args.llm_provider, args.llm_model, args.temperature)

    pipeline = Pipeline(vstore=vstore, all_docs=all_docs, llm=llm)
    result = pipeline.answer(
        query=args.query,
        k=args.k,
        strategy=args.strategy,
        max_context_chars=args.max_context_chars,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
