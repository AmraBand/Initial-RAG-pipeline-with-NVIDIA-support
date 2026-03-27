import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def extract_text_from_html(file_path: Path) -> str:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    return raw


def load_file_as_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def iter_documents(kb_path: Path) -> Iterable[Document]:
    for file_path in kb_path.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = clean_text(load_file_as_text(file_path))
        if not text:
            continue

        metadata: Dict[str, str] = {
            "source": str(file_path),
            "title": file_path.stem,
            "file_name": file_path.name,
            "url": "",
            "date": "",
        }
        yield Document(page_content=text, metadata=metadata)


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def save_chunks_jsonl(chunks: List[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk-{i:05d}"
            payload = {
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "metadata": {
                    **chunk.metadata,
                    "chunk_id": chunk_id,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and chunk knowledge-base documents.")
    parser.add_argument("--kb-path", default="kb", help="Directory containing source KB files")
    parser.add_argument("--out", default="data/chunks.jsonl", help="Path to write chunked output")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb_path = Path(args.kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(f"KB path does not exist: {kb_path}")

    docs = list(iter_documents(kb_path))
    if not docs:
        raise ValueError("No supported documents found. Add files to the KB directory.")

    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    save_chunks_jsonl(chunks, Path(args.out))

    print(f"Loaded docs: {len(docs)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
