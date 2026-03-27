# v1.0.0 — Initial RAG Pipeline with NVIDIA Support

## Short Release Blurb

- End-to-end RAG pipeline in Python (ingest, index, retrieve, generate).
- Citation-aware grounded answers using retrieved context chunks.
- NVIDIA NIM support for both embeddings and LLM inference.
- FAISS vector index plus optional hybrid retrieval (BM25 + vector fusion).
- Includes runnable scripts, setup docs, and evaluation report templates.

## Highlights

- End-to-end Retrieval-Augmented Generation pipeline implemented in Python.
- Knowledge-base ingestion and chunking workflow for PDF/TXT/MD/HTML content.
- FAISS vector index creation for efficient semantic retrieval.
- Retrieval + generation flow with citation-aware, grounded answers.
- NVIDIA NIM integration for both embeddings and LLM generation.
- Optional hybrid retrieval mode (BM25 + vector with reciprocal rank fusion).

## Included Deliverables

- `ingest.py` — parsing, cleaning, chunking, metadata export
- `index.py` — embedding + FAISS indexing
- `retrieve_generate.py` — retriever + prompt + LLM answer generation
- `nvidia_embeddings.py` — NVIDIA asymmetric embedding adapter (`passage` / `query`)
- `requirements.txt` and `README.md`
- Evaluation artifacts:
  - `eval/eval_set.jsonl`
  - `eval/evaluation_report.md`
  - `eval/demo_instructions.md`

## Verified Run

The pipeline was executed successfully end-to-end:

1. Ingestion produced `data/chunks.jsonl`
2. Indexing produced `vectorstore/faiss_index`
3. Generation returned grounded output with chunk citations (e.g., `chunk-00000`)

## Setup Summary

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Configure NVIDIA credentials:
   - `NVIDIA_API_KEY`
   - optional `NVIDIA_BASE_URL` (default: `https://integrate.api.nvidia.com/v1`)
3. Run:
   - `python ingest.py ...`
   - `python index.py --embedding-provider nvidia --embedding-model nvidia/nv-embedqa-e5-v5 ...`
   - `python retrieve_generate.py --embedding-provider nvidia --llm-provider nvidia --llm-model meta/llama-3.1-70b-instruct ...`

## Notes

- `.gitignore` is configured to exclude local secrets and generated artifacts (`.venv`, vectorstore output, local chunk artifacts).
- This release is the baseline implementation; next improvements can include reranking, expanded eval automation, and UI/demo packaging.
