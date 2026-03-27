# RAG Pipeline (LangChain + FAISS)

This repository implements a Retrieval-Augmented Generation (RAG) pipeline that:
- ingests a knowledge base (PDF/TXT/MD/HTML),
- chunks and enriches metadata,
- builds a FAISS vector index,
- retrieves top-k relevant chunks (vector or hybrid BM25+vector),
- generates grounded answers with citations and confidence statements.

## Project Structure

- `ingest.py` — parsing, cleaning, chunking, metadata, JSONL export
- `index.py` — embedding + FAISS index creation
- `retrieve_generate.py` — retrieval + prompt + LLM generation
- `requirements.txt`
- `eval/eval_set.jsonl` — sample evaluation set format
- `eval/evaluation_report.md` — report template (1–2 pages)

## Setup

1) Create and activate Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

3) Set API key for your provider

OpenAI:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

NVIDIA NIM (OpenAI-compatible API):

```powershell
$env:NVIDIA_API_KEY="your_nvidia_key_here"
$env:NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"
```

Optional embedding settings:
- `EMBEDDING_PROVIDER=openai|nvidia|huggingface`
- `EMBEDDING_MODEL=text-embedding-3-small` (OpenAI default)
- `EMBEDDING_MODEL=nvidia/nv-embedqa-e5-v5` (NVIDIA example)
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` (HF example)

Optional LLM setting:
- `LLM_PROVIDER=openai|nvidia`
- `LLM_MODEL=gpt-4o-mini` (OpenAI default)
- `LLM_MODEL=meta/llama-3.1-70b-instruct` (NVIDIA example)

## 1) Ingest and Chunk

Put your documents in `kb/`, then run:

```powershell
python ingest.py --kb-path kb --out data/chunks.jsonl --chunk-size 1000 --chunk-overlap 150
```

Output: `data/chunks.jsonl` (each line includes `chunk_id`, `text`, and metadata such as `source`, `title`, `file_name`, `url`, `date`).

## 2) Embed and Index

```powershell
python index.py --chunks data/chunks.jsonl --index-dir vectorstore/faiss_index --embedding-provider openai --embedding-model text-embedding-3-small
```

NVIDIA example:

```powershell
python index.py --chunks data/chunks.jsonl --index-dir vectorstore/faiss_index --embedding-provider nvidia --embedding-model nvidia/nv-embedqa-e5-v5
```

Output: FAISS index files under `vectorstore/faiss_index`.

## 3) Retrieve + Generate

Vector retrieval:

```powershell
python retrieve_generate.py --query "What are the incident response escalation steps?" --k 5 --strategy vector --llm-provider openai --llm-model gpt-4o-mini
```

Hybrid retrieval (BM25 + vector with RRF):

```powershell
python retrieve_generate.py --query "What are the incident response escalation steps?" --k 5 --strategy hybrid --llm-provider openai --llm-model gpt-4o-mini
```

NVIDIA generation example:

```powershell
python retrieve_generate.py --query "What are the incident response escalation steps?" --k 5 --strategy hybrid --embedding-provider nvidia --embedding-model nvidia/nv-embedqa-e5-v5 --llm-provider nvidia --llm-model meta/llama-3.1-70b-instruct
```

The response JSON contains:
- `answer` (grounded response with inline citations like `[chunk-00012]`),
- `citations` (retrieved chunk IDs in context),
- `retrieved` (source metadata per chunk).

## Evaluation Guidance

1. Build an eval set in `eval/eval_set.jsonl` with:
   - query
   - expected_answer
   - expected_sources

2. For each query, run `retrieve_generate.py` and record:
   - Retrieval: Recall@k, MRR (or Hit@k)
   - Generation: Exact match/F1 (or semantic similarity), citation correctness

3. Fill out `eval/evaluation_report.md`:
   - dataset description
   - metrics table
   - qualitative failures
   - hallucination analysis
   - reflection (5–7 sentences)

## Demo Deliverable

Capture one short GIF/screenshot showing a terminal run where the model answers with cited chunk IDs. Add the artifact path/link to the report.

## Notes

- `ingest.py` uses character-based chunking; if token-accurate chunking is needed, swap to token-based splitters.
- If your corpus includes noisy web pages, improve boilerplate removal in `ingest.py` before indexing.
