# RAG Project Knowledge Base (Sample)

This project implements a Retrieval-Augmented Generation pipeline.

## Goals

1. Ingest knowledge base documents.
2. Split them into chunks with overlap.
3. Compute embeddings and build a FAISS index.
4. Retrieve the most relevant chunks for each question.
5. Generate grounded answers that include citations and a confidence statement.

## Retrieval Strategies

The project supports:
- Vector retrieval using FAISS similarity search.
- Hybrid retrieval that combines BM25 and vector search, then fuses rankings.

## Reliability Guidance

The generator is prompted to use only retrieved context. If information is missing, it should say it does not have enough context.
