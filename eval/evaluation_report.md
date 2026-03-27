# Evaluation Report (RAG Pipeline)

## 1) Dataset Description

- **Knowledge base size:** _N documents, M chunks_
- **Document types:** _PDF / Markdown / TXT / HTML_
- **Domain:** _e.g., policy, product docs, support knowledge_
- **Evaluation queries:** _Q items in eval set_
- **Ground-truth source annotation:** _how expected sources were assigned_

## 2) Retrieval and Generation Metrics

| Metric | Value | Notes |
|---|---:|---|
| Recall@5 |  | Fraction of queries where at least one expected source is retrieved |
| Hit@5 |  | Binary success rate per query |
| MRR |  | Rank-sensitive retrieval quality |
| Citation Precision |  | Fraction of cited chunks that are relevant |
| Answer Correctness (manual/auto) |  | Percent judged correct |

## 3) Qualitative Failure Cases

### Case A
- **Query:**
- **Observed issue:**
- **Likely cause:**
- **Mitigation:**

### Case B
- **Query:**
- **Observed issue:**
- **Likely cause:**
- **Mitigation:**

## 4) Hallucination Analysis

- **Hallucination patterns observed:**
- **How retrieval reduced hallucinations:**
- **Remaining failure conditions (e.g., missing docs, weak chunking):**
- **Confidence statement reliability:**

## 5) Reflection (5–7 sentences)

_Example starter (replace with your own):_

The pipeline worked well for fact-based questions where the relevant chunk appeared in the top-5 retrieval results. Hybrid retrieval improved recall for keyword-heavy queries compared with vector-only search. The largest challenge was balancing chunk size and overlap to preserve context while avoiding noisy retrieval. Citation-aware prompting improved transparency, but citation correctness still depended on retrieval quality. Hallucinations were reduced when prompts strictly constrained answers to the provided context. A major next step is adding a reranker to improve top-k precision before generation. Another improvement is automating evaluation with per-query scoring and error categorization.
