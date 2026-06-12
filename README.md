 # RAG Evaluation Lab

Baseline RAG pipeline built as an Equal Experts take-home exercise.
Evolving into a personal lab for evaluation framework experimentation.

## Baseline (main / v1.0.0-ee-baseline) 
 
 ## :warning: Please read these instructions carefully and entirely first
* Clone this repository to your local machine.
* Use your IDE of choice to complete the assignment.
* When you have completed the assignment, you need to  push your code to this repository and [mark the assignment as completed by clicking here](https://app.snapcode.review/submission_links/ee3c18b9-dc94-4f03-941c-3044295d950b).
* Once you mark it as completed, your access to this repository will be revoked. Please make sure that you have completed the assignment and pushed all code from your local machine to this repository before you click the link.

# RAG Pipeline Exercise

This exercise involves implementing a Retrieval-Augmented Generation (RAG) pipeline to answer questions about Equal Experts case studies. The pipeline combines document chunking, embeddings, vector search, and LLM-based question answering.

## The Challenge

As a starting point, you need to implement:

1. The `chunk_text` method in `SimpleChunker` class (`src/chunking/chunking_strategies.py`)
   - Input: A text string
   - Output: A list of text chunks
   - Consider chunk size and overlap parameters

2. The `add_documents` method in `RAGPipeline` class (`src/rag/pipeline.py`)
   - Load the data from `case_studies.json`
   - Input: A list of documents
   - Process: Chunk documents → Generate embeddings → Store in vector store
   - Replace the current fixed text implementation

This should be enough to get the basic pipeline working.

As a next step, you should implement evaluation metrics to evaluate the performance of the pipeline, you can use external tools if you would like.

As an end goal, you should implement the `AdvancedChunker` class and the `chunk_text` method in the `AdvancedChunker` class.

And evaluate the performance of the pipeline again. Also consider any prompt improvements you might have.

### Tips

✅ Keep the code simple

✅ Include unit tests

### Questions
- How do you think the pipeline could be improved?
- What are the trade-offs between the chunking strategies?
- What do you do to take the pipeline into production?

## Prerequisites

1. Git
2. Python 3.12
3. Poetry (dependency management)
4. Ollama (local LLM service)

## Setting up the environment

### Installing dependencies

```bash
poetry install
```

### Activating the shell

```bash
poetry shell
```

### Running the tests

```bash
pytest
```

### Running the pipeline

#### Installing Ollama

Check the [Ollama website](https://ollama.com) for your operating system.
##### Downloading models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
#### Running the pipeline
Make sure you have the environment activated and ollama running with the models downloaded.

```bash
python -m src.main
```

### Developing Tooling

#### Types
```bash
poetry run mypy src/
``` 

#### ruff
```bash
poetry run ruff check --fix .
```
#### isort
```bash
poetry run isort src/ tests/
```
#### black
```bash
poetry run black src/ tests/
```

---

## Implementation Notes

### What was implemented

**SimpleChunker** - fixed-size character chunking with configurable overlap.
Splits text at fixed character boundaries. Overlap carries the tail of the
previous chunk into the next to preserve boundary context.

**AdvancedChunker** - paragraph-boundary chunking with overlap and sentence
fallback. Splits on natural paragraph boundaries (`\n\n`), merges small
paragraphs up to `chunk_size`, carries an overlap buffer between chunks, and
falls back to sentence-boundary splitting for paragraphs that exceed
`chunk_size`.

**RAGPipeline** - end-to-end pipeline with two independent methods:
- `retrieve(question)` - retrieval layer only, returns raw context chunks
- `query(question)` - full pipeline, retrieve then generate

Separating these methods enables independent evaluation of retrieval quality
vs generation quality - a retrieval failure and a generation failure have
different root causes and different fixes.

**Evaluation framework** - RAGAS-inspired LLM-as-judge evaluation across
five metrics. Judge model is configured independently from generation model
(`LLM_JUDGE_MODEL` in settings) to avoid same-model evaluation bias.

---

## Answers to Questions

### 1. How could the pipeline be improved?

#### Ingestion layer
- **Source Data identification and curation** - Start with what business problem
   you are solving and what is the relevant data source available. Work on the
   datapipeline to automate as much as practically possible to de-dup, version, 
   metadata tagging, transformation where needed and diversification. 
- **Chunking** - Chunking of data ingested can improve the performance, if done
   properly. It also optimises the context window. This is separately discussed
   within the next question. 
- **Selection of Embedding model** - For creating vector for individual chunk, 
   choose a model that is performant and can be multi-threaded. Parallelise 
   embedding tasks for the chunks accordingly. 
- **Choose Vector Store** - Which is reliable, efficient storage, and able to 
   do both semantic and BM25 based search. 

#### Retrieval layer
- **Hybrid search** - combine dense vector search (semantic similarity) with
  sparse BM25 keyword search. Vector search finds semantically related
  content; BM25 catches exact term matches. Neither alone is sufficient -
  hybrid retrieval outperforms either in isolation.
- **Re-ranking** - after initial retrieval, apply a cross-encoder re-ranker
  to re-score the top-k chunks against the query. Initial retrieval
  prioritises recall; re-ranking improves precision. Consider a separate 
  model just for re-ranking.
- **Metadata filtering** - attach source URL, document type, and date to
  each chunk at ingestion. Allow queries to filter by metadata before
  semantic search to reduce noise. You may use a model in the pipeline to 
  provide with specific context or even a set of metadata attached to the 
  chunk for efficient retrieval, if prompted appropriately. 
- **Query expansion** - before retrieval, use the LLM to rewrite the query
  into multiple variants. Retrieve against all variants and union results -
  improves recall for ambiguous queries.

#### Generation layer
- **Stronger LLM** - tinyllama produces fast but low-quality answers.
  llama3.1-70b or a hosted model (Azure OpenAI, Claude) would significantly
  improve answer quality and instruction following.
- **Prompt engineering** - current prompt is minimal. Adding few-shot
  examples, explicit citation instructions, and output format constraints
  improves consistency and reduces hallucination.
- **Citation grounding** - require the model to quote the specific chunk
  passage it used. Makes hallucination detection easier and answers
  verifiable.

#### Evaluation layer
- **Separate judge model from generation model** - using the same model to
  generate answers and judge their quality introduces optimistic bias.
  Earlier runs with tinyllama as both generation and judge scored 0.82
  overall. With llama3.2 as judge and tinyllama as generator, the same
  pipeline scored 0.60 - more honest and actionable. `LLM_JUDGE_MODEL` is
  a separate setting for this reason.
- **Structured JSON output from judge** - removes fragile regex parsing.
  Ollama's native `format: json` mode constrains the model to valid JSON.
- **Parallel metric scoring** - all five metrics are independent. Running
  them concurrently via `ThreadPoolExecutor` reduces eval time from ~35s to
  ~7s per test case.
- **Increase golden test dataset** - 7 test cases is sufficient to
  demonstrate the framework but too small for statistically reliable
  comparison. Production eval would use 50-100 cases minimum.

#### Agentic extension
RAG is a foundational primitive for agentic architectures. `retrieve()` and
`query()` become agent tools that a reasoning agent decides when to invoke.
The eval framework becomes an action guardrail - before an agent executes an
action, inline eval gates whether the information it's acting on is
sufficiently grounded. The HITL escalation pattern applies equally to agent
actions as to RAG responses.

---

### 2. What are the trade-offs between the chunking strategies?

#### SimpleChunker - fixed size with overlap

Splits text at fixed character boundaries with configurable overlap.

**Strengths:**
- Fast, deterministic, predictable chunk count
- Consistent chunk size simplifies top-k reasoning
- Overlap preserves boundary context

**Weaknesses:**
- Splits mid-sentence and mid-paragraph - breaks semantic units
- Embedding represents a confused average when chunks contain unrelated content
- Retrieval quality suffers when chunks lack coherent meaning

**Best for:** homogeneous, densely structured text where coverage matters
more than per-chunk coherence.

---

#### AdvancedChunker - paragraph-boundary with overlap and sentence fallback

Splits on natural paragraph boundaries, merges small paragraphs up to
`chunk_size`, carries overlap buffer between chunks, falls back to
sentence-boundary splitting for oversized paragraphs.

**Strengths:**
- Preserves semantic coherence - each chunk contains a complete thought
- Overlap bridges context gaps between adjacent paragraphs
- Sentence fallback handles edge cases without arbitrary mid-paragraph cuts
- Better per-chunk embedding quality - coherent chunks embed more precisely

**Weaknesses:**
- Variable chunk sizes complicate size-based reasoning
- More complex to implement and test
- Overlap increases total storage
- Requires retuning of `top_k` to maintain equivalent context volume

**Best for:** narrative or structured document content - case studies,
articles, reports - where paragraphs represent discrete ideas.

---

#### Empirical comparison

All evaluations use llama3.2 as judge model and tinyllama as generation
model with `top_k=3`.

**Chunk statistics:**

| | SimpleChunker | AdvancedChunker |
|---|---|---|
| Total chunks | 1,405 | 1,957 |
| Avg chunk size | 485 chars | 344 chars |
| Context per query (k=3) | ~1,455 chars | ~1,032 chars |

**Eval scores:**

| Metric | SimpleChunker | AdvancedChunker (k=3) |
|---|---|---|
| answer_relevance | 0.71 | 0.29 |
| faithfulness | 0.66 | 0.36 |
| answer_correctness | 0.69 | 0.57 |
| hallucination_rate | 0.67 | 0.84 |
| context_relevance | 0.29 | 0.47 |
| **Overall** | **0.60** | **0.50** |

#### Key finding

SimpleChunker outperforms AdvancedChunker at equivalent `top_k=3` due to
context volume difference. AdvancedChunker produces 40% more chunks at 29%
smaller average size - delivering 29% less total context to the LLM per
query. This explains the overall score differential.

Notable exception: `hallucination_rate` (0.84 vs 0.67) and
`context_relevance` (0.47 vs 0.29) are both higher for AdvancedChunker.
Paragraph-aligned chunks embed more precisely - when the right chunk is
retrieved, it contains cleaner signal. The problem is retrieval *coverage*,
not chunk *quality*.

**AdvancedChunker requires `top_k=5` to deliver equivalent context volume
to SimpleChunker at `top_k=3`.** Chunking strategy and retrieval parameters
are tightly coupled - changing one requires retuning the other.

#### Eval score variance

Scores show meaningful variance between runs with smaller judge models.
This is why a stronger, more deterministic judge model is essential in
production. Score *trends* over time are more reliable than point-in-time
values - alert on sustained degradation, not single-run variance.

---

### 3. What do you need to take the pipeline into production?

#### Ingestion pipeline
- **Selective re-embedding** - hash each source document at ingestion. On
  subsequent runs, only re-embed documents whose hash changed. Avoids full
  re-ingestion on minor source updates.
- **Persistent vector store** - replace Milvus Lite with a production vector
  store (Milvus Cloud, Pinecone, Azure AI Search) with replication, backups,
  and access controls.
- **Blue/green embedding deployment** - re-embed into a staging collection,
  run full eval, promote to production only if scores exceed baseline. The
  switch is atomic - no moment where production serves from a partially
  re-embedded store. Keep previous collection warm for 24 hours for instant
  rollback.

#### Evaluation as a deployment gate (offline)
- **Async eval after every ingestion** - queue an eval job after re-embedding
  completes. Ingestion does not block on eval. Eval runs in background via a
  message queue (SQS, RabbitMQ) consumed by eval workers.
- **Score thresholds as promotion gates** - if any metric drops below
  threshold vs previous baseline, new embeddings are not promoted to
  production. Same pattern as CI/CD test gates - tests fail, build doesn't
  ship.
- **Drift detection** - schedule regular regression evals (daily/weekly)
  against a fixed golden dataset. Alert on sustained metric degradation.
  Track trends, not point-in-time values.
- **Audit trail** - every eval run is timestamped and stored with full
  inputs, outputs, and scores. Producible for regulatory review.

#### Inline eval as a response quality gate (online)
Before serving any response to a user, run `faithfulness` and
`hallucination_rate` inline against the retrieved context:

User query -> retrieve context-> generate answer

Inline eval (faithfulness + hallucination)

┌─────────────────────────┐
│ Score ≥ threshold?      │
YES                       NO
│                         │
Return answer            Return friendly fallback
Queue HITL job async

Human reviews query, context,
answer, scores - approves or rewrites

The user never sees a hallucinated answer. The failure is invisible to the
user but fully traceable internally. Only `faithfulness` and
`hallucination_rate` run inline - they directly protect the user and are
fast. The remaining three metrics run offline.

#### Model drift response strategy
Drift response depends on severity - taking the system fully offline is the
last resort:

| Severity | Signal | Response |
|---|---|---|
| Gradual | Scores declining over weeks | Alert, investigate, increase HITL sampling |
| Threshold breach | Score drops below floor | Circuit breaker - route to backup model |
| Backup model failing | Both models degraded | Graceful degradation to human queue |
| Infrastructure failure | Model serving down | Maintenance page with ETA |

**Circuit breaker pattern** - if N consecutive responses fail inline eval,
flip the circuit and route to a stronger backup model (Azure OpenAI, Claude
API). More expensive but more reliable. Reset circuit after recovery timeout
and successful eval pass.

#### Observability and monitoring
- **Prompt and version tracking** - every prompt change versioned in Git and
  linked to eval results. No prompt ships without a passing eval run.
- **Response telemetry** - log query, retrieved chunks, answer, latency, and
  eval scores for every production request.
- **Metric dashboards** - surface score trends over time. Faithfulness
  dropping week-on-week is an early signal of retrieval degradation before
  users notice.
- **Alert thresholds** - sustained metric drops, latency spikes, HITL queue
  depth growing beyond capacity.

#### Security and compliance (regulated environment)
- **Data residency** - embeddings and source documents must not leave
  approved regions. Private endpoints, VNET isolation.
- **Access controls** - RBAC on vector store, eval results, and HITL review
  queue.
- **Responsible AI controls** - content filtering on inputs and outputs.
  Audit log of all queries and responses retained per regulatory requirements
  (APRA CPS234-aligned for financial services).
- **Explainability** - every answer cites the source chunk it was derived
  from, enabling human verification and regulatory producibility.

#### Scalability
- **Async LLM calls** - replace synchronous httpx client with aiohttp for
  non-blocking inference.
- **Model serving** - replace local Ollama with a production model serving
  layer (vLLM, Azure OpenAI, AWS Bedrock) that handles concurrent requests
  natively and provides SLA guarantees.
- **Semantic caching** - cache retrieved chunks against query embeddings in
  a secondary vector store. Similar future queries hit the cache and skip
  vector store search. Cache lookup itself requires similarity search -
  threshold tuning is required to balance hit rate against wrong-cache-hit
  risk. Cache invalidated on every re-embedding run. Most valuable when LLM
  inference is the primary cost driver (hosted models) rather than retrieval.
- **Horizontal scaling** - stateless query service scales independently of
  vector store. Eval workers scale independently of ingestion workers.

#### Background job architecture

A background job will get triggered for every RAG ingestion pipeline run. It can be either the first load or delta.
After loading to vector store, main thread publish a eval job to a middleware queue and does not block for eval to pickup/finish. 
The eval worker picks up the job and runs full regression against golden dataset. 
Compares score against baseline. If passed, embedding is promoted. If fails, raises alert. 

Replace file-based job queue (current implementation) with SQS/RabbitMQ.
Failed eval jobs dead-lettered and alerted - never silently dropped.

#### Different Models Selection for specific tasks
┌─────────────────────────────────────────────────────┐
│ INGESTION (offline) and QUERY EMBEDDING MODEL       │
│   Embedding model -> converts chunks to vectors     │
|   Embedding model -> converts query to vectors      |
│   Runs once per document change (Ingestion)         |
|   Runs once per query (Query Embedding / Retrieval) │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ RE_RANKING (online) MODEL                           │
│   query and retrieved contexts reranked with top K  │
│   Run once per query                                |
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ LLM (online) MODEL                                  │
│   Generate response with query and context          |
|   Once per query                                    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ LLM JUDGE MODEL EVALUATION (async and Sync)         │
│   Evaluate after every ingestion run                |
|   Also evaluate critical thresholds per query inline|   
|   Judge model -> scores answer quality              │
└─────────────────────────────────────────────────────┘

A complete architecture diagram for a production grade RAG system: 

<img width="5532" height="4302" alt="RAG Prod Grade Pipeline" src="https://github.com/user-attachments/assets/3fc8b0b6-4816-4c98-a880-f16970a40ee0" />


## Roadmap
- [ ] feature/ragas-integration — native RAGAS metrics alongside custom eval
- [ ] feature/hybrid-search — BM25 + dense vector + RRF via Milvus
- [ ] feature/reranking — cross-encoder between retrieve() and query()
- [ ] feature/query-expansion — multi-query before embedding