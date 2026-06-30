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

---

## Milvus Store — Implementation Notes and Lessons Learned

This section documents every decision and bug encountered while implementing and upgrading the Milvus vector store layer. It is written as a tutorial so the reasoning behind each change is reproducible.

---

### 1. Adding Sparse BM25 for Hybrid Retrieval

#### Why

Dense vector search (embeddings) finds semantically similar content but misses exact keyword matches. BM25 keyword search finds exact terms but misses paraphrasing. Hybrid retrieval combines both so neither gap exists in isolation.

#### Schema change

A `SPARSE_FLOAT_VECTOR` field was added alongside the existing `FLOAT_VECTOR` embedding field:

```python
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)  # dense
schema.add_field(field_name="sparse",    datatype=DataType.SPARSE_FLOAT_VECTOR)     # sparse BM25
```

The sparse field requires its own index type:

```python
index_params.add_index(field_name="embedding", index_type="AUTOINDEX",            metric_type="IP")
index_params.add_index(field_name="sparse",    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
```

Both fields use Inner Product (`IP`) as the distance metric, which is correct for normalised dense vectors and for BM25 score accumulation.

#### Schema migration guard

If the collection already existed without the `sparse` field (from a previous run), it is dropped and recreated:

```python
fields = client.describe_collection(self.collection_name)["fields"]
has_sparse = any(f["name"] == "sparse" for f in fields)
if stats["row_count"] > 0 and has_sparse:
    return          # reuse: data exists and schema is current
client.drop_collection(...)   # recreate: empty or schema is stale
```

This means re-ingestion is required whenever the schema changes. That is expected — sparse vectors for existing records would be missing otherwise.

#### Hybrid search at query time

Two `AnnSearchRequest` objects (one dense, one sparse) are combined with Reciprocal Rank Fusion:

```python
dense_req  = AnnSearchRequest(data=[query_embedding], anns_field="embedding", ...)
sparse_req = AnnSearchRequest(data=[query_sparse],    anns_field="sparse",    ...)

results = client.hybrid_search(
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),   # k=60 is the standard RRF damping constant
    ...
)
```

RRF ranks candidates by combining their positions in the two result lists rather than their raw scores, which avoids the problem of dense and sparse scores having incompatible scales.

---

### 2. BM25 Encoder — `pymilvus.model` vs custom implementation

#### What `BM25EmbeddingFunction` returns

`pymilvus.model.sparse.BM25EmbeddingFunction.encode_documents()` returns a `scipy.sparse.csr_array` (not a list, not a `csr_matrix`). The `csr_array` class (introduced in scipy 1.8) dropped the `getrow()` method that existed on the older `csr_matrix`.

Extracting per-row dicts from a `csr_array` must use the underlying CSR storage arrays directly:

```python
csr = mat.tocsr()
for i in range(csr.shape[0]):
    start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
    row_dict = {int(c): float(v)
                for c, v in zip(csr.indices[start:end], csr.data[start:end])}
```

This works on both `csr_array` and `csr_matrix` because both share the same underlying storage format.

#### Fitting BM25 before encoding

BM25 IDF weights are corpus-global. `fit()` must be called on the **full chunk corpus** before `encode_documents()` or `encode_query()`:

```python
self.bm25.fit(chunks)                              # build IDF over all chunks
sparse_vectors = self.bm25.encode_documents(chunks) # encode with global IDF
```

Fitting on a subset produces biased IDF weights. If documents are ingested in multiple batches, re-fitting on the combined corpus is required.

#### `encode_query` vs `encode_queries`

The `SparseEncoder` ABC exposes both:
- `encode_queries(texts: list[str]) -> list[dict]` — abstract, batch form
- `encode_query(text: str) -> dict` — concrete default, delegates to `encode_queries([text])[0]`

This keeps the pipeline call site simple (`encode_query(question)`) while the abstract contract only requires implementing the batch form.

---

### 3. pymilvus 2.4 → 2.6 and milvus_lite 2.4 → 3.0 upgrade

#### Storage format change: file → directory

milvus_lite 2.4.x stored everything in a single SQLite file (e.g. `./milvus.db`).  
milvus_lite 3.0 treats that path as a **directory** and calls `os.makedirs()` on it.

If the old flat file still exists, `os.makedirs` raises `FileExistsError` because a file is in the way of the directory it wants to create.

**Fix:** delete the old flat file. milvus_lite 3.0 will create a directory at the same path containing its own internal data files. The URI format still requires a `.db` suffix — pymilvus validates this before handing the path to milvus_lite.

```
# Before (milvus_lite 2.4.x) — a file:
./milvus.db   (192 KB SQLite file)

# After (milvus_lite 3.0) — a directory:
./milvus.db/
  ├── meta.db
  └── ...
```

The config path `./milvus.db` is unchanged; only the on-disk representation changed.

#### ORM API deprecation

pymilvus 2.6 deprecates the ORM-style API (`connections`, `Collection`, `utility`) in favour of `MilvusClient`. The ORM API will be removed in pymilvus 3.1.

| ORM API (deprecated) | MilvusClient API |
|---|---|
| `connections.connect(uri=...)` | `MilvusClient(uri=...)` |
| `utility.has_collection(name)` | `client.has_collection(name)` |
| `Collection(name).drop()` | `client.drop_collection(name)` |
| `Collection(name).insert(entities)` | `client.insert(collection_name, data)` |
| `Collection(name).search(...)` | `client.search(collection_name, ...)` |
| `Collection(name).hybrid_search(...)` | `client.hybrid_search(collection_name, ...)` |
| `hit.entity.get("text")` / `hit.score` | `hit["entity"]["text"]` / `hit["distance"]` |

The schema creation API also changed:

```python
# MilvusClient schema builder
schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field(field_name="id",        datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text",      datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="sparse",    datatype=DataType.SPARSE_FLOAT_VECTOR)

index_params = MilvusClient.prepare_index_params()
index_params.add_index(field_name="embedding", index_type="AUTOINDEX",            metric_type="IP")
index_params.add_index(field_name="sparse",    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

client.create_collection(collection_name=..., schema=schema, index_params=index_params)
```

---

### 4. gRPC GOAWAY — root cause and fix

#### Symptom

```
Got goaway [11] err=UNAVAILABLE: too_many_pings
Received GOAWAY with error code ENHANCE_YOUR_CALM
Current keepalive time (before throttling): 10000ms
```

Followed — minutes later — by:

```
DescribeCollectionException: collection 'documents' does not exist
```

The "collection does not exist" error was **not** a missing collection. The collection was created and persisted to disk. The error was pymilvus's interpretation of a failed gRPC call on a dead channel.

#### Root cause

pymilvus's `GrpcHandler` hardcodes these gRPC channel options:

```python
"grpc.keepalive_time_ms": 10000,              # ping every 10 seconds
"grpc.keepalive_permit_without_calls": True,   # ← ping even when idle
```

`keepalive_permit_without_calls: True` means the gRPC client sends a keepalive ping every 10 seconds **regardless of whether any RPC is in flight**. During the ~10 minutes spent generating 1,405 embeddings via Ollama (with zero Milvus calls happening), the client sent ~60 idle pings. milvus_lite's embedded gRPC server responded with `ENHANCE_YOUR_CALM` (HTTP/2 error code 11), closing the channel.

#### Why reconnecting did not help

`connections.connect()` (ORM API) reuses an existing connection alias and does not rebuild the underlying gRPC channel if one already exists for that alias.

`MilvusClient` pools connections by `address|token` key. Creating a new `MilvusClient` pointing to the same URI returns the same pooled handler with the same dead channel.

#### Fix

`grpc_options` passed to `MilvusClient` are merged over the defaults in `GrpcHandler._setup_grpc_channel`:

```python
self._client = MilvusClient(
    uri=settings.MILVUS_DB_PATH,
    grpc_options={"grpc.keepalive_permit_without_calls": False},
)
```

Setting `keepalive_permit_without_calls` to `False` stops idle pings entirely. Keepalive pings are only sent while a Milvus RPC is actively in progress — which is never the case during embedding generation. The GOAWAY no longer occurs.

---

### 5. Lazy client initialisation

The `MilvusClient` is not created at `MilvusStore.__init__` time. It is created on first use via `_get_client()`:

```python
def _get_client(self) -> MilvusClient:
    if self._client is None:
        self._client = MilvusClient(uri=..., grpc_options=...)
    return self._client
```

**Why:** constructing `MilvusStore` starts a milvus_lite gRPC server in a background thread. If the client were created eagerly at construction time but the first actual Milvus operation happened 10 minutes later, the connection window would be unnecessarily long. Lazy initialisation means the connection is opened at the point of first use, not at object construction.

---

### 6. Batched insertion

1,405 entities inserted as a single payload stressed the gRPC message size. Insertion is batched in groups of 200:

```python
for start in range(0, len(entities), batch_size):
    batch = entities[start : start + batch_size]
    client.insert(collection_name=self.collection_name, data=batch)
```

`batch_size=200` is a parameter on `insert_embeddings`, tunable if the chunk count changes significantly.

---

### 7. Collection lifecycle — `--clean` flag drops collection after `__init__`

#### Symptom

```
milvus_exceptions.MilvusException: collection 'documents' does not exist
```

This error appeared at `insert_embeddings` time, without any preceding GOAWAY — the gRPC channel was healthy, and the collection genuinely was not there.

#### Root cause

`main.py` calls `drop_collection()` only when `--clean` is passed. The sequence is:

```
RAGPipeline.__init__()        # → create_collection() — collection created here
    ...
if args.clean:
    vector_store.drop_collection()  # collection dropped HERE
if vector_store.is_empty():
    pipeline.add_documents(...)     # → insert_embeddings() — collection is gone
```

`__init__` creates the collection eagerly. `--clean` drops it afterwards. Nothing recreates it before `insert_embeddings` runs.

#### Fix

`insert_embeddings` calls `create_collection` at its own start, before any insert:

```python
def insert_embeddings(self, embeddings, texts, sparse_vectors=None, batch_size=200):
    # Re-create if dropped (e.g. --clean) or lost between init and insert.
    self.create_collection(dim=len(embeddings[0]))
    ...
```

`create_collection` is idempotent — it returns early if the collection already exists with data and the correct schema. On the happy path this costs one `has_collection` RPC. On the `--clean` path it recreates the collection. Both paths succeed.

This also covers less common scenarios: milvus_lite server restart between `__init__` and the first insert, or a schema migration that dropped an empty collection.

---

### 8. Collection load state — `--eval` without re-ingestion

#### Background

Milvus separates **storage** from **memory**. After a collection is created and data is inserted, it may not be in the `Loaded` state (i.e. mapped into memory for search). The `Loaded` state is required for both `search` and `hybrid_search`.

During a normal ingestion run this is usually not visible — milvus_lite auto-loads after a fresh `create_collection`. But when running with `--eval` against an already-populated store (no ingestion step), the collection may be in `NotLoad` or `Loading` state from a previous session.

#### Load states

`LoadState` is an enum in `pymilvus.client.types`:

| State | Meaning |
|---|---|
| `NotLoad` (1) | Collection exists on disk but is not in memory |
| `Loading` (2) | Collection is currently being loaded into memory |
| `Loaded` (3) | Collection is in memory — ready for search |

#### Fix

`_ensure_collection_loaded()` is called at the top of `search()`, the single method that requires the collection to be in memory:

```python
def _ensure_collection_loaded(self) -> None:
    client = self._get_client()
    if not client.has_collection(self.collection_name):
        return  # not yet created — ingestion handles creation
    state = client.get_load_state(collection_name=self.collection_name)["state"]
    if state != LoadState.Loaded:
        client.load_collection(self.collection_name)
```

`load_collection` is a no-op if the collection is already `Loaded`, and blocks until loading is complete if it is in `NotLoad` or `Loading`. One call handles all three cases.

**Why not call this from `_get_client()`?**  
`_get_client()` is called by every method — `is_empty`, `drop_collection`, `create_collection`, `upsert`. None of those require the collection to be in memory. Checking load state on every call adds 2 unnecessary RPCs (one `has_collection`, one `get_load_state`) to every non-search operation, including `drop_collection` which would load a collection just to immediately drop it. Placing the check only in `search()` keeps the overhead where it is needed.

---

### 9. BM25 model persistence across sessions

#### Problem

The BM25 encoder is fitted on the full chunk corpus during ingestion (`bm25.fit(chunks)`). The fitted model — IDF weights, corpus statistics — lives only in memory. When the process exits, it is lost.

On the next run with `--eval` or a standalone query (no ingestion), `bm25.is_fitted` is `False` and the pipeline silently falls back to dense-only retrieval. The sparse vectors in Milvus are never used.

#### Fix

After fitting, the model is saved to disk; on startup it is restored if the file exists:

```python
# add_documents — after fit():
self.bm25.save(settings.BM25_MODEL_PATH)   # serialises IDF stats to JSON

# __init__ — before any retrieval:
if self.bm25 is not None and os.path.exists(settings.BM25_MODEL_PATH):
    self.bm25.load(settings.BM25_MODEL_PATH)
```

`BM25EmbeddingFunction.save()` writes a JSON file containing `corpus_size`, `avgdl`, and the full `idf` table. `load()` restores it and sets `is_fitted = True`.

The `--clean` path re-fits on fresh data and overwrites the file, keeping the saved model consistent with the sparse vectors in Milvus.

`BM25_MODEL_PATH` is configurable via `settings` alongside other store paths.

---

### 10. `VectorStore` ABC — store-agnostic interface

#### Why

`RAGPipeline` originally imported `MilvusStore` directly, making it impossible to swap in Elasticsearch, Azure AI Search, Pinecone, or any other store without changing the pipeline.

#### Interface design

The two key design decisions:

**Single `search` method with a `mode` parameter** — not separate `retrieve` / `hybrid_retrieve` methods. Elasticsearch and Azure AI Search perform hybrid retrieval in a single server-side request; forcing them into a two-method split would require them to fake a separation that doesn't exist in the underlying API.

**Two optional query parameters** — `query_text` and `query_sparse` — rather than one:

| Parameter | Used by |
|---|---|
| `query_text` | Stores with native BM25 (Elasticsearch, Azure AI Search) — raw text passed to the store, BM25 handled server-side |
| `query_sparse` | Stores needing pre-computed sparse vectors (Milvus, Pinecone) — pipeline encodes with `SparseEncoder` and passes the result |

```python
class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        limit: int,
        query_text: Optional[str] = None,
        query_sparse: Optional[Dict[int, float]] = None,
        mode: str = "hybrid",          # "dense" | "sparse" | "hybrid"
    ) -> List[Dict]: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def drop_collection(self) -> None: ...

    def create_collection(self, dim: int, **kwargs) -> None: pass  # no-op for serverless
    def close(self) -> None: pass
```

`create_collection` is a concrete no-op default — serverless stores (Pinecone Serverless) have no collection lifecycle concept and should not be forced to implement it.

#### Effect on the pipeline

`RAGPipeline` now imports `VectorStore` instead of `MilvusStore`. The `retrieve()` method collapses to a single `search()` call:

```python
query_sparse = self.bm25.encode_query(question) if (self.bm25 is not None and self.bm25.is_fitted) else None
results = self.vector_store.search(
    query_embedding=question_embedding,
    query_text=question,
    query_sparse=query_sparse,
    limit=top_k,
    mode="hybrid" if query_sparse is not None else "dense",
)
```

The pipeline passes both `query_text` (for native-BM25 stores) and `query_sparse` (for Milvus/Pinecone). Each store implementation uses whichever is relevant and ignores the other.

#### Store comparison

| Store | `upsert` sparse_vectors | `search` uses | SparseEncoder in pipeline? |
|---|---|---|---|
| Milvus | Required — stored as field | `query_sparse` via `hybrid_search` + RRF | Yes |
| Elasticsearch | Ignored — BM25 native | `query_text` via single `knn`+`query` request | No |
| Azure AI Search | Ignored — BM25 native | `query_text` via hybrid query API | No |
| Pinecone | Required — stored as sparse index | `query_sparse` via single hybrid request | Yes |
| pgvector | Ignored — no sparse support | `query_embedding` only, mode forced to `"dense"` | No |
| Chroma | Ignored | `query_embedding` only, mode forced to `"dense"` | No |
