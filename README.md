# feature/ragas-integration — Comprehensive Learning Record

> **Branch purpose:** Integrate native RAGAS evaluation framework alongside the custom LLM-as-judge eval from the baseline submission. Empirically characterise compatibility, limitations, and production suitability with local models (Ollama + llama3.2).
>
> **Outcome summary:** RAGAS 0.4.3 is architecturally unstable for local model deployments. The custom eval framework on `main` is more operationally suitable for regulated environments. RAGAS has value as a reference implementation and for cloud-model deployments. Full findings documented below.

---

## Table of Contents

1. [Environment](#1-environment)
2. [Every Problem We Hit — Chronological](#2-every-problem-we-hit--chronological)
3. [Metrics Reference — All Metrics Worked With](#3-metrics-reference--all-metrics-worked-with)
4. [RAGAS Benefits](#4-ragas-benefits)
5. [RAGAS Tradeoffs and Risks](#5-ragas-tradeoffs-and-risks)
6. [What Production Actually Needs](#6-what-production-actually-needs)
7. [Open Source and Third-Party Alternatives](#7-open-source-and-third-party-alternatives)
8. [Decision Framework — Which Tool When](#8-decision-framework--which-tool-when)
9. [Files Added in This Branch](#9-files-added-in-this-branch)
10. [Next Branches](#10-next-branches)
11. [Interview Talking Points](#11-interview-talking-points)
12. [References](#12-references)

---

## 1. Environment

```
ragas==0.4.3
Python==3.12
Ollama (local, CPU)
  - tinyllama 1.1B    — RAG generator
  - llama3.2 3B       — LLM judge
  - nomic-embed-text  — embeddings (768 dimensions)
Milvus Lite          — vector store (file-persisted)
```

---

## 2. Every Problem Hit — Chronological

This section is a complete debugging record. Every error, root cause, and resolution documented in sequence.

---

### Problem 1 — `discrete_metric` ImportError

**Error:**
```
ImportError: cannot import name 'discrete_metric' from 'ragas.metrics'
```

**Root cause:** `discrete_metric` is a decorator introduced in RAGAS **v0.4.x**. The installed version was `0.2.x` (pinned with `ragas = "^0.2"` in pyproject.toml). `^0.2` resolves to the latest compatible version — which was 0.4.3, not 0.2.x as intended.

**Lesson:** `^` in pyproject.toml means "compatible with" — for RAGAS this allowed a major architectural version jump. **Always pin eval framework dependencies exactly.**

**Resolution:**
```toml
# Wrong — allows major version jumps
ragas = "^0.2"

# Correct — exact pin
ragas = "0.2.14"
```

---

### Problem 2 — Wrong metric names

**Error:**
```
ImportError: cannot import name 'answer_relevance' from 'ragas.metrics'
Did you mean: '_answer_relevance'?
```

**Root cause:** RAGAS renamed metrics between versions. `answer_relevance` (snake_case, no trailing 'y') does not exist. The correct name is `answer_relevancy` (with 'y'). Additionally `answer_correctness` (lowercase) is the old v0.1 instance; `AnswerCorrectness` (PascalCase class) is the v0.4 pattern.

**Metric name mapping across RAGAS versions:**

| What you might try | What actually exists (0.4.3) |
|--------------------|------------------------------|
| `answer_relevance` | `answer_relevancy` (singleton) / `AnswerRelevancy` (class) |
| `answer_correctness` | `answer_correctness` (singleton) / `AnswerCorrectness` (class) |
| `context_relevance` | `ContextRelevance` (class only in collections) |
| `ResponseRelevancy` | Does not exist — use `AnswerRelevancy` |

**Always check what's available before assuming:**
```bash
python3 -c "import ragas.metrics.collections; print(dir(ragas.metrics.collections))"
```

---

### Problem 3 — Deprecated import path

**Warning:**
```
DeprecationWarning: Importing faithfulness from 'ragas.metrics' is deprecated
and will be removed in v1.0. Please use 'ragas.metrics.collections' instead.
```

**Root cause:** RAGAS 0.4.x moved all metrics to `ragas.metrics.collections`. The old `ragas.metrics` path is deprecated but still functional. v1.0 will remove it.

**Status:** Not fixed — see Problem 5 for why migrating to collections is not viable with `evaluate()`.

---

### Problem 4 — `collections` metrics require `InstructorLLM`

**Error:**
```
ValueError: Collections metrics only support modern InstructorLLM.
Found: OllamaLLM.
Use: llm_factory('gpt-4o-mini', client=openai_client)
```

**Root cause:** `ragas.metrics.collections` classes validate their LLM argument at instantiation:
```python
# Inside ragas/metrics/collections/base.py
def _validate_llm(self):
    if not isinstance(self.llm, InstructorLLM):
        raise ValueError("Collections metrics only support modern InstructorLLM...")
```

`InstructorLLM` is RAGAS's own LLM abstraction built around the `instructor` Python library, which enforces Pydantic schema compliance at the OpenAI client level. Custom `OllamaLLM`, `LangchainLLMWrapper`, and any other non-RAGAS LLM class is rejected.

**Resolution:** Use `llm_factory()` pointed at Ollama's OpenAI-compatible endpoint:
```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
ragas_llm = llm_factory("llama3.2", client=client)
```

---

### Problem 5 — MRO mismatch: collections metrics fail `evaluate()` validation

**Error:**
```
TypeError: All metrics must be initialised metric objects,
e.g: metrics=[BleuScore(), AspectCritic()]
```

**Root cause:** This is a **genuine RAGAS 0.4.3 bug** — the most significant finding of this branch.

`ragas.evaluate()` validates metrics with:
```python
# ragas/evaluation.py line 131
if any(not isinstance(m, Metric) for m in metrics):
    raise TypeError("All metrics must be initialised metric objects...")
```

Collections metrics inherit from `BaseMetric → SimpleBaseMetric`. `ragas.metrics.base.Metric` is a **completely separate class hierarchy**.

**Verified:**
```python
from ragas.metrics.base import Metric
from ragas.metrics.collections import Faithfulness

m = Faithfulness(llm=ragas_llm)
print(type(m).__mro__)
# ['Faithfulness', 'BaseMetric', 'SimpleBaseMetric', 'NumericValidator', 'BaseValidator', 'ABC', 'object']
# Metric is NOT in this chain

isinstance(m, Metric)  # False — fails evaluate() validation
```

Collections metrics are correctly instantiated but fail the `isinstance` check because the two architectures don't share a common base class. The tutorials tell you to use `ragas.metrics.collections`; `evaluate()` rejects them. **This is an internal RAGAS consistency failure.**

**Workaround:** Use the deprecated singleton API which DOES inherit from `Metric`:
```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision
# lowercase = pre-instantiated Metric instances — pass isinstance check
```

---

### Problem 6 — `collections` embeddings require `embedding_factory`

**Error:**
```
ValueError: Collections metrics only support modern embeddings.
Found: OllamaRagasEmbeddings.
Use: embedding_factory('openai', model='text-embedding-ada-002', ...)
```

**Same pattern as Problem 4** — collections metrics validate embeddings too:
```python
def _validate_embeddings(self):
    if not isinstance(self.embeddings, ModernEmbeddings):
        raise ValueError("Collections metrics only support modern embeddings...")
```

**Resolution:**
```python
from ragas.embeddings import OpenAIEmbeddings

ragas_embeddings = OpenAIEmbeddings(
    model="nomic-embed-text",
    client=client,  # AsyncOpenAI pointed at Ollama
)
```

---

### Problem 7 — `embedding_factory` deprecated

**Warning:**
```
DeprecationWarning: Importing embedding_factory from ragas.embeddings is deprecated.
Import directly from ragas.embeddings.base or use modern providers:
from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings
```

Another internal RAGAS migration — `embedding_factory` was the recommended import in the docs but deprecated in the same version. Use `OpenAIEmbeddings` directly.

---

### Problem 8 — `evaluate` import shadowed by LangSmith

**Error:** `evaluate()` raised LangSmith-specific errors unrelated to RAGAS.

**Root cause:** LangSmith also exports an `evaluate` function. If both are installed and imported, the wrong one can be called.

**Resolution:** Always alias on import:
```python
from ragas import evaluate as ragas_evaluate
```

Never rely on unaliased `evaluate` when LangSmith is in the dependency tree.

---

### Problem 9 — `InstructorLLM` structured output failure with llama3.2 3B

**Error:**
```
InstructorRetryException: 1 validation error for StatementGeneratorOutput
statements: Field required
```

**Root cause:** This is the most important finding of the branch — a **model capability ceiling**, not a code bug.

RAGAS `Faithfulness` uses a two-stage decompose-then-verify approach. Stage 1 asks the judge to return:
```json
{"statements": ["claim 1", "claim 2", "claim 3"]}
```

`llama3.2` 3B consistently returns the **JSON schema definition** instead of an instance:
```json
{
  "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
  "required": ["statements"],
  "title": "StatementGeneratorOutput",
  "type": "object"
}
```

The `instructor` library catches the Pydantic validation failure and retries 3 times. All three attempts return identical schema echoes. `InstructorRetryException` is raised. RAGAS records `nan` for faithfulness.

**Why this happens:** `instructor`-style structured output enforcement requires the model to understand the difference between a JSON schema definition and a JSON schema instance. This is an emergent capability that appears reliably in GPT-4 class models (70B+ parameters) but not in sub-7B models. llama3.2 3B has not learned to instantiate schemas — it echoes what it sees.

**Attempted fix — `RobustOllamaInstructorLLM`:**
```python
class RobustOllamaInstructorLLM(InstructorLLM):
    """
    Intercepts InstructorRetryException, falls back to direct httpx call
    with format:'json' and explicit field-extraction prompt.
    """
    async def agenerate(self, prompt, response_model):
        from instructor.exceptions import InstructorRetryException
        try:
            return await super().agenerate(prompt=prompt, response_model=response_model)
        except InstructorRetryException:
            return await self._call_ollama_direct(prompt, response_model)
```

See `src/evaluation/robust_llm.py` for full implementation.

---

### Problem 10 — `OpenAIEmbeddings` missing `embed_query`

**Error:**
```
AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'
```

**Root cause:** `ragas.embeddings.OpenAIEmbeddings` in 0.4.3 implements async methods (`aembed_query`, `aembed_documents`) but not their synchronous counterparts. The `answer_relevancy` metric internally calls `embed_query()` synchronously. **The RAGAS 0.4.3 embeddings class is incomplete.**

**Fix — `RobustOllamaEmbeddings`:**
```python
class RobustOllamaEmbeddings(RagasOpenAIEmbeddings):
    def embed_query(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_documents(texts))
```

---

### Problem 11 — Timeout failures with local models

**Error:**
```
Exception raised in Job[0]: TimeoutError()
{'faithfulness': nan, 'answer_relevancy': nan, 'context_precision': 1.0}
```

**Root cause:** RAGAS default timeout (60s) is too short for local Ollama on CPU. The two-stage faithfulness eval takes 60-120 seconds per test case. RAGAS also defaults to parallel job execution (`max_workers > 1`), which overwhelms a local CPU model — requests queue and timeout simultaneously.

**Resolution:**
```python
from ragas.run_config import RunConfig

results = ragas_evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(
        timeout=300,      # 5 minutes per job
        max_retries=3,
        max_workers=1,    # sequential — critical for local models
    )
)
```

`max_workers=1` is the most important parameter for local deployment. Parallel execution is designed for API calls where latency is network-bound. For local CPU inference, parallelism causes contention.

---

### Problem 12 — Dataset schema requirements

**Root cause:** RAGAS `evaluate()` requires exact column names. Case-sensitive. No aliases.

**Required schema:**
```python
{
    "user_input":         str,        # the question
    "response":           str,        # the generated answer
    "retrieved_contexts": List[str],  # MUST be a list of strings, not a string
    "reference":          str,        # expected/ground truth answer
}
```

**Critical:** `retrieved_contexts` must be a list of strings per row. A joined string will fail or produce incorrect scores. Even with one chunk: `[context_string]` not `context_string`.

---

### Problem 13 — Metrics passed as classes not instances

**Error:**
```
TypeError: All metrics must be initialised metric objects,
e.g: metrics=[BleuScore(), AspectCritic()]
```

When using the deprecated singleton API, the singletons are already instances — no `()` needed. When using collections classes, `()` is required. Mixing the two causes confusion.

```python
# Singleton API — already instantiated, no ()
from ragas.metrics import faithfulness
metrics = [faithfulness]  # correct

# Collections API — must instantiate
from ragas.metrics.collections import Faithfulness
metrics = [Faithfulness(llm=ragas_llm)]  # correct
metrics = [Faithfulness]  # wrong — class not instance
```

---

### Final scores achieved

After all fixes (old singleton API + RunConfig + RobustOllamaInstructorLLM + RobustOllamaEmbeddings):

```
context_precision:   1.00   — completed, score likely inflated by small judge model
faithfulness:        nan    — InstructorRetryException persisting despite override
answer_relevancy:    nan    — embed_query fallback inconsistent
```

`context_precision` completing at 1.0 with llama3.2 as judge is suspicious — likely the judge is too generous on a small corpus where every chunk is broadly relevant to the query topic.

**Baseline custom eval scores for comparison (main branch):**
```
SimpleChunker overall:       0.60
answer_relevance:            ~0.65
faithfulness:                ~0.58
answer_correctness:          ~0.62
hallucination_rate:          ~0.55 (higher = less hallucination)
context_relevance:           ~0.45  ← weakest metric
```

---

## 3. Metrics Reference — All Metrics Worked With

### 3.1 Custom metrics (baseline submission — main branch)

These five metrics were implemented from scratch using a single-prompt LLM-as-judge pattern.

---

#### `answer_relevance`

**What it measures:** Whether the generated answer is relevant to and directly addresses the question asked.

**Implementation pattern:** Single LLM judge prompt asking to score relevance holistically on 0-1 scale.

**Failure mode it catches:** Off-topic or tangential answers — the generator produces a factually correct statement that doesn't answer the specific question.

**RAGAS equivalent:** `answer_relevancy` / `AnswerRelevancy`

**Reference:** [RAGAS paper Section 3.2](https://arxiv.org/abs/2309.15217)

---

#### `faithfulness`

**What it measures:** Whether the answer is grounded in the retrieved context — are the claims in the answer supported by what was actually retrieved?

**Implementation pattern (custom — single prompt):**
```
"Given this context and this answer, rate how faithful the answer
is to the context on a scale of 0 to 1."
```

**Implementation pattern (RAGAS — two-stage decompose then verify):**
```
Stage 1: "Break this answer into individual atomic claims."
         → ["claim 1", "claim 2", "claim 3"]
Stage 2: "For each claim, is it supported by the context?"
         → [true, false, true]
Score = supported_claims / total_claims = 2/3 = 0.67
```

**Why RAGAS approach is superior:** A holistic judge can be lenient — it might give 0.7 to an answer with one hallucinated claim among three true ones. Claim-level verification forces the judge to evaluate each assertion independently. Harder to be systematically lenient.

**Cost:** 2 LLM calls per test case vs 1 for custom implementation.

**Failure mode it catches:** Confabulation — the generator produces plausible-sounding but ungrounded statements that go beyond what the context actually says.

**Reference:** [RAGAS paper Section 3.3](https://arxiv.org/abs/2309.15217) | [GitHub prompt source](https://github.com/vibrantlabsai/ragas/blob/main/src/ragas/metrics/collections/faithfulness)

---

#### `answer_correctness`

**What it measures:** Semantic similarity between the generated answer and the expected (ground truth) answer. Requires a reference answer in the test case.

**Implementation pattern:** LLM judge comparing semantic meaning, not token overlap.

**Why LLM-as-judge over BLEU/ROUGE:** BLEU and ROUGE measure token overlap — two answers that are semantically identical but worded differently score near zero. An LLM judge evaluates meaning.

**RAGAS equivalent:** `AnswerCorrectness` / `FactualCorrectness`

**Failure mode it catches:** Answers that are grounded in the context but miss the key point of the expected answer — correct-sounding but substantively wrong.

**Reference:** [RAGAS FactualCorrectness docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)

---

#### `hallucination_rate`

**What it measures:** Specific claim-level detection of ungrounded content. Returns a list of identified hallucinated claims, not just an aggregate score. Higher score = less hallucination.

**Implementation pattern:** Judge prompt asks to identify specific statements in the answer that cannot be verified from the retrieved context.

**Distinction from faithfulness:** Faithfulness scores the proportion of grounded content. Hallucination rate specifically lists what is hallucinated — more actionable for debugging.

**RAGAS equivalent:** No direct equivalent. `Faithfulness` covers the grounding aspect. `NoiseSensitivity` is related.

**Failure mode it catches:** Answers that mix accurate retrieved information with hallucinated additions — passes a holistic faithfulness check but contains specific false claims.

---

#### `context_relevance`

**What it measures:** Whether the retrieved context chunks are actually relevant to answering the question. Evaluates the retriever layer, not the generator.

**Implementation pattern:** LLM judge assesses whether the retrieved chunks contain information useful for answering the question.

**Key architectural insight:** This metric evaluates `retrieve()`, not `query()`. Low context_relevance points to a retrieval problem (chunking strategy, embedding model, top_k setting) — not a generation problem. The clean `retrieve()` / `query()` separation in the pipeline makes this layer independently evaluable.

**Finding from submission:** Context_relevance was the weakest metric (0.45) and the root cause was identified as context volume deficit — AdvancedChunker at top_k=3 retrieved 29% less context surface than SimpleChunker at the same top_k.

**RAGAS equivalent:** `ContextRelevance` (in collections, available in 0.4.3) / `context_precision` (singleton)

**Reference:** [RAGAS Context Precision docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)

---

### 3.2 RAGAS built-in metrics (attempted in this branch)

---

#### `Faithfulness` / `faithfulness`

**Class:** `ragas.metrics.collections.Faithfulness` (0.4.x) | `ragas.metrics.faithfulness` (singleton, deprecated)

**Input fields:** `response`, `retrieved_contexts`

**Scoring:** Decompose answer into atomic claims → verify each against context → score = supported/total

**Judge calls per test case:** 2 (decomposition + verification)

**Small model behaviour:** llama3.2 3B echoes the Pydantic schema back instead of filling it in. Results in `InstructorRetryException` and `nan` score.

**Reference:** [RAGAS Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)

---

#### `AnswerRelevancy` / `answer_relevancy`

**Class:** `ragas.metrics.collections.AnswerRelevancy` | `ragas.metrics.answer_relevancy` (singleton)

**Input fields:** `user_input`, `response`, embeddings model

**Scoring:** Embeds question and answer independently, computes cosine similarity. Does NOT use LLM judge — uses embeddings. This is why it requires an embeddings object at instantiation.

**Important distinction:** This is the only core RAGAS metric that is embedding-based rather than LLM-judge-based. Faster and cheaper than LLM judge metrics. Also means it measures semantic similarity, not semantic correctness — an irrelevant but topically adjacent answer can score well.

**Small model behaviour:** `OpenAIEmbeddings.embed_query()` missing in 0.4.3 — `AttributeError` until `RobustOllamaEmbeddings` override applied.

**Reference:** [RAGAS Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)

---

#### `ContextPrecision` / `context_precision`

**Class:** `ragas.metrics.collections.ContextPrecision` | `ragas.metrics.context_precision` (singleton)

**Input fields:** `user_input`, `retrieved_contexts`, `reference`

**Scoring:** For each retrieved chunk, LLM judge determines whether it was useful for answering the question given the ground truth reference. Precision@k — are the most relevant chunks ranked highest?

**Distinction from context_relevance:** Context_relevance asks "is this chunk relevant to the question?" Context_precision asks "given that we know the correct answer, were the right chunks retrieved and ranked correctly?"

**Requires ground truth:** Yes — `reference` column must contain expected answer.

**Variants in 0.4.3:**
- `ContextPrecisionWithReference` — uses reference answer for evaluation
- `ContextPrecisionWithoutReference` — reference-free version

**Result in this branch:** 1.0 on 7 test cases — likely inflated by small judge and small corpus.

**Reference:** [RAGAS Context Precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)

---

#### `ContextRecall`

**Class:** `ragas.metrics.collections.ContextRecall` | `ragas.metrics.context_recall` (singleton)

**Input fields:** `reference`, `retrieved_contexts`

**Scoring:** Does the retrieved context contain all the information needed to answer the question correctly? Measures coverage — complement of precision. Low recall means the retriever missed relevant information.

**Not used in this branch** — included here for completeness as the complement to context_precision.

**Precision vs Recall tradeoff:**
```
High precision, low recall  → Retrieved chunks are relevant but incomplete
High recall, low precision  → Retrieved all needed info but with noise
```

**Reference:** [RAGAS Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

---

#### `FactualCorrectness`

**Class:** `ragas.metrics.collections.FactualCorrectness`

**Input fields:** `response`, `reference`

**Scoring:** Claim-level factual comparison between generated answer and reference answer. More rigorous than `AnswerCorrectness` — uses the same decompose-then-verify pattern as Faithfulness but compares against reference rather than context.

**Reference:** [RAGAS Factual Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)

---

### 3.3 Traditional NLP metrics (available in RAGAS, not used)

These are deterministic, non-LLM metrics — fast, cheap, but poorly correlated with human judgment for open-ended generation tasks.

| Metric | What it measures | Limitation for RAG |
|--------|-----------------|-------------------|
| **BLEU** | N-gram overlap with reference | Penalises paraphrase — semantically identical answers score near zero if worded differently |
| **ROUGE** | Recall-oriented n-gram overlap | Same paraphrase problem as BLEU, better for summarisation |
| **CHRF** | Character n-gram F-score | Better than BLEU/ROUGE for morphologically rich languages |
| **Exact Match** | String equality | Only useful for structured outputs (SQL, code) |
| **Semantic Similarity** | Embedding cosine similarity | Faster than LLM judge, misses logical errors |

**When to use these:** In a hybrid eval stack — fast deterministic metrics as a first filter (cheap), LLM-as-judge metrics on borderline cases (expensive). Exact match is appropriate for structured RAG outputs like form field extraction.

---

### 3.4 Metric selection guide for RAG pipelines

```
Question you're asking              →  Metric to use
─────────────────────────────────      ──────────────────────────────
Does the answer address the query?  →  answer_relevancy (fast, embedding-based)
Is the answer grounded in context?  →  faithfulness (LLM judge, 2 calls)
Does the answer match ground truth? →  factual_correctness / answer_correctness
Is the retriever finding the right  →  context_relevance + context_recall
  chunks?
Are chunks ranked correctly?        →  context_precision
Is there specific hallucination?    →  hallucination_rate (custom), noise_sensitivity
```

---

## 4. RAGAS Benefits

**Research-backed metrics:** RAGAS faithfulness decompose-then-verify approach is published in a peer-reviewed paper and has been validated against human judgment. The multi-stage prompting is more rigorous than ad-hoc single-prompt scoring.

**Reference-free evaluation:** Most RAGAS metrics (faithfulness, answer_relevancy, context_precision without reference) don't require ground truth labels. You can evaluate production queries without pre-annotated datasets — significant operational advantage.

**Synthetic test data generation:** RAGAS can generate question/answer pairs from your corpus automatically — reduces the effort of building golden evaluation sets.

**Standardised vocabulary:** Using RAGAS metric names creates a shared language between engineering, product, and research teams. "Faithfulness 0.73" means the same thing across organisations that have adopted RAGAS.

**Experiment tracking integration:** RAGAS integrates with LangSmith, Weights & Biases, and Arize Phoenix for persistent experiment tracking and score trending over time.

**Active development:** Frequent releases, large community, active issue resolution. The framework is evolving rapidly toward a production-capable state.

---

## 5. RAGAS Tradeoffs and Risks

### 5.1 API instability — the most critical risk

RAGAS has made **two major breaking architectural changes** in quick succession:

```
v0.1  →  Original four metrics, basic evaluate() API
v0.2  →  Extended metrics, LangChain integration
v0.3  →  Intermediate
v0.4  →  Collections + InstructorLLM + experiment() API
          BREAKS v0.2 patterns documented in official tutorials
          Old evaluate() incompatible with new collections metrics
```

The tutorials at `docs.ragas.io` document patterns from v0.2 that silently fail in v0.4.3. The migration guide exists but is incomplete. **What the docs say and what the code does are inconsistent.**

### 5.2 InstructorLLM model requirements

`InstructorLLM` requires models that can instantiate Pydantic JSON schemas — not merely return JSON. This is a capability that emerges reliably only in GPT-4 class models (estimated 70B+ effective parameters).

```
Models that work reliably:    GPT-4o, GPT-4-turbo, Claude Sonnet+
Models that fail:             llama3.2 3B, tinyllama 1.1B, Mistral 7B (inconsistent)
Borderline:                   Llama 3.1 70B, Mixtral 8x7B
```

**Implication:** RAGAS 0.4.x collections metrics effectively require a paid API for reliable operation. This is a cost and data sovereignty constraint for regulated environments.

### 5.3 Dependency tree complexity

RAGAS 0.4.3 pulls in:
- `instructor` — structured output enforcement
- `langchain-core` — LLM abstractions
- `openai` — client library
- `datasets` (HuggingFace) — dataset management
- `pydantic` v2 — schema validation

Each of these has its own release cadence. A patch to `instructor` or `pydantic` can break RAGAS silently. This is a large attack surface for a library sitting on the deployment gate critical path.

### 5.4 Benchmark contamination risk (eval quality)

RAGAS judge prompts are public and widely used. If your fine-tuning data includes RAGAS-evaluated outputs, the judge model may be biased toward scoring RAGAS-style responses highly — inflating your eval scores without improving real quality.

### 5.5 Cost at production scale

At production scale with GPT-4 class judge (required for reliable InstructorLLM):

```
Faithfulness alone (2 LLM calls per query):
  1,000 queries/day × 2 calls × ~500 tokens = 1M tokens/day
  At GPT-4o pricing: ~$5-10/day just for faithfulness evaluation

Full eval suite (5 metrics, mixed LLM + embedding):
  Estimated $15-30/day for 1,000 production queries
```

Mitigation: Run expensive metrics (faithfulness) only on sampled or low-confidence responses. Run cheap metrics (answer_relevancy, embedding-based) on every response.

### 5.6 Goodhart's Law exposure

Once teams optimise for RAGAS scores, Goodhart's Law applies: the metrics become targets and cease to be good measures. A pipeline can be tuned to score well on faithfulness (by being conservative and grounding everything) while performing poorly on user satisfaction (by refusing to synthesise or interpret).

---

## 6. What Production Actually Needs

A production RAG eval framework for a regulated environment (superannuation, banking, government) requires:

### 6.1 Dependency stability

```toml
# Exact pins — never ranges for eval-critical libraries
ragas = "0.2.14"

# Upgrade process:
# 1. Branch: chore/ragas-upgrade-x.x.x
# 2. Pin new version
# 3. Run upgrade_validator.py against golden test set
# 4. Block merge if any metric regresses > 5%
# 5. Update this README
```

### 6.2 Version-controlled judge prompts

The LLM judge prompts must be version-controlled alongside application code — not embedded in a third-party library you don't control. If RAGAS changes its faithfulness prompt between releases, your scores change silently and you can't reproduce historical results.

```
/src/evaluation/
├── prompts/
│   ├── faithfulness_v1.txt     # pinned prompt, never change without new version
│   ├── hallucination_v1.txt
│   └── context_relevance_v1.txt
```

### 6.3 Eval as a deployment gate

```python
DEPLOYMENT_THRESHOLDS = {
    "faithfulness":       0.70,   # minimum before promoting to production
    "answer_relevancy":   0.65,
    "context_relevance":  0.55,
    "hallucination_rate": 0.60,   # higher = less hallucination
}

def can_deploy(eval_results: dict) -> bool:
    for metric, threshold in DEPLOYMENT_THRESHOLDS.items():
        if eval_results.get(metric, 0) < threshold:
            return False
    return True
```

### 6.4 Tiered eval strategy

Not every query needs expensive full eval. Production tiering:

```
Tier 1 — every response (cheap, <100ms):
  answer_relevancy (embedding cosine similarity, no LLM call)
  length/format checks (deterministic)

Tier 2 — low-confidence responses (moderate, <5s):
  faithfulness single-prompt (1 LLM call)
  hallucination_rate spot check

Tier 3 — HITL escalation queue (expensive, async):
  faithfulness decompose-then-verify (2 LLM calls)
  human review
  feedback loop into golden test set
```

### 6.5 Score trending and alerting

Eval scores must be tracked over time — not just run once. A score that was 0.72 last week and is 0.61 this week is a degradation signal even if 0.61 is above the deployment threshold.

```python
# Production monitoring pattern
ALERT_THRESHOLD_DELTA = -0.05   # alert if any metric drops > 5% week-over-week

def check_regression(current: dict, previous: dict) -> List[str]:
    alerts = []
    for metric in current:
        delta = current[metric] - previous.get(metric, current[metric])
        if delta < ALERT_THRESHOLD_DELTA:
            alerts.append(f"{metric} degraded: {previous[metric]:.2f} → {current[metric]:.2f}")
    return alerts
```

### 6.6 Audit trail for APRA CPS234

In a regulated environment (Care Super, NAB), every AI output that reaches a member or advisor must have a demonstrable control. The eval framework is that control. Required:

- Every query → eval scores logged with correlation ID
- Low-confidence queries → HITL queue with reviewer identity logged
- Eval score distribution reports available for regulatory review
- Judge model version and prompt version logged alongside scores

---

## 7. Open Source and Third-Party Alternatives

### 7.1 Comparison matrix

| Framework | Local models | RAG-specific | Observability | Regulated env | Active (2025) |
|-----------|-------------|-------------|---------------|---------------|---------------|
| **RAGAS** | Partial (0.2.x) | ✓ Best-in-class | Via integrations | Risk (instability) | ✓ |
| **DeepEval** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **TruLens** | ✓ | ✓ (RAG Triad) | ✓ OpenTelemetry | ✓ | ✓ |
| **Arize Phoenix** | ✓ Self-host | ✓ | ✓ Best | ✓ Self-hostable | ✓ |
| **Opik** | ✓ | ✓ | ✓ | ✓ Self-hostable | ✓ |
| **LangSmith** | ✗ Cloud only | ✓ | ✓ | Risk (data leaves) | ✓ |
| **Custom (main)** | ✓ | ✓ | Build yourself | ✓ Full control | N/A |

---

### 7.2 DeepEval

**What it is:** pytest-compatible LLM evaluation framework with 50+ metrics covering RAG, agents, multi-turn conversations, safety, and hallucination.

**Key strengths:**
- Works with local models natively — no InstructorLLM requirement
- pytest integration — evals run as part of your standard test suite in CI/CD
- G-Eval metric — custom evaluation criteria without writing judge prompts from scratch
- RAG component-level diagnosis — identifies whether embedding model, chunking, reranker, or prompt is underperforming

**Relevant for your pipeline:**
```python
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What work did EE do for IG Group?",
    actual_output=pipeline.query(question),
    retrieval_context=[pipeline.retrieve(question)],
    expected_output=expected_answer,
)

faithfulness = FaithfulnessMetric(threshold=0.7, model="ollama/llama3.2")
faithfulness.measure(test_case)
print(faithfulness.score, faithfulness.reason)
```

**Tradeoff vs RAGAS:** Broader metric library, better local model support, pytest integration. Less research pedigree than RAGAS's paper-backed approach. Metric implementations may differ from RAGAS's definitions.

**GitHub:** https://github.com/confident-ai/deepeval
**Docs:** https://deepeval.com/docs

---

### 7.3 TruLens

**What it is:** OpenTelemetry-based LLM tracing + evaluation framework. Pioneered the RAG Triad: context relevance, groundedness, answer relevance.

**Key strengths:**
- OpenTelemetry tracing — every LLM call, retrieval, and eval spans are recorded and queryable
- Real-time dashboard — production monitoring out of the box
- Human feedback loops — annotation queues built in
- Works with local models via LangChain wrappers

**The RAG Triad (TruLens's core contribution):**
```
Context Relevance:  Is retrieved context relevant to the query?
Groundedness:       Is the answer grounded in the context?
Answer Relevance:   Does the answer address the query?

These three metrics together form a complete RAG eval surface.
```

**Best for:** Teams that need combined eval + observability — one tool for both rather than RAGAS (eval) + separate observability stack.

**GitHub:** https://github.com/truera/trulens
**Docs:** https://www.trulens.org

---

### 7.4 Arize Phoenix

**What it is:** Open-source LLM observability platform — started in traditional ML monitoring, expanded into LLM tracing and evaluation.

**Key strengths:**
- Self-hostable — no data leaves your infrastructure (critical for regulated environments)
- Embedding visualisation — t-SNE/UMAP plots of your retrieved chunks reveal clustering issues invisible in aggregate scores
- Framework-agnostic — instruments any pipeline regardless of LangChain/LlamaIndex usage
- OpenTelemetry native — standard instrumentation protocol

**Relevant for Care Super:** Self-hosting capability means member data never hits a third-party cloud. Embedding visualisation helps diagnose why context_relevance is low without running expensive LLM eval on every query.

**GitHub:** https://github.com/Arize-ai/phoenix
**Docs:** https://docs.arize.com/phoenix

---

### 7.5 Opik (by Comet ML)

**What it is:** Self-hostable LLM evaluation and monitoring platform with dataset management, experiment tracking, and prompt optimisation.

**Key strengths:**
- Self-hostable — Docker deployment, data sovereignty maintained
- End-to-end testing — from prompt to production monitoring in one tool
- Experiment tracking — compare eval results across pipeline versions
- Native LangChain and LlamaIndex integration

**Tradeoff vs Phoenix:** Opik has stronger experiment management and prompt optimisation. Phoenix has better embedding visualisation and broader framework support.

**GitHub:** https://github.com/comet-ml/opik
**Docs:** https://www.comet.com/docs/opik

---

### 7.6 LangSmith

**What it is:** LangChain's commercial LLM ops platform — tracing, evaluation, dataset management, and annotation queues.

**Key strengths:**
- Tight LangChain/LangGraph integration — if your pipeline uses LangChain, minimal instrumentation required
- Annotation queues — human review workflows built in
- Dataset management — curate golden sets from production traces

**Significant limitation for regulated environments:** Cloud-only — all traces including query content and retrieved context are sent to LangChain's servers. Incompatible with data sovereignty requirements for member data (Care Super, NAB). **Do not use in production for financial services without explicit legal review.**

**GitHub:** https://github.com/langchain-ai/langsmith-sdk
**Docs:** https://docs.smith.langchain.com

---

### 7.7 Custom eval framework (main branch baseline)

The custom implementation in the baseline submission is a legitimate production choice — not just a prototype.

**Advantages over all third-party frameworks:**

```
Zero external dependencies on the critical path
  → No breaking changes without your action
  → No transitive dependency conflicts
  → Works with any model, any deployment

Version-controlled judge prompts
  → Prompt changes are code reviews
  → Historical scores reproducible
  → Prompt drift is a git diff

Full observability ownership
  → No data leaving your infrastructure
  → Audit trail is your own logs
  → APRA CPS234 compliance is straightforward

Coupling to your specific domain
  → Judge prompts can be tuned for superannuation terminology
  → Thresholds calibrated against your specific corpus
  → Test cases reflect real member queries
```

**When to prefer a third-party framework:**

- Team doesn't want to maintain judge prompts
- Synthetic test data generation is needed
- Cross-team standardisation on metric definitions is required
- Observability dashboard is needed without building one

**Recommendation for Care Super:**

```
Development/CI:      Custom eval framework (main branch)
                     Stable, local, no external dependencies

Production monitoring: Arize Phoenix (self-hosted)
                       Embedding visualisation + tracing
                       Data stays on-premise

Quarterly deep eval:  RAGAS pinned to stable version
                      Run against golden set
                      Compare against custom framework scores
                      Document delta and root cause
```

---

## 8. Decision Framework — Which Tool When

```
START
  │
  ├── Do you need data sovereignty (financial services, health, government)?
  │     YES → Arize Phoenix (self-hosted) + Custom eval framework
  │     NO  → continue
  │
  ├── Is your pipeline LangChain-native?
  │     YES → LangSmith for tracing, RAGAS or DeepEval for eval metrics
  │     NO  → continue
  │
  ├── Do you need CI/CD eval gates with pytest?
  │     YES → DeepEval (pytest-native, local model support)
  │     NO  → continue
  │
  ├── Do you need production observability (traces, dashboards)?
  │     YES → TruLens (eval + observability in one) or Arize Phoenix
  │     NO  → continue
  │
  ├── Are you using local models (Ollama, self-hosted)?
  │     YES → DeepEval, TruLens, or Custom (avoid RAGAS 0.4.x collections)
  │     NO  → RAGAS with GPT-4 class judge (reliable InstructorLLM)
  │
  └── Are you primarily doing offline/development evaluation?
        YES → RAGAS 0.2.x pinned, or Custom eval framework
        NO  → See production monitoring options above
```

---

## 9. Files Added in This Branch

```
src/evaluation/
├── ragas_evaluator.py          # RAGAS evaluation runner — full pipeline
└── robust_llm.py               # RAGAS compatibility shims for local models
    ├── RobustOllamaInstructorLLM
    │     Subclass of InstructorLLM
    │     Intercepts InstructorRetryException
    │     Falls back to direct httpx + format:'json' (proven baseline pattern)
    │     Manually validates Pydantic response_model from parsed JSON
    │
    └── RobustOllamaEmbeddings
          Subclass of ragas.embeddings.OpenAIEmbeddings
          Adds missing embed_query() and embed_documents() sync methods
          Wraps async methods with run_until_complete()
```

---

## 10. Next Branches

| Branch | What it addresses | Key experiment |
|--------|------------------|----------------|
| `feature/hybrid-search` | context_relevance deficit (0.45) | BM25 + dense vector + RRF — does hybrid retrieval improve context scores? |
| `feature/reranking` | retrieval precision | Cross-encoder between retrieve() and query() — measures precision improvement vs latency cost |
| `feature/query-expansion` | retrieval recall | Multi-query before embedding — does wider net improve context_relevance? |
| `feature/azure-openai` | model quality ceiling | Swap tinyllama for GPT-4o — how much do scores improve with production model? |

---

## 11. Interview Talking Points

**On RAGAS API instability:**
> "RAGAS 0.4.x has a genuine MRO mismatch between its collections metric classes and the evaluate() validator — correctly instantiated metrics fail isinstance() checks because the two architectures don't share a common base class. This is an internal RAGAS consistency failure that the migration docs don't mention. For a regulated deployment gate where a silent framework break means hallucinating answers reach production users, that's not an acceptable dependency. The custom eval framework on main is more operationally stable — owned prompts, no external dependencies, reproducible results."

**On InstructorLLM + small models:**
> "RAGAS 0.4.x uses the instructor library to enforce Pydantic schemas at the OpenAI client level — more robust than prompt-based JSON enforcement. But it requires models that can instantiate JSON schemas, not just return JSON. llama3.2 3B consistently echoes the schema definition back rather than filling it in. This is a model capability that emerges reliably only in GPT-4 class models. Sub-7B models generally cannot do it. This is the root cause of the faithfulness nan scores — not a configuration issue."

**On the decompose-then-verify pattern:**
> "RAGAS faithfulness is more rigorous than our submission's holistic scoring because it decomposes the answer into atomic claims first, then verifies each independently. A judge can be lenient on a holistic 'rate this 0-1' prompt. It's harder to be systematically lenient on four individual claim verifications. The cost is two LLM calls per test case. The right production pattern is cheap holistic gate on every response, expensive decompose-then-verify only on low-confidence answers routed to HITL."

**On eval framework selection for Care Super:**
> "For a superannuation fund with APRA CPS234 obligations, data sovereignty is non-negotiable. LangSmith is cloud-only — member queries leaving the infrastructure is not acceptable. RAGAS requires GPT-4 class models for reliable structured output which adds cost and OpenAI dependency. The pragmatic stack is: custom eval framework for CI deployment gates (stable, local, auditable), Arize Phoenix self-hosted for production observability (embedding visualisation, trace storage on-prem), and quarterly RAGAS runs on a golden set for benchmark comparison."

---

## 12. References

### Papers
- **RAGAS original paper:** Es, S. et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217. https://arxiv.org/abs/2309.15217
- **RAGChecker:** Ru, S. et al. (2024). RAGChecker: A Fine-grained Framework for Diagnosing RAG. arXiv:2408.08067
- **EncouRAGe:** Local, Fast, Reliable RAG Evaluation. arXiv:2511.04696

### RAGAS documentation
- Metrics overview: https://docs.ragas.io/en/stable/concepts/metrics/overview/
- Faithfulness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
- Context Precision: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
- Context Recall: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
- Answer Relevancy: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
- Factual Correctness: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/
- Migration v0.3 to v0.4: https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
- RAG evaluation tutorial: https://docs.ragas.io/en/stable/tutorials/rag/

### Framework repositories
- RAGAS: https://github.com/vibrantlabsai/ragas
- DeepEval: https://github.com/confident-ai/deepeval
- TruLens: https://github.com/truera/trulens
- Arize Phoenix: https://github.com/Arize-ai/phoenix
- Opik: https://github.com/comet-ml/opik
- instructor library: https://github.com/instructor-ai/instructor

### Pydantic validation
- Pydantic v2 error reference: https://errors.pydantic.dev/2.10/v/missing
- StatementGeneratorOutput schema echo error — root cause documented in this branch

### Evaluation landscape
- RAGAS vs TruLens vs DeepEval comparison (2026): https://atlan.com/know/llm-evaluation-frameworks-compared/
- Top RAG evaluation tools (2026): https://www.getmaxim.ai/articles/the-5-best-rag-evaluation-tools-you-should-know-in-2026/
- DeepEval alternatives: https://latitude.so/blog/deepeval-alternatives
