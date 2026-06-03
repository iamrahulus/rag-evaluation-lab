 ## :warning: Please read these instructions carefully and entirely first
* Clone this repository to your local machine.
* Use your IDE of choice to complete the assignment.
* When you have completed the assignment, you need to  push your code to this repository and [mark the assignment as completed by clicking here]({{submission_link}}).
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

