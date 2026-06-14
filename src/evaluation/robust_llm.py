import json
import httpx
import typing as t
from src.llm.ollama import OllamaLLM
from ragas.llms.base import InstructorLLM, InstructorTypeVar

import nest_asyncio
nest_asyncio.apply()

class RobustOllamaInstructorLLM(InstructorLLM):
    """
    Subclass of InstructorLLM that handles small model structured output failure.

    llama3.2 3B echoes JSON schema instead of instantiating it.
    Override intercepts InstructorRetryException, retries with explicit
    field-extraction prompt via direct httpx call — our proven pattern.
    """

    def _is_schema_echo(self, content: str) -> bool:
        """True if model returned schema definition instead of data instance."""
        try:
            parsed = json.loads(content)
            return all(k in parsed for k in ["properties", "title", "type"])
        except (json.JSONDecodeError, TypeError):
            return False

    def _empty_instance(self, response_model: t.Type[InstructorTypeVar]) -> InstructorTypeVar:
        """Last-resort fallback — empty but valid Pydantic instance."""
        fields = list(response_model.model_fields.keys())
        defaults = {}
        for field_name, field_info in response_model.model_fields.items():
            annotation = field_info.annotation
            # Default empty value based on type hint
            if annotation in (list, t.List[str]):
                defaults[field_name] = []
            elif annotation == str:
                defaults[field_name] = ""
            elif annotation == float:
                defaults[field_name] = 0.0
            else:
                defaults[field_name] = None
        return response_model(**defaults)

    async def _call_ollama_direct(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """
        Direct httpx call bypassing instructor entirely.
        Uses format:'json' — our proven pattern from the submission.
        """
        fields = list(response_model.model_fields.keys())

        extraction_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY a valid JSON object with these fields "
            f"filled in with actual extracted values: {fields}\n"
            f"Do NOT describe or return the schema. "
            f"Return the data itself."
        )

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": extraction_prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.01},
                },
            )
            response.raise_for_status()
            raw = response.json()["response"]

        if self._is_schema_echo(raw):
            # Model still echoing schema after fallback — return empty instance
            return self._empty_instance(response_model)

        try:
            parsed = json.loads(raw)
            return response_model(**parsed)
        except Exception:
            return self._empty_instance(response_model)

    async def agenerate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """
        Override agenerate — try instructor path first, fallback on failure.
        """
        from instructor.exceptions import InstructorRetryException

        try:
            # Normal instructor path
            return await super().agenerate(
                prompt=prompt,
                response_model=response_model,
            )
        except InstructorRetryException:
            # Schema-echo or validation failure — use direct httpx fallback
            return await self._call_ollama_direct(prompt, response_model)

    def generate(
        self,
        prompt: str,
        response_model: t.Type[InstructorTypeVar],
    ) -> InstructorTypeVar:
        """Sync wrapper over agenerate."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.agenerate(prompt, response_model)
        )
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
import typing as t

from ragas.embeddings import BaseRagasEmbeddings
class RobustOllamaEmbeddings(RagasOpenAIEmbeddings):
    """
    Subclass of RAGAS OpenAIEmbeddings adding missing embed_query method.
    
    ragas==0.4.3 OpenAIEmbeddings missing embed_query — called by
    answer_relevancy metric internally.
    """

    def embed_query(self, text: str) -> t.List[float]:
        """Sync embed a single query string."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_query(text))

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Sync embed a list of documents."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aembed_documents(texts))
    

import asyncio
import typing as t
import httpx
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings


class RobustOllamaEmbeddings(RagasOpenAIEmbeddings):
    """
    Complete embeddings implementation for Ollama + RAGAS compatibility.
    
    ragas.embeddings.OpenAIEmbeddings in 0.4.3 is incomplete:
    - Missing embed_query() sync method
    - Missing aembed_query() async method  
    - Missing embed_documents() sync method
    - Missing aembed_documents() async method
    
    All four implemented here via direct httpx calls to Ollama embed endpoint.
    Bypasses the broken OpenAIEmbeddings implementation entirely.
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        # Don't call super().__init__() — it requires an OpenAI client
        # we don't want to use
        self.model = model
        self.base_url = base_url

    async def _embed_single(self, text: str) -> t.List[float]:
        """Core async embed call — direct httpx to Ollama."""
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]

    async def _embed_batch(self, texts: t.List[str]) -> t.List[t.List[float]]:
        """Embed multiple texts concurrently."""
        tasks = [self._embed_single(text) for text in texts]
        return await asyncio.gather(*tasks)

    # ── Async methods (called by RAGAS internally) ──

    async def aembed_query(self, text: str) -> t.List[float]:
        return await self._embed_single(text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return await self._embed_batch(texts)

    # ── Sync methods (called by answer_relevancy internally) ──

    def embed_query(self, text: str) -> t.List[float]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._embed_single(text))

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._embed_batch(texts))