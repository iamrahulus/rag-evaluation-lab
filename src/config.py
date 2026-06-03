from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CASE_STUDIES_PATH: str = "./case_studies.json"
    MILVUS_DB_PATH: str = "./milvus.db"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    COLLECTION_NAME: str = "documents"

    # LLM Models
    LLM_MODEL: str = "llama3.2"

    # Model Parameters
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    MAX_TOKENS: int = 2048
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSION: int = 768

    # Chunking Settings
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50
    DEFAULT_TOP_K: int = 3


settings = Settings()
