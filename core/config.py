"""
GrantBridge Configuration Module
Loads environment variables and provides typed config access.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present, else fall back to env.example
_env_path = Path(__file__).parent.parent / ".env"
_example_path = Path(__file__).parent.parent / "env.example"

if _env_path.exists():
    load_dotenv(_env_path)
elif _example_path.exists():
    load_dotenv(_example_path)

class Config:
    # IBM WatsonX
    IBM_API_KEY: str = os.getenv("IBM_API_KEY", "")
    IBM_PROJECT_ID: str = os.getenv("IBM_PROJECT_ID", "")
    IBM_WATSONX_URL: str = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    GRANITE_MODEL_ID: str = os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2")
    GRANITE_EMBEDDING_MODEL: str = os.getenv("GRANITE_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "grant_knowledge_base")

    # Security
    APP_SECRET_KEY: str = os.getenv("APP_SECRET_KEY", "grantbridge-dev-secret")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

    # Rate Limiting & Operational
    MAX_API_CALLS_PER_MINUTE: int = int(os.getenv("MAX_API_CALLS_PER_MINUTE", "20"))
    MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4096"))
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
    MAX_EXECUTION_SECONDS: int = int(os.getenv("MAX_EXECUTION_SECONDS", "120"))

    # Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")

    # Operational
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_AUDIT_TRAIL: bool = os.getenv("ENABLE_AUDIT_TRAIL", "True").lower() == "true"

    @classmethod
    def has_ibm_credentials(cls) -> bool:
        return bool(cls.IBM_API_KEY and cls.IBM_API_KEY != "your_ibm_api_key_here"
                    and cls.IBM_PROJECT_ID and cls.IBM_PROJECT_ID != "your_ibm_project_id_here")

config = Config()
