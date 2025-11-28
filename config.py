import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path
from typing import List

load_dotenv()


# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist and are created if they don't
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class Config(BaseSettings):
    # VLM 
    vlm_model_path: str = "NexaAI/Qwen3-VL-4B-Instruct-GGUF/Qwen3-VL-4B-Instruct.Q4_0.gguf"
    mmproj_path: str = "NexaAI/Qwen3-VL-4B-Instruct-GGUF/mmproj.F32.gguf"
    gpu_layers: int = 0
    max_tokens: int = 200
    plugin_id: str = "nexaml"
    
    # Embedder
    embedder_model_path: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedder_batch_size: int = 32


    # Database
    db_path: Path = DATA_DIR / "image_database.json"
    
    # Search Settings
    min_similarity: float = 0.3
    top_k: int = 5
    faiss: bool = False

    # Files
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    
    # Logging (INFO, DEBUG, WARNING, ERROR)
    log_level: str = "INFO"

    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"

config = Config()