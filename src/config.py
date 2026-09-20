import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)


class _Section:
    def __init__(self, data: dict):
        self.__dict__.update(data)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


class Config:
    app = _Section(_cfg["app"])
    database = _Section(_cfg["database"])
    rag = _Section(_cfg["rag"])
    nlp = _Section(_cfg["nlp"])
    classification = _Section(_cfg["classification"])
    risk_scoring = _Section(_cfg["risk_scoring"])
    security = _Section(_cfg["security"])
    escalation = _Section(_cfg["escalation"])
    llm = _Section(_cfg["llm"])

    # Secrets from environment
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    ROOT: Path = _ROOT
    DATA_DIR: Path = _ROOT / "data"
    KB_DIR: Path = _ROOT / "data" / "knowledge_base"


config = Config()
