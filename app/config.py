import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "ekt-reorder-agent"
DEFAULT_API_TOKEN = "change-me-token"
DEFAULT_POSTGRES_DSN = "postgresql+psycopg2://postgres:postgres@localhost:5432/ekt"

API_TOKEN = os.getenv("APP_API_TOKEN", DEFAULT_API_TOKEN)
POSTGRES_DSN = os.getenv("POSTGRES_DSN", DEFAULT_POSTGRES_DSN)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
