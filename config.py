import os
import pathlib
from typing import Optional, List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, PostgresDsn

BASE_DIR = pathlib.Path(__file__).resolve().parent
DOTENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Life Science CRM Backend"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    PROJECT_VERSION: str = "0.1.0"
    DATABASE_URL: Optional[PostgresDsn] = None
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_SCHEMA: str = "public"

    # 🔵 GROQ API KEY (REPLACES GEMINI_API_KEY)
    GROQ_API_KEY: str

    # Twilio WhatsApp Configuration
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_WHATSAPP_NUMBER: str = ""

    # Gmail Configuration
    GMAIL_SENDER_EMAIL: str = ""
    GMAIL_APP_PASSWORD: str = ""

    BACKEND_CORS_ORIGINS: Union[str, List[str]] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(f"Invalid value for BACKEND_CORS_ORIGINS: {v}")

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()
Settings.model_rebuild()