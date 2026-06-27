import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # --- Api keys ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_WHATSAPP_NUMBER: str = os.getenv("TWILIO_WHATSAPP_NUMBER", "")

    # --- Model settings ---
    GEMINI_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100

    # --- System paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_DIR: str = os.path.join(BASE_DIR, "chroma_db")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")

    def validate_keys(self):
        """تأكد من أن جميع المفاتيح الأساسية صالحة"""
        missing = [key for key in ["GOOGLE_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"] if not getattr(self, key)]
        if missing:
            raise ValueError(f"❌ خطأ حرج: المتغيرات التالية ناقصة في ملف البيئة: {', '.join(missing)}")


settings = Settings()
settings.validate_keys()