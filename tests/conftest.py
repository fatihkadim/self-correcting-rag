import os

# Testler gerçek .env'e veya API anahtarına bağımlı olmamalı.
# app.core.config import edilmeden önce zorunlu ayarları doldur.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MODEL_NAME", "test-model")
