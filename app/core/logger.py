import logging
import sys

def get_logger(name:str) -> logging.Logger:

    # 1. Logger nesnesini oluştur (Bu isimle bir muhabir yarat)
    logger = logging.getLogger(name)
    
    # 2. Seviyeyi ayarla (INFO ve üzerini yakala: INFO, WARNING, ERROR...)
    logger.setLevel(logging.INFO)

    # 3. Eğer bu logger'ın daha önce ayarlanmış bir handler'ı yoksa...
    if not logger.handlers:
        # 4. Handler (Yazıcı) oluştur: Logları konsola (sys.stdout) bas
        handler = logging.StreamHandler(sys.stdout)
        
        # 5. Formatı belirle: Zaman | İsim | Seviye | Mesaj
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        
        # 6. Yazıcıya bu formatı uygula
        handler.setFormatter(formatter)
        
        # 7. Muhabire (logger) bu yazıcıyı (handler) ekle
        logger.addHandler(handler)
        
    # 8. Hazırlanan logger nesnesini geri döndür
    return logger
