![Uygulama Ekranı](assets/mimari.png)

# 🧠 Self-Correcting RAG

> **Kanıtlanabilir ve savunulabilir cevaplar üreten, kendi kendini düzelten bir RAG sistemi.**

Klasik RAG sistemleri cevap üretir ama doğrulamaz. Bu proje ürettiği her cevabı **iddialara (claims) böler**, her iddiayı **kanıtlarla doğrular** ve gerektiğinde **kendi kendini düzeltir**.

---

## 🎯 Proje Felsefesi

Bu proje "en hızlı cevap" değil, **kanıtlanabilir ve savunulabilir cevap** üretmeyi hedefler.

| Klasik RAG | Self-Correcting RAG |
|---|---|
| Cevap üretir ve sunar | Cevap üretir, iddialara böler, doğrular |
| LLM'e güvenir | LLM'i sorgular |
| Hallucination kontrolü yok | Claim-level hallucination tespiti |
| Tek geçiş (one-shot) | İteratif, geri beslemeli döngü |

---

## 🏗️ Sistem Mimarisi

```
User Query
    │
    ▼
┌─────────────────┐
│   Retrieval     │  ← High-Recall mode (geniş bağlam)
│   (Qdrant)      │
└────────┬────────┘
         │ chunks
         ▼
┌─────────────────┐
│ Answer Generator│  ← LLM (low temperature)
│   (LLM)         │
└────────┬────────┘
         │ answer
         ▼
┌─────────────────┐
│ Claim Extractor │  ← Cevabı doğrulanabilir iddialara böler
└────────┬────────┘
         │ claims[]
         ▼
┌─────────────────┐
│ Verification    │  ← Her claim için High-Precision re-retrieval
│ Engine          │     + LLM Judge → Supported / Refuted / Unknown
└────────┬────────┘
         │ verification result
         ▼
┌─────────────────────────┐
│ Self-Correction         │  ← Karar: Accept / Retry / Repair
│ Controller (Agent Core) │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│ Answer Repair   │  ← Sadece doğrulanmış claim'lerden yeni cevap
└────────┬────────┘
         │
         ▼
    Final Answer
```

---

## 📁 Klasör Yapısı

```
self-correcting-rag/
│
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── pipeline.py             # Ana RAG pipeline sınıfı
│   │
│   ├── api/
│   │   └── routes.py           # /query endpoint
│   │
│   ├── core/
│   │   ├── config.py           # Env & model ayarları (pydantic-settings)
│   │   ├── logger.py           # Merkezi logging altyapısı
│   │   └── settings.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py        # QdrantRetriever (High-Recall / High-Precision)
│   │   ├── chunking.py         # Recursive semantic chunking + overlap
│   │   ├── embedder.py         # Sentence Transformers embedding
│   │   ├── vector_store.py     # Qdrant vektör deposu işlemleri
│   │   └── query_rewrite.py    # Retry için sorgu reformülasyonu
│   │
│   ├── generation/
│   │   ├── answer.py           # İlk cevap üretim modülü
│   │   └── repair.py           # Doğrulanmış claim'lerden cevap yeniden yazımı
│   │
│   ├── claims/
│   │   ├── extractor.py        # LLM tabanlı claim extraction
│   │   └── models.py           # Claim Pydantic modelleri
│   │
│   ├── verification/
│   │   ├── verifier.py         # Claim verification engine
│   │   └── judge.py            # LLM Judge / NLI değerlendirme
│   │
│   ├── agent/
│   │   ├── controller.py       # Self-correction karar mantığı (agentic core)
│   │   └── policies.py         # Retry / stop politikaları
│   │
│   ├── prompts/
│   │   ├── answer.txt          # Cevap üretim prompt'u
│   │   ├── claim_extract.txt   # Claim extraction prompt'u
│   │   ├── verify.txt          # Doğrulama prompt'u
│   │   └── repair.txt          # Cevap düzeltme prompt'u
│   │
│   ├── schemas/
│   │   ├── query.py            # QueryRequest / QueryResponse
│   │   ├── retrieval.py        # Chunk / RetrievalResult
│   │   ├── claims.py           # Claim / ClaimType
│   │   ├── verification.py     # ClaimVerification / VerificationStatus
│   │   ├── answer.py
│   │   └── response.py
│   │
│   └── evaluation/
│       ├── metrics.py          # Özel metrikler (hallucination rate, retry count)
│       └── baseline.py         # Klasik RAG karşılaştırması
│
├── data/
│   ├── raw/                    # Kaynak dokümanlar
│   ├── processed/              # Chunk'lanmış veriler
│   └── eval/                   # Test soruları & ground truth
│
├── scripts/
│   ├── ingest.py               # Doküman yükleme & indexleme
│   └── reindex.py              # Vector DB yeniden indexleme
│
├── tests/
│   ├── test_claims.py
│   ├── test_verification.py
│   └── test_agent.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/
│   ├── projeaciklamasi.md      # Detaylı proje açıklaması
│   └── sprint_plan.md          # 15 sprint'lik öğrenim planı
│
├── .env                        # Ortam değişkenleri (git'e ekleme!)
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.11+
- Docker & Docker Compose

### 1. Repoyu Klonla

```bash
git clone https://github.com/fatihkadim/self-correcting-rag.git
cd self-correcting-rag
```

### 2. Sanal Ortam ve Bağımlılıklar

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. Ortam Değişkenlerini Ayarla

`.env` dosyası oluştur:

```env
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
MAX_RETRY=3
CONFIDENCE_THRESHOLD=0.7
TOP_K=5
OPENAI_API_KEY=sk-...
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. Qdrant'ı Başlat (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 5. API'yi Çalıştır

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🚀 Kullanım

### Sağlık Kontrolü

```bash
curl http://localhost:8000/health
```

### Sorgu Gönder

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Python ne zaman geliştirildi?"}'
```

**Örnek yanıt:**

```json
{
  "answer": "Python, Guido van Rossum tarafından 1991 yılında yayımlandı.",
  "sources": [...],
  "claims": [
    {
      "claim": "Python 1991 yılında yayımlandı.",
      "status": "supported",
      "confidence": 0.95,
      "evidence": ["..."]
    }
  ]
}
```

### İnteraktif API Dökümantasyonu

Uygulama çalışırken tarayıcıda açın: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧩 Temel Kavramlar

### Claim Nedir?
Tek başına **doğru veya yanlış** olarak değerlendirilebilen minimum bilgi birimi.

| ✅ Claim | ❌ Claim Değil |
|---|---|
| "Python 1991'de geliştirildi." | "Python güzel bir dildir." |
| "FastAPI async destekler." | "Bu framework kullanışlıdır." |

### Verification Statüleri
| Statü | Anlam |
|---|---|
| `supported` | Kanıt bulundu ve doğrulandı |
| `refuted` | Kanıt bulundu ve çürütüldü |
| `unknown` | Yeterli kanıt bulunamadı |

### Retrieval Modları
| Mod | Kullanım | top_k | Threshold |
|---|---|---|---|
| `HIGH_RECALL` | İlk cevap üretimi | 10 | 0.3 |
| `HIGH_PRECISION` | Claim doğrulama | 3 | 0.7 |

---

## 📊 Değerlendirme Metrikleri

- **Claim doğruluk oranı** — Kaç claim doğrulandı?
- **Hallucination azalma oranı** — Klasik RAG'a göre ne kadar az hata?
- **Retry sayısı** — Self-correction kaç kez devreye girdi?
- **Token maliyeti** — Ek doğrulama adımlarının maliyeti

---

## 🗺️ Sprint Planı (15 Sprint · ~45 Saat)

| Grup | Sprintler | Konu |
|---|---|---|
| 🟢 Temel | 1–4 | Kurulum, Config, Mock Retrieval, Pipeline |
| 🟡 Retrieval | 5–7 | Qdrant, Chunking, Gerçek Retrieval |
| 🟠 Generation | 8–9 | LLM Entegrasyonu, Prompt Engineering |
| 🔴 Claims | 10–11 | Claim Extraction, Schema |
| 🔴 Verification | 12–13 | Verification Engine, Answer Repair |
| 🟣 Final | 14–15 | Self-Correction Controller, Evaluation |

Detaylar için → [`docs/sprint_plan.md`](docs/sprint_plan.md)

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Veri Doğrulama | Pydantic v2 |
| Vector DB | Qdrant |
| Embedding | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | OpenAI API (gpt-4o-mini) / Ollama |
| Konteyner | Docker + Docker Compose |

---

## 📜 Lisans

MIT
