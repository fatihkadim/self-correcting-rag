**🇬🇧 [English](#self-correcting-rag) | 🇹🇷 [Türkçe](#self-correcting-rag-tr)**

# Self-Correcting RAG

> **A self-correcting RAG system that produces provable and defensible answers.**

Classical RAG systems produce answers but do not verify them. This project **breaks down every answer it produces into claims**, **verifies each claim with evidence**, and **self-corrects** when necessary.

---

## Project Philosophy

This project aims to produce **provable and defensible answers**, rather than just the "fastest answer".

| Classical RAG | Self-Correcting RAG |
|---|---|
| Produces and presents the answer | Produces answer, breaks into claims, verifies |
| Trusts the LLM | Questions the LLM |
| No hallucination control | Claim-level hallucination detection |
| Single pass (one-shot) | Iterative, feedback loop |

---

## System Architecture

After generating the initial answer to the user's question, the system follows these steps:
1. Extracts each independent statement of information (claim) within the answer.
2. Performs a separate, high-precision data search for each claim (Evidence Check).
3. Compares the evidence with the LLM Judge and labels the claims as supported, refuted, or unknown.
4. If there are unverified claims, it rewrites the answer using only the verified information (Answer Repair).
5. If none of the claims are supported, it reformulates the query and tries again (Query Rewrite + Retry).

![System Architecture](./assets/mimari.png)

---

## Folder Structure

```
self-correcting-rag/
│
├── app/
│   ├── main.py                 # FastAPI entrypoint (Web UI and API)
│   ├── pipeline.py             # Basic RAG pipeline
│   │
│   ├── static/
│   │   └── index.html          # Web Interface (Glassmorphism Agent UI)
│   │
│   ├── core/
│   │   ├── config.py           # Env & model settings (pydantic-settings)
│   │   └── logger.py           # Centralized logging
│   │
│   ├── retrieval/
│   │   ├── retriever.py        # QdrantRetriever (High-Recall / High-Precision)
│   │   ├── reranker.py         # CrossEncoder based reranking
│   │   ├── chunking.py         # Recursive semantic chunking + overlap
│   │   ├── embedder.py         # Sentence Transformers embedding
│   │   └── query_rewriter.py   # Query reformulation using LLM
│   │
│   ├── generation/
│   │   ├── llm.py              # OpenAI API client
│   │   ├── answer.py           # Initial answer generation
│   │   └── repair.py           # Answer rewriting from verified claims
│   │
│   ├── claims/
│   │   └── extractor.py        # LLM-based claim extraction
│   │
│   ├── verification/
│   │   ├── verifier.py         # Claim verification engine
│   │   └── judge.py            # LLM Judge / NLI evaluation
│   │
│   ├── agent/
│   │   ├── controller.py       # Self-correction logic (agentic core)
│   │   └── policies.py         # Accept / Repair / Retry policies
│   │
│   ├── utils/
│   │   └── prompts.py          # PromptLoader helper
│   │
│   ├── prompts/
│   │   ├── answer_system.txt       # Answer generation system prompt
│   │   ├── answer_user.txt         # Answer generation user prompt
│   │   ├── claim_extract.txt       # Claim extraction prompt
│   │   ├── claim_extract_system.txt# Claim extraction system prompt
│   │   ├── verify_user.txt         # Verification prompt
│   │   ├── verify_system.txt       # Verification system prompt
│   │   ├── repair.txt              # Answer repair prompt
│   │   ├── repair_system.txt       # Answer repair system prompt
│   │   ├── query_rewrite.txt       # Query reformulation prompt
│   │   └── query_rewrite_system.txt# Query reformulation system prompt
│   │
│   ├── schemas/
│   │   ├── query.py            # Query schemas
│   │   ├── retrieval.py        # Retrieval schemas
│   │   ├── claims.py           # Claim schemas
│   │   └── verification.py     # Verification schemas
│   │
│   ├── evaluation/
│   │   ├── ragas_eval.py       # RAGAS metric evaluation
│   │   ├── baseline.py         # Classical RAG vs Self-Correcting RAG
│   │   └── metrics.py          # Claim accuracy metrics
│   │
│   └── scripts/
│       └── generate_dataset.py # Automated test generation from Qdrant
│
├── scripts/
│   ├── ingest.py               # Document ingestion & indexing
│   └── reindex.py              # Vector DB reindexing
│
├── tests/
│   ├── test_verification.py    # Verification engine tests
│   ├── test_agent.py           # Agent controller tests
│   ├── test_api.py             # FastAPI endpoint tests
│   ├── test_chunking.py        # Chunking module tests
│   ├── test_claims.py          # Claim extraction tests
│   ├── test_policies.py        # Policy logic tests
│   ├── test_query_rewriter.py  # Query rewriter tests
│   └── test_repair.py          # Answer repair tests
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── assets/
│   ├── mimari.png              # System architecture diagram
│   └── logo.png                # Project logo
│
├── .env.example                # Example environment variables
├── .dockerignore
├── pyproject.toml
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### 1. Clone the Repository

```bash
git clone https://github.com/fatihkadim/self-correcting-rag.git
cd self-correcting-rag
```

### 2. Virtual Environment and Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. Set Environment Variables

Copy `.env.example` to `.env` and configure your values:

```bash
cp .env.example .env
```

`.env` content:

```env
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
MAX_RETRY=2
QDRANT_URL=http://localhost:6333
```

> **Note:** `CONFIDENCE_THRESHOLD` (default: 0.7) and `TOP_K` (default: 5) are defined in `app/core/config.py`.

### 4. Start Qdrant (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 5. Index Documents

Add your PDF or TXT files under `data/raw` and run:

```bash
python scripts/ingest.py
```

### 6. Run the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Usage

### Web Interface (Recommended)
While the app is running, navigate to the Glassmorphism interface in your browser:
http://localhost:8000/

### API Health Check

```bash
curl http://localhost:8000/health
```

### API Send Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?"}'
```

### Upload Document (API)

```bash
curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
```

### List Uploaded Files (API)

```bash
curl http://localhost:8000/files
```

### Interactive API Documentation (Swagger UI)

http://localhost:8000/docs

---

## Core Concepts

### What is a Claim?
The minimal unit of information that can be independently evaluated as **true or false**.

| Valid Claim | Invalid Claim |
|---|---|
| "Python was developed in 1991." | "Python is a beautiful language." |
| "FastAPI supports async." | "This framework is useful." |

### Verification Statuses
| Status | Meaning |
|---|---|
| supported | Evidence found and verified |
| refuted | Evidence found and refuted |
| unknown | Insufficient evidence |

### Policy Decisions
| Decision | Condition | Action |
|---|---|---|
| ACCEPT | All claims supported | Returns answer as-is |
| REPAIR | Some claims supported | Rewrites using only verified claims |
| RETRY | No claims supported | Reformulates the query and retries |

### Retrieval Modes
| Mode | Usage | Retrieval top_k | Final top_k | Threshold |
|---|---|---|---|---|
| HIGH_RECALL | Initial answer generation | 25 | 10 | 0.25 |
| HIGH_PRECISION| Claim verification | 15 | 3 | 0.35 |

> **Note:** Retrieval is a two-stage process — fetches `Retrieval top_k` from Qdrant, then reranks with a CrossEncoder to `Final top_k`.

---

## Tests

Run all tests:

```bash
pytest tests/
```

---

## Evaluation (RAGAS)

The system is evaluated against Classical RAG using the [RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) framework on a 20-question benchmark dataset (`gpt-4o-mini`).

### 📊 Benchmark Results

#### Latest Benchmark Run (20 Questions)
| Metric | Classical RAG | Self-Correcting RAG | Improvement (Δ) | Status |
|---|:---:|:---:|:---:|:---:|
| **Faithfulness** | 0.8000 (80.0%) | **0.8750 (87.5%)** | **+9.38% (+0.075)** | 🟢 Improved |
| **Context Recall** | 0.7500 (75.0%) | **0.8000 (80.0%)** | **+6.67% (+0.050)** | 🟢 Improved |
| **Answer Relevancy** | 0.6224 (62.2%) | 0.6215 (62.2%) | -0.14% (-0.001) | ⚪ Equal |
| **Context Precision** | 0.6175 (61.8%) | 0.5909 (59.1%) | -4.31% (-0.027) | ⚪ Similar |

#### Multi-Run Stability Average (6 Benchmark Runs, 20 Questions)
| Metric | Classical RAG (Avg) | Self-Correcting RAG (Avg) | Difference (Δ) |
|---|:---:|:---:|:---:|
| **Faithfulness** | 0.7920 (79.2%) | **0.8462 (84.6%)** | **+6.84% (+0.054)** |
| **Context Precision** | 0.5830 (58.3%) | **0.6561 (65.6%)** | **+12.54% (+0.073)** |
| **Context Recall** | 0.7395 (74.0%) | **0.7417 (74.2%)** | **+0.30% (+0.002)** |
| **Answer Relevancy** | 0.5853 (58.5%) | **0.5917 (59.2%)** | **+1.09% (+0.006)** |

### 💡 Key Findings
- **Higher Faithfulness**: Claim-level extraction and NLI verification catch hallucinated or unsupported claims, improving factual faithfulness by **~7-9%**.
- **Enhanced Context Quality**: Two-stage retrieval (Qdrant semantic search + CrossEncoder reranker) combined with targeted claim verification searches achieves superior context precision and recall.
- **Answer Relevancy Preserved**: Answer repair maintains tight alignment with the user's original query while filtering out unfounded statements.

### Running Evaluations

```bash
# Run RAGAS metric evaluation
python -m app.evaluation.ragas_eval

# Run baseline comparison
python -m app.evaluation.baseline

# Generate synthetic evaluation dataset
python -m app.scripts.generate_dataset
```

### Metric Definitions
| Metric | Description |
|---|---|
| **Faithfulness** | Measures the factual consistency of the generated answer against the retrieved context. |
| **Answer Relevancy** | Measures how directly the generated answer addresses the user's prompt. |
| **Context Precision** | Measures whether relevant chunks are ranked higher in retrieval. |
| **Context Recall** | Measures whether the retrieved context contains all information needed to answer the question. |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Vector DB | Qdrant |
| Embedding | Sentence Transformers (all-MiniLM-L6-v2) |
| Reranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| LLM | OpenAI API (gpt-4o-mini) |
| PDF Processing| PyMuPDF |
| Container | Docker + Docker Compose |
| Evaluation | RAGAS + LangChain |
| Testing | pytest |

---

## License

MIT

---
---

<a id="self-correcting-rag-tr"></a>

**🇬🇧 [English](#self-correcting-rag) | 🇹🇷 [Türkçe](#self-correcting-rag-tr)**

# Self-Correcting RAG (TR)

> **Kanıtlanabilir ve savunulabilir cevaplar üreten, kendini düzelten bir RAG sistemi.**

Klasik RAG sistemleri cevap üretir ama doğrulamaz. Bu proje **ürettiği her cevabı iddialara ayırır**, **her iddiayı kanıtlarla doğrular** ve gerektiğinde **kendini düzeltir**.

---

## Proje Felsefesi

Bu proje "en hızlı cevap" yerine **kanıtlanabilir ve savunulabilir cevaplar** üretmeyi hedefler.

| Klasik RAG | Self-Correcting RAG |
|---|---|
| Cevabı üretir ve sunar | Cevabı üretir, iddialara ayırır, doğrular |
| LLM'e güvenir | LLM'i sorgular |
| Halüsinasyon kontrolü yok | İddia düzeyinde halüsinasyon tespiti |
| Tek geçişli (one-shot) | Yinelemeli, geri bildirim döngüsü |

---

## Sistem Mimarisi

Kullanıcının sorusuna ilk cevabı ürettikten sonra sistem şu adımları izler:
1. Cevap içindeki her bağımsız bilgi ifadesini (iddia) çıkarır.
2. Her iddia için ayrı, yüksek hassasiyetli bir veri araması yapar (Kanıt Kontrolü).
3. Kanıtları LLM Yargıcı ile karşılaştırır ve iddiaları desteklendi, çürütüldü veya bilinmiyor olarak etiketler.
4. Doğrulanmamış iddialar varsa, cevabı yalnızca doğrulanmış bilgilerle yeniden yazar (Cevap Onarımı).
5. Hiçbir iddia desteklenmiyorsa, sorguyu yeniden formüle eder ve tekrar dener (Sorgu Yeniden Yazma + Tekrar).

![Sistem Mimarisi](./assets/mimari.png)

---

## Kurulum

### Gereksinimler
- Python 3.11+
- Docker & Docker Compose

### 1. Depoyu Klonlayın

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

### 3. Ortam Değişkenlerini Ayarlayın

`.env.example` dosyasını `.env` olarak kopyalayın ve değerlerinizi girin:

```bash
cp .env.example .env
```

`.env` içeriği:

```env
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
MAX_RETRY=2
QDRANT_URL=http://localhost:6333
```

> **Not:** `CONFIDENCE_THRESHOLD` (varsayılan: 0.7) ve `TOP_K` (varsayılan: 5) `app/core/config.py` içinde tanımlanmıştır.

### 4. Qdrant'ı Başlatın (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 5. Dokümanları İndeksleyin

PDF veya TXT dosyalarınızı `data/raw` altına ekleyin ve çalıştırın:

```bash
python scripts/ingest.py
```

### 6. Uygulamayı Çalıştırın

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Kullanım

### Web Arayüzü (Önerilen)
Uygulama çalışırken tarayıcınızda aşağıdaki adrese gidin:
http://localhost:8000/

### API Sağlık Kontrolü

```bash
curl http://localhost:8000/health
```

### API Sorgu Gönderme

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Transformer mimarisi nedir?"}'
```

### Doküman Yükleme (API)

```bash
curl -X POST http://localhost:8000/upload -F "file=@dokuman.pdf"
```

### Yüklenen Dosyaları Listeleme (API)

```bash
curl http://localhost:8000/files
```

### İnteraktif API Dokümantasyonu (Swagger UI)

http://localhost:8000/docs

---

## Temel Kavramlar

### İddia (Claim) Nedir?
Bağımsız olarak **doğru veya yanlış** olarak değerlendirilebilen en küçük bilgi birimi.

| Geçerli İddia | Geçersiz İddia |
|---|---|
| "Python 1991'de geliştirildi." | "Python güzel bir dildir." |
| "FastAPI async destekler." | "Bu framework faydalıdır." |

### Doğrulama Durumları
| Durum | Anlamı |
|---|---|
| supported | Kanıt bulundu ve doğrulandı |
| refuted | Kanıt bulundu ve çürütüldü |
| unknown | Yetersiz kanıt |

### Politika Kararları
| Karar | Koşul | Eylem |
|---|---|---|
| ACCEPT | Tüm iddialar desteklendi | Cevabı olduğu gibi döndürür |
| REPAIR | Bazı iddialar desteklendi | Yalnızca doğrulanmış iddialarla yeniden yazar |
| RETRY | Hiçbir iddia desteklenmedi | Sorguyu yeniden formüle eder ve tekrar dener |

### Retrieval Modları
| Mod | Kullanım | Retrieval top_k | Final top_k | Eşik |
|---|---|---|---|---|
| HIGH_RECALL | İlk cevap üretimi | 25 | 10 | 0.25 |
| HIGH_PRECISION | İddia doğrulama | 15 | 3 | 0.35 |

> **Not:** Retrieval iki aşamalı bir süreçtir — Qdrant'tan `Retrieval top_k` kadar sonuç getirir, ardından CrossEncoder ile `Final top_k`'ya yeniden sıralar.

---

## Testler

Tüm testleri çalıştırın:

```bash
pytest tests/
```

---

## Değerlendirme (RAGAS)

Sistem, Klasik RAG ile karşılaştırmalı olarak [RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) çerçevesi kullanılarak 20 soruluk genişletilmiş test veri seti (`gpt-4o-mini`) üzerinde değerlendirilmektedir.

### 📊 Karşılaştırmalı Test Sonuçları

#### Son Değerlendirme Çalışması (20 Soru)
| Metrik | Klasik RAG | Self-Correcting RAG | İyileşme (Δ) | Durum |
|---|:---:|:---:|:---:|:---:|
| **Faithfulness (Sadakat)** | 0.8000 (%80.0) | **0.8750 (%87.5)** | **+%9.38 (+0.075)** | 🟢 İyileşme |
| **Context Recall (Kapsama)** | 0.7500 (%75.0) | **0.8000 (%80.0)** | **+%6.67 (+0.050)** | 🟢 İyileşme |
| **Answer Relevancy (Uygunluk)** | 0.6224 (%62.2) | 0.6215 (%62.2) | -%0.14 (-0.001) | ⚪ Eşit |
| **Context Precision (Hassasiyet)** | 0.6175 (%61.8) | 0.5909 (%59.1) | -%4.31 (-0.027) | ⚪ Benzer |

#### Çoklu Çalışma Kararlılık Ortalaması (6 Test Çalışması, 20 Soru)
| Metrik | Klasik RAG (Ort.) | Self-Correcting RAG (Ort.) | Fark (Δ) |
|---|:---:|:---:|:---:|
| **Faithfulness (Sadakat)** | 0.7920 (%79.2) | **0.8462 (%84.6)** | **+%6.84 (+0.054)** |
| **Context Precision (Hassasiyet)** | 0.5830 (%58.3) | **0.6561 (%65.6)** | **+%12.54 (+0.073)** |
| **Context Recall (Kapsama)** | 0.7395 (%74.0) | **0.7417 (%74.2)** | **+%0.30 (+0.002)** |
| **Answer Relevancy (Uygunluk)** | 0.5853 (%58.5) | **0.5917 (%59.2)** | **+%1.09 (+0.006)** |

### 💡 Öne Çıkan Bulgular
- **Yüksek Sadakat (Faithfulness)**: İddia düzeyinde ayrıştırma ve NLI doğrulama mekanizması sayesinde halüsinasyonlar ve kanıtsız ifadeler elenir, sadakat puanı **%7-9** oranında artar.
- **Gelişmiş Bağlam Kalitesi**: İki aşamalı getirme (Qdrant vektör arama + CrossEncoder yeniden sıralama) ve iddia bazlı yüksek hassasiyetli arama ile bağlam kalitesi yükseltilir.
- **Korunan Yanıt Uygunluğu**: Yanıt onarma (Answer Repair) adımı, uydurma iddiaları temizlerken kullanıcının sorusuna doğrudan odaklı kalmaya devam eder.

### Değerlendirmeleri Çalıştırma

```bash
# RAGAS metrik değerlendirmesini çalıştır
python -m app.evaluation.ragas_eval

# Klasik RAG vs Self-Correcting karşılaştırması
python -m app.evaluation.baseline

# Otomatik değerlendirme veri seti üretimi
python -m app.scripts.generate_dataset
```

### Metrik Tanımları
| Metrik | Açıklama |
|---|---|
| **Faithfulness** | Üretilen cevabın getirilen bağlama ne kadar sadık olduğunu (halüsinasyon kontrolü) ölçer. |
| **Answer Relevancy** | Cevabın kullanıcının sorusuna ne kadar doğrudan ve net cevap verdiğini ölçer. |
| **Context Precision** | İlgili parçaların getirme sırasında ne kadar üst sıralarda yer aldığını ölçer. |
| **Context Recall** | Getirilen bağlamın doğru cevabı üretmek için gereken tüm bilgileri kapsama oranını ölçer. |

---

## Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Veri Doğrulama | Pydantic v2 |
| Vektör Veritabanı | Qdrant |
| Embedding | Sentence Transformers (all-MiniLM-L6-v2) |
| Yeniden Sıralama | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| LLM | OpenAI API (gpt-4o-mini) |
| PDF İşleme | PyMuPDF |
| Container | Docker + Docker Compose |
| Değerlendirme | RAGAS + LangChain |
| Test | pytest |

---

## Lisans

MIT
