# Self-Correcting RAG

> **Kanitlanabilir ve savunulabilir cevaplar ureten, kendi kendini duzelten bir RAG sistemi.**

Klasik RAG sistemleri cevap uretir ama dogrulamaz. Bu proje urettigi her cevabi **iddialara (claims) boler**, her iddiayi **kanitlarla dogrular** ve gerektiginde **kendi kendini duzeltir**.

---

## Proje Felsefesi

Bu proje "en hizli cevap" degil, **kanitlanabilir ve savunulabilir cevap** uretmeyi hedefler.

| Klasik RAG | Self-Correcting RAG |
|---|---|
| Cevap uretir ve sunar | Cevap uretir, iddialara boler, dogrular |
| LLM'e guvenir | LLM'i sorgular |
| Hallucination kontrolu yok | Claim-level hallucination tespiti |
| Tek gecis (one-shot) | Iteratif, geri beslemeli dongu |

---

## Sistem Mimarisi

Sistem, kullanici sorusuna ilk cevabi urettikten sonra su adimlari izler:
1. Cevap icindeki her bir bagimsiz bilgi ifadesini (claim) cikarir.
2. Her bir iddia icin ayri ayri ve yuksek kesinlikli (High-Precision) veri aramasi yapar (Evidence Check).
3. LLM Judge ile kanitlari kiyaslayarak iddialari supported (desteklendi), refuted (curutuldu) veya unknown (bilinmiyor) olarak etiketler.
4. Dogrulanamayan iddialar varsa, sadece dogrulanmis bilgilerle cevabi yeniden yazar (Answer Repair).

---

## Klasor Yapisi

```
self-correcting-rag/
│
├── app/
│   ├── main.py                 # FastAPI entrypoint (Web UI ve API sunumu)
│   ├── pipeline.py             # Ana RAG pipeline sinifi
│   │
│   ├── api/
│   │   └── routes.py           # /query endpoint
│   │
│   ├── static/
│   │   └── index.html          # Web Arayuzu (Glassmorphism Ajan Arayuzu)
│   │
│   ├── core/
│   │   ├── config.py           # Env & model ayarlari (pydantic-settings)
│   │   ├── logger.py           # Merkezi logging altyapisi
│   │   └── settings.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py        # QdrantRetriever (High-Recall / High-Precision)
│   │   ├── chunking.py         # Recursive semantic chunking + overlap
│   │   ├── embedder.py         # Sentence Transformers embedding
│   │   └── query_rewrite.py    # Retry icin sorgu reformulasyonu
│   │
│   ├── generation/
│   │   ├── answer.py           # Ilk cevap uretim modulu
│   │   └── repair.py           # Dogrulanmis claim'lerden cevap yeniden yazimi
│   │
│   ├── claims/
│   │   ├── extractor.py        # LLM tabanli claim extraction
│   │   └── models.py           # Claim Pydantic modelleri
│   │
│   ├── verification/
│   │   ├── verifier.py         # Claim verification engine
│   │   └── judge.py            # LLM Judge / NLI degerlendirme
│   │
│   ├── agent/
│   │   ├── controller.py       # Self-correction karar mantigi (agentic core)
│   │   └── policies.py         # Retry / stop politikalari
│   │
│   ├── prompts/
│   │   ├── answer.txt          # Cevap uretim prompt'u
│   │   ├── claim_extract.txt   # Claim extraction prompt'u
│   │   ├── verify.txt          # Dogrulama prompt'u
│   │   └── repair.txt          # Cevap duzeltme prompt'u
│   │
│   ├── schemas/
│   │   ├── query.py            # QueryRequest / QueryResponse
│   │   ├── retrieval.py        # Chunk / RetrievalResult
│   │   ├── claims.py           # Claim / ClaimType
│   │   └── verification.py     # ClaimVerification / VerificationStatus
│   │
│   └── evaluation/
│       ├── metrics.py          # Ozel metrikler (hallucination rate, retry count)
│       └── baseline.py         # Klasik RAG karsilastirmasi
│
├── data/
│   ├── raw/                    # Kaynak dokumanlar (PDF ve TXT)
│   ├── processed/              # Chunk'lanmis veriler
│   └── eval/                   # Test sorulari & ground truth
│
├── scripts/
│   ├── ingest.py               # Dokuman yukleme & indexleme
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
│   ├── projeaciklamasi.md      # Detayli proje aciklamasi
│   └── sprint_plan.md          # 15 sprint'lik ogrenim plani
│
├── .env                        # Ortam degiskenleri
├── pyproject.toml
└── requirements.txt
```

---

## Kurulum

### Gereksinimler
- Python 3.11+
- Docker & Docker Compose

### 1. Repoyu Klonla

```bash
git clone https://github.com/fatihkadim/self-correcting-rag.git
cd self-correcting-rag
```

### 2. Sanal Ortam ve Bagimliliklar

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. Ortam Degiskenlerini Ayarla

`.env` dosyasi olustur:

```env
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
MAX_RETRY=3
CONFIDENCE_THRESHOLD=0.7
TOP_K=5
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
```

### 4. Qdrant'i Baslat (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 5. Dokumanlari Indexle

`data/raw` altina PDF veya TXT test dosyalarinizi ekledikten sonra:

```bash
python scripts/ingest.py
```

### 6. Uygulamayi Calistir

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Kullanim

### Web Arayuzu (Tavsiye Edilen)
Uygulama calisirken tarayicinizdan su adrese giderek premium Glassmorphism arayuzunu kullanabilirsiniz:
http://localhost:8000/

Arayuz uzerinden sorgularinizi gonderebilir, veri arama, iddia cikarimi, kanit kontrolu ve cevap onarma adimlarini canli izleyebilir ve onarilan cevaplari karsilastirabilirsiniz.

### API Saglik Kontrolu

```bash
curl http://localhost:8000/health
```

### API Sorgu Gonder

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Python ne zaman gelistirildi?"}'
```

### Interaktif API Dokumantasyonu (Swagger UI)

Uygulama calisirken tarayicida acin: http://localhost:8000/docs

---

## Temel Kavramlar

### Claim Nedir?
Tek basina **dogru veya yanlis** olarak degerlendirilebilen minimum bilgi birimi.

| Gecerli Claim | Gecerli Olmayan Claim |
|---|---|
| "Python 1991'de gelistirildi." | "Python guzel bir dildir." |
| "FastAPI async destekler." | "Bu framework kullanislidir." |

### Verification Statuleri
| Statu | Anlam |
|---|---|
| supported | Kanit bulundu ve dogrulandi |
| refuted | Kanit bulundu ve curutuldu |
| unknown | Yeterli kanit bulunamadi |

### Retrieval Modlari
| Mod | Kullanim | top_k | Threshold |
|---|---|---|---|
| HIGH_RECALL | Ilk cevap uretimi | 10 | 0.3 |
| HIGH_PRECISION | Claim dogrulama | 3 | 0.4 |

---

## Degerlendirme Metrikleri & Karsilastirma (Evaluation)

Self-Correcting RAG'in degerini kanitlamak icin klasik RAG ile karsilastirmali testler yapabilirsiniz. 

```bash
python -m app.evaluation.baseline
```

Script calistiginda asagidaki metrikleri karsilastirmali olarak sunar:
- Claim dogruluk orani (Supported)
- Uydurma orani (Refuted)
- Bilinmeyen orani (Unknown)
- Islenen toplam sorgu ve iddia sayisi

---

## Sprint Plani (15 Sprint · ~45 Saat)

| Grup | Sprintler | Konu |
|---|---|---|
| Temel | 1-4 | Kurulum, Config, Mock Retrieval, Pipeline |
| Retrieval | 5-7 | Qdrant, Chunking, Gercek Retrieval |
| Generation | 8-9 | LLM Entegrasyonu, Prompt Engineering |
| Claims | 10-11 | Claim Extraction, Schema |
| Verification | 12-13 | Verification Engine, Answer Repair |
| Final | 14-15 | Self-Correction Controller, Evaluation |

Detaylar icin -> docs/sprint_plan.md

---

## Teknoloji Yigini

| Katman | Teknoloji |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Veri Dogrulama | Pydantic v2 |
| Vector DB | Qdrant |
| Embedding | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI API (gpt-4o-mini) |
| Konteyner | Docker + Docker Compose |

---

## Lisans

MIT
