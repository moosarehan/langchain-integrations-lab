# 🦜 LangChain Models

A collection of LangChain integration examples demonstrating how to use various **Chat Models**, **LLMs**, and **Embedding Models** from Google Gemini, Hugging Face (API & Local) using Python.

---

## 📁 Project Structure

```
langchain-models/
│
├── .env                        # API Keys (GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN)
│
├── ChatModels/
│   ├── chat-gemini.py          # Google Gemini 3.6 Flash via ChatGoogleGenerativeAI
│   ├── chat-huggingface.py     # Qwen3-235B-A22B via HuggingFace Endpoint API
│   └── huggingface-local.py    # Qwen (local) via HuggingFacePipeline
│
├── Embeddings/
│   ├── hf-api-embedding.py     # Single query embedding via HuggingFace API
│   ├── hf-api-docs.py          # Batch document embedding via HuggingFace API
│   ├── hf-local-embed.py       # Local sentence-transformers embedding
│   └── document-similar.py     # Cosine similarity search across documents
│
└── LLM/
    └── llm-demo.py             # Google Gemini 3.6 Flash via GoogleGenerativeAI (LLM mode)
```

---

## 🤖 Models Used

### 🔵 Google Gemini

| Model | Version | Type | Used In |
|-------|---------|------|---------|
| **Gemini 3.6 Flash** | `gemini-3.6-flash` | Chat Model | `ChatModels/chat-gemini.py` |
| **Gemini 3.6 Flash** | `gemini-3.6-flash` | LLM (text completion) | `LLM/llm-demo.py` |

> Accessed via [`langchain-google-genai`](https://pypi.org/project/langchain-google-genai/) using the `GOOGLE_API_KEY` environment variable.

---

### 🟠 Hugging Face — API (Serverless Inference)

| Model | Repo ID | Type | Used In |
|-------|---------|------|---------|
| **Qwen3-235B-A22B** | `Qwen/Qwen3.8-2.4T-A95B` | Chat / Text Generation | `ChatModels/chat-huggingface.py` |
| **all-MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | Text Embedding (384-dim) | `Embeddings/hf-api-embedding.py` |
| **all-MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | Batch Doc Embedding | `Embeddings/hf-api-docs.py` |
| **all-MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | Document Similarity | `Embeddings/document-similar.py` |

> Accessed via `HuggingFaceEndpoint` and `HuggingFaceEndpointEmbeddings` using the `HUGGINGFACEHUB_API_TOKEN` environment variable.

---

### 🟡 Hugging Face — Local (On-device)

| Model | Repo ID | Type | Used In |
|-------|---------|------|---------|
| **Qwen (local)** | `Qwen/Qwen3.8-2.4T-A95B` | Local Pipeline Text Generation | `ChatModels/huggingface-local.py` |
| **all-MiniLM-L6-v2** | `sentence-transformers/all-MiniLM-L6-v2` | Local Embedding (384-dim) | `Embeddings/hf-local-embed.py` |

> Loaded directly onto the machine via `HuggingFacePipeline` and `HuggingFaceEmbeddings` from `langchain-huggingface`. Weights are downloaded and cached locally using the `HF_HOME` environment variable.

---

## 📂 Script Details

### 💬 ChatModels

#### `chat-gemini.py`
- **Model**: Google Gemini 3.6 Flash (`gemini-3.6-flash`)
- **Interface**: `ChatGoogleGenerativeAI`
- **Task**: Explains SOLID design principles
- **Auth**: `GOOGLE_API_KEY` via `.env`

```python
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
response = model.invoke("Explain solid principles each principle in 2 lines")
```

---

#### `chat-huggingface.py`
- **Model**: Qwen3 via Hugging Face Serverless Inference API
- **Interface**: `HuggingFaceEndpoint` + `ChatHuggingFace`
- **Task**: Describes the beauty of Faisalabad city
- **Auth**: `HUGGINGFACEHUB_API_TOKEN` via `.env`

```python
llm = HuggingFaceEndpoint(repo_id='Qwen/Qwen3.8-2.4T-A95B', task='text-generation')
model = ChatHuggingFace(llm=llm)
result = model.invoke("explain the beauty of Faisalabad")
```

---

#### `huggingface-local.py`
- **Model**: Qwen loaded locally via `HuggingFacePipeline`
- **Interface**: `HuggingFacePipeline` + `ChatHuggingFace`
- **Task**: Local text generation without API calls
- **Cache**: `E:/huggingface_cache` (configurable via `HF_HOME`)

```python
llm = HuggingFacePipeline.from_model_id(
    model_id='Qwen/Qwen3.8-2.4T-A95B',
    task='text-generation',
    pipeline_kwargs=dict(temperature=0.8)
)
```

---

### 🔢 Embeddings

#### `hf-api-embedding.py`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Interface**: `HuggingFaceEndpointEmbeddings`
- **Task**: Embed a single query string into a **384-dimensional** vector
- **Auth**: `HUGGINGFACEHUB_API_TOKEN` via `.env`

```python
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
result = embeddings.embed_query("Faisalabad is known for its textile industry...")
# Output: list of 384 float values
```

---

#### `hf-api-docs.py`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Interface**: `HuggingFaceEndpointEmbeddings`
- **Task**: Embed a **batch of documents** into 384-dim vectors via the HuggingFace API

```python
docs = ["Doc 1...", "Doc 2...", "Doc 3..."]
result = embeddings.embed_documents(docs)
# Output: list of 3 vectors, each 384-dimensional
```

---

#### `hf-local-embed.py`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (downloaded locally)
- **Interface**: `HuggingFaceEmbeddings`
- **Task**: Embed a query fully **offline** without any API calls, weights cached locally

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embeddings.embed_query("Faisalabad is known for its textile industry...")
# Output: list of 384 float values (no internet required after first download)
```

---

#### `document-similar.py`
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Interface**: `HuggingFaceEndpointEmbeddings` + `sklearn cosine_similarity`
- **Task**: Embed a set of documents and a query, then rank documents by **semantic similarity**

```python
docs = ["Faisalabad...", "Breaking bad...", "Cricket player...", "Erling Haaland...", "AI applications..."]
query = "tell me about erling halaand"

doc_vectors = embeddings.embed_documents(docs)
query_vector = embeddings.embed_query(query)
scores = cosine_similarity([query_vector], doc_vectors)[0]
# Output: [0.10966223 0.01633256 0.03171901 0.34299423 0.10264589]
# → Haaland document (index 3) is most similar ✅
```

---

### 🧠 LLM

#### `llm-demo.py`
- **Model**: Google Gemini 3.6 Flash (`gemini-3.6-flash`)
- **Interface**: `GoogleGenerativeAI` (raw LLM, not chat)
- **Task**: Plain text completion — explains linear regression in 2 sentences
- **Auth**: `GOOGLE_API_KEY` via `.env`

```python
llm = GoogleGenerativeAI(model="gemini-3.6-flash")
response = llm.invoke("Explain linear regression in 2 sentences")
```

---

## ⚙️ Setup

### 1. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
A `.env.example` file is included as a template. Copy it and fill in your real keys:
```bash
cp langchain-models/.env.example langchain-models/.env
```
Or manually create a `.env` file inside the `langchain-models/` folder:
```env
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> ⚠️ **Never commit your real `.env` file.** It is already listed in `.gitignore`. Only `.env.example` (with placeholder values) should be committed.

### 4. Run any script
```bash
# From the langchain-models/ directory:
python ChatModels/chat-gemini.py
python ChatModels/chat-huggingface.py
python Embeddings/hf-api-embedding.py
python Embeddings/hf-local-embed.py
python Embeddings/document-similar.py
python LLM/llm-demo.py
```

---

## 🔑 API Keys Required

| Service | Variable | Where to Get |
|---------|----------|-------------|
| Google Gemini | `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| Hugging Face | `HUGGINGFACEHUB_API_TOKEN` | [HuggingFace Settings](https://huggingface.co/settings/tokens) |

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LangChain framework |
| `langchain-google-genai` | Gemini model integrations |
| `langchain-huggingface` | Hugging Face integrations |
| `transformers` | Underlying HuggingFace model loading |
| `sentence-transformers` | Local embedding models |
| `torch` | PyTorch backend for local models |
| `scikit-learn` | Cosine similarity computation |
| `python-dotenv` | `.env` file loading |
