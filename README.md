
---

# 📄💬 10-K RAG Chat

A high-performance, Streamlit-based Retrieval-Augmented Generation (RAG) application specifically optimized for parsing, indexing, and chatting with complex financial documents (like 10-K Annual Reports). 

This project implements a multi-model architecture, allowing seamless switching between local models (Ollama) and cloud APIs (OpenAI, DeepSeek, Gemini, SiliconFlow), backed by parallel embedding processing and intelligent document routing.



## ✨ 主要功能 (Key Features)

* **🚀 极速文档处理 (Extreme Performance)**: Uses `PyMuPDF` for lightning-fast PDF parsing. Implements parallel vectorization using `ThreadPoolExecutor` to max out your GPU/CPU during FAISS index building.
* **🧠 多模型无缝切换 (Multi-Model Support)**: Plug-and-play support for local Ollama models and OpenAI-compatible APIs (DeepSeek, SiliconFlow, Custom) as well as Google Gemini. Mix and match your LLM and Embedding models.
* **🗂️ 智能分桶与路由 (Smart Bucketing & Routing)**: Automatically analyzes uploaded PDFs, extracts company names/topics, and groups them into "Buckets". The query router automatically directs user questions to the most relevant document bucket.
* **💾 本地缓存与会话管理 (Disk Cache & Persistent Sessions)**: 
  * FAISS vector stores are cached locally to disk, skipping the embedding phase for previously uploaded documents.
  * Chat history is serialized to a local JSON file, allowing you to switch between past conversations seamlessly.
* **🔍 精准引用体系 (Precise Citations)**: Includes a custom citation engine that traces the generated answer back to the exact chunk and page number in the original PDF, ensuring zero hallucination for financial data.
* **📊 实时硬件监控 (Hardware Telemetry)**: A floating UI panel monitors GPU Utilization (via `nvidia-smi`), VRAM, CPU, and System RAM in real-time during heavy embedding workloads.

---

## 🛠️ 安装指南 (Installation)

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/10k-rag-chat.git](https://github.com/yourusername/10k-rag-chat.git)
cd 10k-rag-chat

```

**2. Install dependencies:**
Make sure you have Python 3.10+ installed. Run the following command to install all required packages:

```bash
pip install -r requirements.txt

```

*(Note: If you are running on a Linux machine with an NVIDIA GPU and want FAISS to run on GPU, uninstall `faiss-cpu` and install `faiss-gpu` instead).*

**3. Configure Cache Directory (Important):**
By default, the vector cache and chat history are saved to `D:\Ollama\vector_cache` for Windows users.
If you are on Mac/Linux, or want to change this, open `app.py` and edit the `CACHE_ROOT` variable (around Line 20):

```python
CACHE_ROOT = "./vector_cache" # Or any path you prefer

```

---

## 🚀 启动与使用 (Startup & Usage)

### Optional: Enable Ollama Parallel Processing

To fully utilize the parallel embedding feature with local Ollama, you **must** enable parallel processing in your Ollama environment before starting the server. Open a terminal and run:

* **Windows (CMD):** `set OLLAMA_NUM_PARALLEL=8`
* **Linux/Mac:** `export OLLAMA_NUM_PARALLEL=8`
*(Then start your Ollama service: `ollama serve`)*

### Run the Application

Start the Streamlit web interface:

```bash
streamlit run app.py

```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Usage Guide

1. **Settings:** Open the left sidebar. Configure your Ollama host URL or input your API keys for OpenAI/DeepSeek/Gemini. Select your desired LLM and Embedding model.
2. **Upload:** Drop your 10-K PDFs into the uploader. The system will process them once, split them intelligently, and build/cache the FAISS indexes in parallel.
3. **Tune:** Use the sliders in the sidebar to adjust chunk sizes, MMR parameters, and parallel worker counts to match your hardware capabilities.
4. **Chat:** Ask specific financial questions. The system will automatically route the question, retrieve the context, and generate an answer with exact page citations.

---

## 📄 License

MIT License
