import os, time, hashlib, tempfile, re
from datetime import datetime
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

st.set_page_config(page_title="10-K RAG Chat", layout="wide")
st.title("📄💬 10-K RAG Chat (Ollama + Multi-API + Disk Cache + Parallel Embeddings)")
perf_panel_placeholder = st.empty()

CACHE_ROOT = r"D:\Ollama\vector_cache"

DEFAULT_PERSONA = 'You are a careful assistant. Use ONLY the provided context. If the answer is not supported by the context, say: "I don\'t have enough information to answer this question." Always cite page numbers and quote short phrases as evidence. Do not guess.'
PROMPT_GENERAL = ChatPromptTemplate.from_template(
    """{persona}

Chat History:
<history>
{chat_history}
</history>

Use ONLY the context below to answer. If the answer is not in the context, refuse.

<context>
{context}
</context>

Question: {question}
Answer:"""
)
PROMPT_NUMERIC = ChatPromptTemplate.from_template(
    """{persona}

Chat History:
<history>
{chat_history}
</history>

You must answer using ONLY the context. If the exact value cannot be found, say: "I don't have enough information to answer this question."
Rules:
- Extract the exact number(s) as written (do NOT change units).
- Include the original supporting quote(s).
- Cite page numbers for every number.
- If multiple candidates exist, list them and explain why.

<context>
{context}
</context>

Question: {question}
Answer (with bullets):"""
)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def fp_uploads(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8"))
        h.update(f.getvalue())
    return h.hexdigest()

def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9&\-]{2,}", (text or "").lower())

def infer_bucket_label(file_name: str, raw_docs) -> str:
    base = os.path.splitext(os.path.basename(file_name))[0]
    base_tokens = [t for t in tokenize(base) if t not in {"10k", "annual", "report", "form", "inc", "corp", "ltd", "plc"}]
    if base_tokens:
        return base_tokens[0].capitalize()

    probe = "\n".join(d.page_content[:800] for d in raw_docs[:2])
    upper_phrases = re.findall(r"\b([A-Z][A-Z&\.]{2,}(?:\s+[A-Z][A-Z&\.]{2,}){0,2})\b", probe)
    if upper_phrases:
        cleaned = [p.replace(".", "").strip() for p in upper_phrases if p.strip()]
        if cleaned:
            return Counter(cleaned).most_common(1)[0][0][:40]
    return "Document"

def build_bucket_profiles(docs_by_bucket: dict) -> dict:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "have", "has", "had",
        "our", "their", "its", "not", "but", "into", "about", "through", "during", "year", "years", "page",
        "report", "form", "company", "inc", "corp", "ltd", "plc", "all"
    }
    profiles = {}
    for label, docs in docs_by_bucket.items():
        if label == "All" or not docs:
            continue
        sample = "\n".join(d.page_content[:1200] for d in docs[:6])
        words = [w for w in tokenize(sample) if w not in stop]
        common = [w for w, _ in Counter(words).most_common(25)]
        profiles[label] = set(common + tokenize(label))
    return profiles

def route_bucket(question: str, labels: list[str], profiles: dict) -> str:
    if not labels:
        return "All"
    candidates = [x for x in labels if x != "All"]
    if not candidates:
        return "All"

    q_tokens = set(tokenize(question))
    if not q_tokens:
        return "All"

    best_label = "All"
    best_score = 0
    for label in candidates:
        label_tokens = set(tokenize(label))
        profile_tokens = profiles.get(label, set())
        score = len(q_tokens & label_tokens) * 3 + len(q_tokens & profile_tokens)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label if best_score > 0 else "All"

def is_numeric_question(q: str) -> bool:
    ql = q.lower()
    keys = [
        "cash", "liquidity", "cash equivalents", "cash and cash equivalents", "total cash",
        "amount", "how much", "$", "million", "billion", "balance sheet", "current assets",
        "current liabilities", "working capital", "net cash", "free cash flow", "fcf"
    ]
    if any(k in ql for k in keys):
        return True
    if re.search(r"\b(20\d{2})\b", ql) and re.search(r"\b(increase|decrease|compared|vs\.?|versus|change)\b", ql):
        return True
    return False

def format_history(msgs, limit=20):
    ms = msgs[-limit:]
    out = []
    for m in ms:
        out.append(("User" if m["role"] == "user" else "Assistant") + ": " + m["content"])
    return "\n\n".join(out)

def format_docs(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "unknown")
        blocks.append(f"[Chunk {i} | Page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

def excerpt(text: str, start: int, size: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return ""
    left = max(start - size // 2, 0)
    right = min(left + size, len(t))
    s = t[left:right].strip()
    return s + ("..." if right < len(t) else "")

def build_answer_citations(answer: str, docs, max_items: int = 5) -> list[dict]:
    rows = []
    seen = set()

    quotes = re.findall(r'"([^"\n]{6,140})"|“([^”\n]{6,140})”', answer or "")
    quote_texts = []
    for a, b in quotes:
        q = (a or b or "").strip()
        if q and q.lower() not in seen:
            quote_texts.append(q)
            seen.add(q.lower())

    for q in quote_texts:
        found = False
        for i, d in enumerate(docs or [], 1):
            txt = d.page_content or ""
            idx = txt.lower().find(q.lower())
            if idx >= 0:
                rows.append({
                    "Quote": q[:80],
                    "Page": d.metadata.get("page", "unknown"),
                    "Chunk": i,
                    "Snippet": excerpt(txt, idx),
                })
                found = True
                break
        if found and len(rows) >= max_items:
            return rows

    if rows:
        return rows[:max_items]

    for i, d in enumerate(docs or [], 1):
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        rows.append({
            "Quote": "(retrieved context)",
            "Page": d.metadata.get("page", "unknown"),
            "Chunk": i,
            "Snippet": excerpt(txt, 0),
        })
        if len(rows) >= max_items:
            break
    return rows

def capture_docs(label):
    def _cap(docs):
        if "last_retrieved" not in st.session_state:
            st.session_state.last_retrieved = {"All": []}
        st.session_state.last_retrieved[label] = docs
        return docs
    return _cap

def process_all_pdfs_once(files, chunk_size, chunk_overlap):
    buckets = {"All": []}
    
    split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            p = os.path.join(td, f.name)
            
            with open(p, "wb") as w:
                w.write(f.getbuffer())
 
            raw_docs = PyMuPDFLoader(p).load()
            chunks = split.split_documents(raw_docs)
            label = infer_bucket_label(f.name, raw_docs)
            label = label if label not in buckets else f"{label}_{len(buckets)}"

            for c in chunks:
                c.metadata["bucket"] = label

            buckets["All"].extend(chunks)
            buckets[label] = chunks
            
    return buckets

def build_indexes():
    if not uploaded:
        return
    index_key = current_store_key()
    if (not rebuild) and st.session_state.stores_key == index_key and st.session_state.stores:
        return

    os.makedirs(CACHE_ROOT, exist_ok=True)
    em = build_embeddings(emb_provider, emb_model, keys)

    with st.spinner("🚀 极速解析 PDFs 并切片中 (每个文件仅处理一次)..."):
        # 调用新函数，直接拿到分类好的 docs
        docs_by_company = process_all_pdfs_once(uploaded, chunk_size, chunk_overlap)
    bucket_labels = ["All"] + [k for k in docs_by_company.keys() if k != "All"]
    bucket_profiles = build_bucket_profiles(docs_by_company)

    stores = {}
    update_perf_panel("Preparing indexes ...", 0)

    for bucket in bucket_labels:
        if docs_by_company.get(bucket):
            stores[bucket] = build_faiss_store_with_parallel(docs_by_company[bucket], em, index_key, bucket, update_perf_panel)

    st.session_state.stores = stores
    st.session_state.stores_key = index_key
    st.session_state.bucket_labels = bucket_labels
    st.session_state.bucket_profiles = bucket_profiles
    st.session_state.bucket_stats = {k: len(v) for k, v in docs_by_company.items()}
    st.session_state.last_retrieved = {k: [] for k in bucket_labels}
    update_perf_panel("Ready", 100)

def env_or_input(env_key: str, fallback: str) -> str:
    v = (os.getenv(env_key, "") or "").strip()
    return v if v else (fallback or "").strip()

def norm_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")

def openai_compat_models(base_url: str, api_key: str) -> list[str]:
    if not base_url or not api_key:
        return []
    u = norm_base_url(base_url)
    if not u.endswith("/v1"):
        u = u + "/v1"
    url = u + "/models"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = []
        for it in items:
            mid = it.get("id")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
        return sorted(list(dict.fromkeys(ids)))
    except Exception:
        return []

def ollama_models(ollama_host: str) -> list[str]:
    host = (ollama_host or "").strip().rstrip("/")
    if not host:
        return []
    url = host + "/api/tags"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return []
        data = r.json()
        models = data.get("models", [])
        out = []
        for m in models:
            name = m.get("name")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
        return sorted(list(dict.fromkeys(out)))
    except Exception:
        return []

def gemini_models(api_key: str) -> dict[str, list[str]]:
    if not api_key:
        return {"chat": [], "embed": []}
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        chat, embed = [], []
        for m in genai.list_models():
            name = getattr(m, "name", None)
            methods = getattr(m, "supported_generation_methods", []) or []
            if isinstance(name, str) and name.strip():
                if "generateContent" in methods:
                    chat.append(name.strip())
                if "embedContent" in methods:
                    embed.append(name.strip())
        chat = sorted(list(dict.fromkeys(chat)))
        embed = sorted(list(dict.fromkeys(embed)))
        return {"chat": chat, "embed": embed}
    except Exception:
        return {"chat": [], "embed": []}

def build_embeddings(provider: str, model: str, keys: dict):
    p = provider.lower()
    if p == "ollama":
        # 优化 2 应用：使用官方包 langchain_ollama 以支持原生批量接口
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        k = keys.get("OPENAI_API_KEY", "")
        if not k:
            raise ValueError("OpenAI key required for OpenAI embeddings.")
        return OpenAIEmbeddings(model=model, api_key=k, base_url=keys.get("OPENAI_BASE_URL") or None)
    if p == "deepseek":
        from langchain_openai import OpenAIEmbeddings
        k = keys.get("DEEPSEEK_API_KEY", "")
        if not k:
            raise ValueError("DeepSeek key required for DeepSeek embeddings.")
        return OpenAIEmbeddings(model=model, api_key=k, base_url=keys.get("DEEPSEEK_BASE_URL"))
    if p == "siliconflow":
        from langchain_openai import OpenAIEmbeddings
        k = keys.get("SILICONFLOW_API_KEY", "")
        if not k:
            raise ValueError("SiliconFlow key required for SiliconFlow embeddings.")
        return OpenAIEmbeddings(model=model, api_key=k, base_url=keys.get("SILICONFLOW_BASE_URL"))
    if p == "custom":
        from langchain_openai import OpenAIEmbeddings
        k = keys.get("CUSTOM_API_KEY", "")
        if not k:
            raise ValueError("Custom key required for Custom embeddings.")
        return OpenAIEmbeddings(model=model, api_key=k, base_url=keys.get("CUSTOM_BASE_URL"))
    if p == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        k = keys.get("GOOGLE_API_KEY", "")
        if not k:
            raise ValueError("Google key required for Gemini embeddings.")
        os.environ["GOOGLE_API_KEY"] = k
        return GoogleGenerativeAIEmbeddings(model=model)
    raise ValueError("Unknown embeddings provider.")

def build_llm(provider: str, model: str, temp: float, keys: dict):
    p = provider.lower()
    if p == "ollama":
        # 优化 2 应用：使用官方包 langchain_ollama
        from langchain_ollama import ChatOllama
        # 优化 3 应用：强制注入 num_ctx=8192 防止 RAG 检索上下文超限
        return ChatOllama(model=model, temperature=temp, num_ctx=8192)
    if p == "openai":
        from langchain_openai import ChatOpenAI
        k = keys.get("OPENAI_API_KEY", "")
        if not k:
            raise ValueError("OpenAI key required for OpenAI LLM.")
        return ChatOpenAI(model=model, temperature=temp, api_key=k, base_url=keys.get("OPENAI_BASE_URL") or None)
    if p == "deepseek":
        from langchain_openai import ChatOpenAI
        k = keys.get("DEEPSEEK_API_KEY", "")
        if not k:
            raise ValueError("DeepSeek key required for DeepSeek LLM.")
        return ChatOpenAI(model=model, temperature=temp, api_key=k, base_url=keys.get("DEEPSEEK_BASE_URL"))
    if p == "siliconflow":
        from langchain_openai import ChatOpenAI
        k = keys.get("SILICONFLOW_API_KEY", "")
        if not k:
            raise ValueError("SiliconFlow key required for SiliconFlow LLM.")
        return ChatOpenAI(model=model, temperature=temp, api_key=k, base_url=keys.get("SILICONFLOW_BASE_URL"))
    if p == "custom":
        from langchain_openai import ChatOpenAI
        k = keys.get("CUSTOM_API_KEY", "")
        if not k:
            raise ValueError("Custom key required for Custom LLM.")
        return ChatOpenAI(model=model, temperature=temp, api_key=k, base_url=keys.get("CUSTOM_BASE_URL"))
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        k = keys.get("GOOGLE_API_KEY", "")
        if not k:
            raise ValueError("Google key required for Gemini LLM.")
        os.environ["GOOGLE_API_KEY"] = k
        return ChatGoogleGenerativeAI(model=model, temperature=temp)
    raise ValueError("Unknown LLM provider.")

def gpu_stats_text():
    try:
        import subprocess
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        if not out:
            return "GPU: n/a"
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 5:
            return "GPU: n/a"
        util, mem_used, mem_total, power, temp = parts[:5]
        return f"GPU util {util}% · VRAM {mem_used}/{mem_total} MiB · {power} W · {temp}°C"
    except Exception:
        return "GPU: n/a"

def sys_stats_text():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024**3)
        total_gb = vm.total / (1024**3)
        return f"CPU {cpu:.0f}% · RAM {used_gb:.1f}/{total_gb:.1f} GB"
    except Exception:
        return "CPU/RAM: n/a"

def update_perf_panel(status: str, progress: int):
        p = min(max(int(progress), 0), 100)
        perf_panel_placeholder.markdown(
                f"""
                <div style="position: fixed; top: 0.8rem; right: 1rem; z-index: 9999; width: 360px; background: rgba(20,22,28,0.92); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 10px 12px; color: #f1f5f9; box-shadow: 0 8px 24px rgba(0,0,0,0.35);">
                    <div style="font-weight: 700; margin-bottom: 6px;">Performance</div>
                    <div style="font-size: 12px; opacity: 0.95;">{status}</div>
                    <div style="height: 6px; margin: 8px 0; border-radius: 999px; background: rgba(255,255,255,0.18); overflow: hidden;">
                        <div style="height: 100%; width: {p}%; background: #3b82f6;"></div>
                    </div>
                    <div style="font-size: 12px; opacity: 0.9;">{gpu_stats_text()}</div>
                    <div style="font-size: 12px; opacity: 0.9; margin-top: 2px;">{sys_stats_text()}</div>
                </div>
                """,
                unsafe_allow_html=True,
        )

def embed_parallel(texts, embeddings, batch_size: int, max_workers: int, progress_cb):
    n = len(texts)
    if n == 0:
        return []
    results = [None] * n

    batches = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batches.append((start, end, texts[start:end]))

    done = 0
    total_batches = len(batches)

    def worker(item):
        start, end, batch = item
        vecs = embeddings.embed_documents(batch)
        return start, end, vecs

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, b) for b in batches]
        for fut in as_completed(futs):
            start, end, vecs = fut.result()
            for i, v in enumerate(vecs):
                results[start + i] = v
            done += 1
            if progress_cb:
                progress_cb(done, total_batches)

    if any(v is None for v in results):
        raise RuntimeError("Embedding parallel returned missing vectors.")
    return results

def cache_dir_for(index_key: str, company: str) -> str:
    return os.path.join(CACHE_ROOT, index_key, company)

def vectorstore_load_if_exists(index_key: str, company: str, embeddings):
    path = cache_dir_for(index_key, company)
    if os.path.isdir(path):
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            return None
    return None

def vectorstore_save(vs, index_key: str, company: str):
    path = cache_dir_for(index_key, company)
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)

@st.cache_data(ttl=30)
def cached_ollama_models(host: str):
    return ollama_models(host)

@st.cache_data(ttl=30)
def cached_openai_compat_models(base_url: str, api_key: str):
    return openai_compat_models(base_url, api_key)

@st.cache_data(ttl=60)
def cached_gemini_models(api_key: str):
    return gemini_models(api_key)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "stores" not in st.session_state:
    st.session_state.stores = {}
if "stores_key" not in st.session_state:
    st.session_state.stores_key = None
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = {"All": []}
if "bucket_labels" not in st.session_state:
    st.session_state.bucket_labels = ["All"]
if "bucket_profiles" not in st.session_state:
    st.session_state.bucket_profiles = {}
if "bucket_stats" not in st.session_state:
    st.session_state.bucket_stats = {"All": 0}
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "last_answer_bucket" not in st.session_state:
    st.session_state.last_answer_bucket = "All"
if "citation_history" not in st.session_state:
    st.session_state.citation_history = []

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Local Ollama")
    ollama_host = st.text_input("Ollama host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_list = cached_ollama_models(ollama_host)
    st.caption(f"Ollama models detected: {len(ollama_list)}")

    st.divider()
    st.subheader("API Providers (OpenAI-compatible)")

    enable_openai = st.checkbox("Enable OpenAI", value=bool(os.getenv("OPENAI_API_KEY", "").strip()))
    openai_base = st.text_input("OpenAI base_url (optional)", value=os.getenv("OPENAI_BASE_URL", ""))
    openai_key_in = st.text_input("OPENAI_API_KEY (fallback)", type="password", value="")
    openai_key = env_or_input("OPENAI_API_KEY", openai_key_in) if enable_openai else ""

    enable_deepseek = st.checkbox("Enable DeepSeek", value=bool(os.getenv("DEEPSEEK_API_KEY", "").strip()))
    deepseek_base = st.text_input("DeepSeek base_url", value=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    deepseek_key_in = st.text_input("DEEPSEEK_API_KEY (fallback)", type="password", value="")
    deepseek_key = env_or_input("DEEPSEEK_API_KEY", deepseek_key_in) if enable_deepseek else ""

    enable_siliconflow = st.checkbox("Enable SiliconFlow", value=bool(os.getenv("SILICONFLOW_API_KEY", "").strip()))
    siliconflow_base = st.text_input("SiliconFlow base_url", value=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    siliconflow_key_in = st.text_input("SILICONFLOW_API_KEY (fallback)", type="password", value="")
    siliconflow_key = env_or_input("SILICONFLOW_API_KEY", siliconflow_key_in) if enable_siliconflow else ""

    enable_custom = st.checkbox("Enable Custom OpenAI-compatible", value=False)
    custom_base = st.text_input("Custom base_url", value=os.getenv("CUSTOM_BASE_URL", ""))
    custom_key_in = st.text_input("CUSTOM_API_KEY (fallback)", type="password", value="")
    custom_key = env_or_input("CUSTOM_API_KEY", custom_key_in) if enable_custom else ""

    st.divider()
    st.subheader("Gemini (optional)")
    enable_gemini = st.checkbox("Enable Gemini", value=bool(os.getenv("GOOGLE_API_KEY", "").strip()))
    google_key_in = st.text_input("GOOGLE_API_KEY (fallback)", type="password", value="")
    google_key = env_or_input("GOOGLE_API_KEY", google_key_in) if enable_gemini else ""

    keys = {
        "OPENAI_API_KEY": openai_key,
        "OPENAI_BASE_URL": norm_base_url(openai_base) or None,
        "DEEPSEEK_API_KEY": deepseek_key,
        "DEEPSEEK_BASE_URL": norm_base_url(deepseek_base),
        "SILICONFLOW_API_KEY": siliconflow_key,
        "SILICONFLOW_BASE_URL": norm_base_url(siliconflow_base),
        "CUSTOM_API_KEY": custom_key,
        "CUSTOM_BASE_URL": norm_base_url(custom_base),
        "GOOGLE_API_KEY": google_key,
    }

    st.divider()
    st.subheader("Chunking / Retrieval")
    chunk_size = st.slider("Chunk size", 400, 2200, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 10)
    k = st.slider("Top-k", 1, 12, 6, 1)
    fetch_k = st.slider("MMR fetch_k", 5, 80, 30, 1)
    lambda_mult = st.slider("MMR lambda_mult", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    st.subheader("Embedding Speed")
    batch_size = st.slider("Embed batch size", 8, 256, 96, 8)
    max_workers = st.slider("Parallel workers", 1, 16, 6, 1)

    st.divider()
    st.subheader("Routing / Behavior")
    route_options = ["Auto"] + st.session_state.bucket_labels
    route_mode = st.selectbox("Route", route_options, index=0)
    strict_refusal = st.checkbox("Strict refusal when weak context", value=True)
    min_context_chars = st.slider("Min context chars (strict)", 0, 5000, 600, 50)
    persona = st.text_area("Persona", value=DEFAULT_PERSONA, height=120)

    st.divider()
    rebuild = st.button("🔄 Force rebuild indexes")
    clear_chat = st.button("🧹 Clear chat")


providers = []
if ollama_list:
    providers.append("Ollama")
if enable_openai and openai_key:
    providers.append("OpenAI")
if enable_deepseek and deepseek_key:
    providers.append("DeepSeek")
if enable_siliconflow and siliconflow_key:
    providers.append("SiliconFlow")
if enable_custom and custom_key and custom_base.strip():
    providers.append("Custom")
if enable_gemini and google_key:
    providers.append("Gemini")

def provider_models(provider_name: str) -> dict:
    p = provider_name.lower()
    if p == "ollama":
        ms = cached_ollama_models(ollama_host)
        return {"chat": ms, "embed": ms}
    if p == "openai":
        base = keys.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ms = cached_openai_compat_models(base, openai_key)
        em = [m for m in ms if "embed" in m.lower()]
        return {"chat": ms, "embed": em or ms}
    if p == "deepseek":
        ms = cached_openai_compat_models(keys.get("DEEPSEEK_BASE_URL"), deepseek_key)
        em = [m for m in ms if "embed" in m.lower()]
        return {"chat": ms, "embed": em or ms}
    if p == "siliconflow":
        ms = cached_openai_compat_models(keys.get("SILICONFLOW_BASE_URL"), siliconflow_key)
        em = [m for m in ms if "embed" in m.lower()]
        return {"chat": ms, "embed": em or ms}
    if p == "custom":
        ms = cached_openai_compat_models(keys.get("CUSTOM_BASE_URL"), custom_key)
        em = [m for m in ms if "embed" in m.lower()]
        return {"chat": ms, "embed": em or ms}
    if p == "gemini":
        ms = cached_gemini_models(google_key)
        return {"chat": ms["chat"], "embed": ms["embed"]}
    return {"chat": [], "embed": []}

with st.sidebar:
    st.divider()
    st.subheader("Model Pickers")
    if not providers:
        st.warning("No providers available. Enable Ollama or enter valid API keys.")
        llm_provider = "Ollama"
        emb_provider = "Ollama"
        llm_models = ollama_list
        emb_models = ollama_list
        llm_model = st.selectbox("LLM Model", llm_models if llm_models else ["(no models)"], index=0)
        emb_model = st.selectbox("Embeddings Model", emb_models if emb_models else ["(no models)"], index=0)
    else:
        llm_provider = st.selectbox("LLM Provider", providers, index=0)
        llm_models = provider_models(llm_provider)["chat"]
        llm_model = st.selectbox("LLM Model", llm_models if llm_models else ["(no models)"], index=0)

        emb_provider = st.selectbox("Embeddings Provider", providers, index=0)
        emb_models = provider_models(emb_provider)["embed"]
        emb_model = st.selectbox("Embeddings Model", emb_models if emb_models else ["(no models)"], index=0)

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)


uploaded = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if clear_chat:
    st.session_state.messages = []
    st.session_state.last_retrieved = {k: [] for k in st.session_state.bucket_labels}

def current_store_key():
    if not uploaded:
        return None
    pdf_fp = fp_uploads(uploaded)
    sig = f"{pdf_fp}::{emb_provider.lower()}::{emb_model}::cs={chunk_size};co={chunk_overlap}"
    return sha256_str(sig)

def build_faiss_store_with_parallel(docs, embeddings, index_key, company, ui_update):
    loaded = vectorstore_load_if_exists(index_key, company, embeddings)
    if loaded is not None:
        ui_update(f"{company}: loaded from cache", 100)
        return loaded

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    n = len(texts)

    ui_update(f"{company}: embedding {n} chunks (parallel) ...", 0)

    t0 = time.time()
    
    # 优化 4 应用：防抖机制，保护主线程避免被 nvidia-smi 卡死
    last_update_time = [time.time()]

    def on_prog(done, total):
        current_time = time.time()
        p = int((done / max(total, 1)) * 100)
        ui_update(f"{company}: embedding {n} chunks (parallel) ...", p)
        
        if current_time - last_update_time[0] > 1.0 or done == total:
            ui_update(f"{company}: embedding {n} chunks (parallel) ...", p)
            last_update_time[0] = current_time

    vectors = embed_parallel(texts, embeddings, batch_size=batch_size, max_workers=max_workers, progress_cb=on_prog)
    vs = FAISS.from_embeddings(text_embeddings=list(zip(texts, vectors)), embedding=embeddings, metadatas=metadatas)
    vectorstore_save(vs, index_key, company)

    dt = time.time() - t0
    ui_update(f"{company}: built & cached in {dt:.1f}s", 100)

    return vs


if uploaded:
    try:
        update_perf_panel("Preparing indexes ...", 0)
        with st.spinner("Preparing vector stores (load cache or build)..."):
            build_indexes()
        st.success("✅ Vector stores ready (cached by embedding model + chunk params).")
        st.caption(f"Cache root: {CACHE_ROOT} · Cache key: {st.session_state.stores_key}")
    except Exception as e:
        st.error(f"Index build failed: {e}")
        st.stop()
else:
    st.info("Upload PDFs to begin.")

col1, col2 = st.columns([2, 1], gap="large")
with col1:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

with col2:
    st.subheader("Detected Buckets")
    bucket_rows = [{"Bucket": k, "Chunks": v} for k, v in st.session_state.bucket_stats.items() if k in st.session_state.bucket_labels]
    if bucket_rows:
        st.table(bucket_rows)
    else:
        st.caption("No buckets yet.")

    st.subheader("Retrieved Context")
    show_for = st.selectbox("Show context for", st.session_state.bucket_labels, index=0)
    docs = st.session_state.last_retrieved.get(show_for, []) or []
    if docs:
        for i, d in enumerate(docs[:3], 1):
            page = d.metadata.get("page", "unknown")
            st.markdown(f"**Chunk {i} | Page {page}**")
            st.write(excerpt(d.page_content, 0, size=220))
        with st.expander("Show all retrieved context"):
            for i, d in enumerate(docs, 1):
                page = d.metadata.get("page", "unknown")
                st.markdown(f"**Chunk {i} | Page {page}**")
                st.write(d.page_content)
                st.markdown("---")
    else:
        st.caption("No retrieved chunks yet.")

    st.subheader("Answer Citations")
    st.caption(f"Latest answer bucket: {st.session_state.last_answer_bucket}")
    if st.session_state.last_citations:
        st.table(st.session_state.last_citations[:3])
    else:
        st.caption("No citations yet.")

    if st.session_state.citation_history:
        with st.expander("Citation history"):
            history = st.session_state.citation_history
            page_size = st.selectbox("Page size", [5, 10, 20], index=0)
            total_pages = max(1, (len(history) + page_size - 1) // page_size)
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=total_pages, step=1)
            start = (page - 1) * page_size
            end = start + page_size
            visible = history[start:end]

            options = [
                f"{h['ts']} · Q{start + i + 1}: {h['question'][:60]}"
                for i, h in enumerate(visible)
            ]
            sel = st.selectbox("Pick a question", options, index=len(options) - 1)
            pick = visible[options.index(sel)]
            st.caption(f"Bucket: {pick['bucket']}")
            if pick["citations"]:
                st.table(pick["citations"])
            else:
                st.caption("No citations captured for this question.")

q = st.chat_input("Ask about the uploaded 10-Ks...")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    if not st.session_state.stores:
        with st.chat_message("assistant"):
            st.markdown("Please upload PDFs and ensure vector stores are ready.")
        st.stop()

    try:
        llm = build_llm(llm_provider, llm_model, temperature, keys)
    except Exception as e:
        with st.chat_message("assistant"):
            st.markdown(f"LLM setup failed: {e}")
        st.stop()

    chosen = route_mode
    if chosen == "Auto":
        chosen = route_bucket(q, st.session_state.bucket_labels, st.session_state.bucket_profiles)
    if chosen not in st.session_state.bucket_labels:
        chosen = "All"

    store = st.session_state.stores.get(chosen, st.session_state.stores.get("All"))
    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})

    hist = format_history(st.session_state.messages[:-1], limit=10)
    prompt = PROMPT_NUMERIC if is_numeric_question(q) else PROMPT_GENERAL

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(q)
    else:
        docs = retriever.get_relevant_documents(q)
    st.session_state.last_retrieved[chosen] = docs
    context_text = format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(lambda _: context_text),
            "question": RunnablePassthrough(),
            "persona": RunnableLambda(lambda _: persona),
            "chat_history": RunnableLambda(lambda _: hist),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Thinking..."):
        ans = chain.invoke(q)

    if strict_refusal:
        ctx_text = format_docs(st.session_state.last_retrieved.get(chosen, []) or [])
        if len(ctx_text.strip()) < min_context_chars and "I don't have enough information" not in ans:
            ans = "I don't have enough information to answer this question."

    with st.chat_message("assistant"):
        ph = st.empty()
        out = ""
        for w in ans.split():
            out += w + " "
            ph.markdown(out)
            time.sleep(0.008)

    st.session_state.messages.append({"role": "assistant", "content": ans})
    docs_for_citations = st.session_state.last_retrieved.get(chosen, []) or []
    st.session_state.last_citations = build_answer_citations(ans, docs_for_citations, max_items=5)
    st.session_state.last_answer_bucket = chosen
    st.session_state.citation_history.append({
        "question": q,
        "answer": ans,
        "bucket": chosen,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "citations": st.session_state.last_citations,
    })
    st.rerun()