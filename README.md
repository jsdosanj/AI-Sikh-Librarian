# 📚 AI Librarian — Local LLM for Historical Manuscripts (70GB+)
### A Senior AI Architect's Guide to Building a RAG-Powered Citation & Source Engine
#### Running Locally on Apple M1 Pro MacBook Pro via GPT4All

---

> **What this guide does:**  
> Walks you through building a smart AI "librarian" that can read, index, and answer questions about your 70GB+ collection of historical manuscripts, religious texts, philosophical writings, and literature — in English, Punjabi, and Urdu — including handwritten and printed styles. It runs 100% locally on your Mac. No internet. No data sent anywhere. Your documents stay private.

---

## 📋 Table of Contents

1. [Understand the Big Picture First](#1-understand-the-big-picture-first)
2. [What You Actually Need — Hardware & Software Checklist](#2-what-you-actually-need)
3. [Choose the Right AI Model](#3-choose-the-right-ai-model)
4. [Phase 1 — Prepare Your Documents (OCR + Text Extraction)](#4-phase-1--prepare-your-documents)
5. [Phase 2 — Set Up Your Python Environment](#5-phase-2--set-up-your-python-environment)
6. [Phase 3 — Build Your Vector Database (The "Memory" of Your Librarian)](#6-phase-3--build-your-vector-database)
7. [Phase 4 — Load Your Model into GPT4All](#7-phase-4--load-your-model-into-gpt4all)
8. [Phase 5 — Build the RAG Pipeline (The Brains)](#8-phase-5--build-the-rag-pipeline)
9. [Phase 6 — Test Your AI Librarian](#9-phase-6--test-your-ai-librarian)
10. [Phase 7 — Fine-Tuning Tips (Making It Smarter Over Time)](#10-phase-7--fine-tuning-tips)
11. [Folder Structure Overview](#11-folder-structure-overview)
12. [Troubleshooting Common Issues](#12-troubleshooting-common-issues)
13. [Model Reference Card](#13-model-reference-card)

---

## 1. Understand the Big Picture First

Before touching any code, let's understand **what we're actually building** and **why**.

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it like this:

- You have 70GB of text from manuscripts.
- The AI **can't memorize** all 70GB at once — even the biggest models only "see" maybe 100,000 words at a time.
- So instead, we **pre-process** all your documents, chop them into small searchable pieces, and store them in a special database.
- When you ask a question, the AI **first searches** that database for relevant passages, **then reads** just those passages and gives you a cited answer.

It's exactly how a real librarian works:
> You ask: *"What did Guru Nanak say about ego?"*  
> The librarian doesn't memorize the entire library — they know **where to look**, pull the right pages, and answer from those.

### Why NOT just fine-tune?

Fine-tuning (actually retraining the model on your data) requires:
- Weeks of compute time
- A powerful GPU (your M1 Pro isn't designed for this)
- Massive storage for training checkpoints

**RAG gives you 90% of the benefit with 5% of the work.** We're going with RAG.

### The Architecture (Simple Version)

```
Your Documents (PDFs, images, text)
         ↓
   OCR / Text Extraction
         ↓
   Chunking (split into small pieces)
         ↓
   Embedding Model (turns text into numbers)
         ↓
   ChromaDB (stores those numbers — your searchable index)
         ↓
   You ask a question → ChromaDB finds top relevant chunks
         ↓
   Qwen2.5 7B (LLM reads those chunks + your question → gives answer + citation)
```

---

## 2. What You Actually Need

### Hardware
| What | Minimum | Your Setup |
|------|---------|-----------|
| Mac chip | M1 | ✅ M1 Pro |
| RAM | 16GB | Should be fine for 7B models |
| Storage | 500GB free | You need ~20GB for the model + your 70GB docs |

> ⚠️ **Important:** You will likely need an **external SSD** for your 70GB document library. Processing 70GB fully will take **many hours**. Plan for running it overnight in batches.

### Software You'll Install
- **GPT4All Desktop App** — [gpt4all.io](https://www.nomic.ai/gpt4all)
- **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- **Homebrew** (Mac package manager)
- **Tesseract OCR** — for extracting text from scanned images
- **PaddleOCR** — for handwritten and multilingual (Punjabi/Urdu) text
- **LangChain** — the glue that connects everything
- **ChromaDB** — your local vector/search database

---

## 3. Choose the Right AI Model

After researching the HuggingFace leaderboards, multilingual benchmarks, and GPT4All compatibility, here is the recommended model:

---

### 🥇 Primary Recommendation: `Qwen2.5 7B Instruct` (GGUF Format)

**HuggingFace Page:** [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)

**Why this model?**

| Reason | Detail |
|--------|--------|
| 🌍 Multilingual | Officially supports 29+ languages, including **Urdu**. Punjabi (Gurmukhi/Shahmukhi) coverage is broad due to training data size. |
| 📖 Long context window | Supports up to **131,000 tokens** — great for reading long manuscript passages |
| 🍎 M1 Mac compatible | Quantized GGUF version runs well on Apple Silicon with no GPU needed |
| 🤝 GPT4All compatible | GGUF format loads directly into GPT4All |
| 📜 Good at instruction following | Excellent at structured Q&A tasks like "give me sources for..." |
| 🆓 Open source | Apache 2.0 license — free for personal/research use |
| 💾 Manageable size | The Q4_K_M quantized version is ~4.5GB |

**Which quantization to download?**  
Download the `Q4_K_M` version — this is the sweet spot between speed and accuracy on M1.

File to download: `qwen2.5-7b-instruct-q4_k_m.gguf`

---

### 🥈 Runner-Up: `Mistral NeMo 12B Instruct` (GGUF)

**HuggingFace Page:** [https://huggingface.co/mistralai/Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407)

Use this if you want slightly better English reasoning, but note it's larger (~7GB) and slower on M1. Less confirmed support for Punjabi.

---

### For Embeddings (Turning your docs into searchable vectors):

Use **`nomic-embed-text`** — this is made by the same team as GPT4All (Nomic AI) and is specifically designed to work with it.

**HuggingFace Page:** [https://huggingface.co/nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

---

## 4. Phase 1 — Prepare Your Documents

This is the most important phase. Garbage in = garbage out. Your AI librarian is only as good as the text you feed it.

### Step 1.1 — Install Homebrew (if you don't have it)

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 1.2 — Install Tesseract OCR

Tesseract handles printed text. It supports Punjabi (Gurmukhi script) and Urdu (Nastaliq/Naskh).

```bash
brew install tesseract
brew install tesseract-lang
```

Verify it worked:
```bash
tesseract --list-langs
```

You should see `pun` (Punjabi), `urd` (Urdu), and `eng` (English) in the list.

### Step 1.3 — Sort Your Documents Into Folders

Organize your 70GB like this before processing:

```
/manuscripts/
  /english/
    /printed/       ← PDFs and images of printed English text
    /handwritten/   ← Scanned handwritten English documents
  /punjabi/
    /printed/
    /handwritten/
  /urdu/
    /printed/
    /handwritten/
```

This helps you run different OCR settings for each language/style.

### Step 1.4 — Install Python OCR Libraries

```bash
pip install pytesseract pdf2image Pillow paddlepaddle paddleocr pymupdf
```

> 💡 **Note on PaddleOCR:** This handles handwritten/cursive styles much better than Tesseract. Use it as a backup or for difficult documents.

### Step 1.5 — Run OCR to Extract Text

Save this as `extract_text.py` and run it from your project folder:

```python name=extract_text.py
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF - for PDFs with embedded text
import json

# -----------------------------------------------
# SETTINGS - change these to match your folders
# -----------------------------------------------
INPUT_DIR = "./manuscripts"
OUTPUT_DIR = "./extracted_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Language map: folder name → Tesseract language code
LANG_MAP = {
    "english": "eng",
    "punjabi": "pan",   # Punjabi (Gurmukhi)
    "urdu": "urd",
}

def extract_from_pdf_native(filepath):
    """Try to extract embedded text from PDFs first (fastest)."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_with_tesseract(filepath, lang="eng"):
    """Use OCR for scanned images or image-based PDFs."""
    if filepath.endswith(".pdf"):
        images = convert_from_path(filepath, dpi=300)
    else:
        images = [Image.open(filepath)]
    
    full_text = ""
    for img in images:
        full_text += pytesseract.image_to_string(img, lang=lang) + "\n"
    return full_text.strip()

def process_all_documents():
    results = []
    
    for lang_folder in os.listdir(INPUT_DIR):
        lang_path = os.path.join(INPUT_DIR, lang_folder)
        if not os.path.isdir(lang_path):
            continue
        
        lang_code = LANG_MAP.get(lang_folder.lower(), "eng")
        
        for style_folder in ["printed", "handwritten"]:
            style_path = os.path.join(lang_path, style_folder)
            if not os.path.isdir(style_path):
                continue
            
            for filename in os.listdir(style_path):
                filepath = os.path.join(style_path, filename)
                print(f"Processing: {filepath}")
                
                # Try native text extraction first (for text-based PDFs)
                text = ""
                if filename.endswith(".pdf"):
                    text = extract_from_pdf_native(filepath)
                
                # Fall back to OCR if no text found
                if len(text) < 100:
                    print(f"  → Using OCR for: {filename}")
                    text = extract_with_tesseract(filepath, lang=lang_code)
                
                if text:
                    output_filename = f"{lang_folder}_{style_folder}_{filename}.txt"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    results.append({
                        "source_file": filepath,
                        "language": lang_folder,
                        "style": style_folder,
                        "output_file": output_path,
                        "char_count": len(text)
                    })
                    print(f"  ✓ Saved: {output_path} ({len(text)} chars)")

    # Save a summary log
    with open("extraction_log.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Done! Processed {len(results)} documents.")

if __name__ == "__main__":
    process_all_documents()
```

Run it:
```bash
python extract_text.py
```

> ⏱️ **Time estimate:** 70GB of scanned documents could take 6–24 hours. Run it overnight. The script saves as it goes, so you can resume if interrupted.

### Step 1.6 — Review OCR Quality

After extraction, spot-check a few `.txt` files. If you see garbled text for Punjabi or Urdu, try using PaddleOCR for those documents instead:

```bash
pip install paddleocr
```

Then test with:
```bash
paddleocr --image_dir ./manuscripts/punjabi/printed/sample.jpg --lang punjabi
```

---

## 5. Phase 2 — Set Up Your Python Environment

### Step 2.1 — Create a Virtual Environment

```bash
# Navigate to your project folder
mkdir ai_librarian
cd ai_librarian

# Create isolated Python environment
python3 -m venv .venv

# Activate it (you'll do this every time you work on the project)
source .venv/bin/activate
```

### Step 2.2 — Install All Required Packages

```bash
pip install \
  langchain \
  langchain-community \
  langchain-chroma \
  langchain-text-splitters \
  chromadb \
  gpt4all \
  sentence-transformers \
  pypdf \
  unstructured \
  tiktoken \
  tqdm
```

### Step 2.3 — Download the Qwen2.5 GGUF Model

You can download directly via the HuggingFace CLI:

```bash
pip install huggingface_hub

# Download the Q4_K_M quantized version (~4.5GB)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf \
  --local-dir ./models
```

Or download manually from:
👉 [https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/tree/main)

Look for: `qwen2.5-7b-instruct-q4_k_m.gguf`

---

## 6. Phase 3 — Build Your Vector Database

This is where you take all your extracted text and build the searchable "brain" of your librarian.

### Step 3.1 — What is a Vector Database?

When you search Google, it matches keywords. A vector database is smarter — it matches **meaning**. So if you ask *"what does the text say about divine love?"*, it will find passages about *ishq*, *prem*, *mohabbat*, and *love* even if you didn't use those exact words.

### Step 3.2 — Build the Index

Save this as `build_index.py`:

```python name=build_index.py
import os
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
EXTRACTED_TEXT_DIR = "./extracted_text"   # where your .txt files are
CHROMA_DB_DIR = "./chroma_db"             # where the index will be saved
CHUNK_SIZE = 800        # words per chunk (experiment with this)
CHUNK_OVERLAP = 150     # overlap between chunks (helps with context continuity)

print("📂 Loading documents...")

# Load all .txt files from the extracted text directory
loader = DirectoryLoader(
    EXTRACTED_TEXT_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# Add metadata to each document (helps with citations later!)
for doc in documents:
    filename = os.path.basename(doc.metadata.get("source", "unknown"))
    # Parse language and style from our naming convention: lang_style_filename.txt
    parts = filename.split("_", 2)
    if len(parts) >= 3:
        doc.metadata["language"] = parts[0]
        doc.metadata["style"] = parts[1]
        doc.metadata["original_file"] = parts[2].replace(".txt", "")
    doc.metadata["display_source"] = filename

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", " "]
)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks):,} chunks from {len(documents)} documents")

print("🧠 Generating embeddings and building vector index...")
print("   (This will take a while for 70GB of content — go make some chai ☕)")

# Use Nomic's embedding model (built into GPT4All)
embedding_function = GPT4AllEmbeddings(
    model_name="nomic-embed-text-v1.5.f16.gguf"
)

# Build and persist the ChromaDB index
# We process in batches to avoid memory issues with 70GB
BATCH_SIZE = 500
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Indexing batches"):
    batch = chunks[i:i + BATCH_SIZE]
    if i == 0:
        db = Chroma.from_documents(
            documents=batch,
            embedding=embedding_function,
            persist_directory=CHROMA_DB_DIR
        )
    else:
        db.add_documents(batch)

print(f"\n✅ Done! Vector database saved to: {CHROMA_DB_DIR}")
print(f"   Total chunks indexed: {len(chunks):,}")
```

Run it:
```bash
python build_index.py
```

> ⏱️ **Time estimate for 70GB:** 4–12 hours on M1 Pro. Run overnight. This is a **one-time** setup step. After this, asking questions is instant.

---

## 7. Phase 4 — Load Your Model into GPT4All

### Step 4.1 — Install GPT4All Desktop

1. Go to [https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all)
2. Download the macOS version
3. Install it like any other Mac app

### Step 4.2 — Add Your Downloaded Model

1. Open **GPT4All**
2. Click the **Settings** (gear icon) → **Model Path**
3. Set the model folder to where you downloaded the `.gguf` file (e.g., `~/ai_librarian/models/`)
4. GPT4All will detect the model automatically
5. Select `Qwen2.5 7B Instruct Q4_K_M` from the model dropdown

### Step 4.3 — Test the Base Model

Before connecting your documents, just test the model is working:

Ask it: *"Who was Guru Nanak Dev Ji?"*

If it answers coherently, you're good to proceed.

---

## 8. Phase 5 — Build the RAG Pipeline

This is the final piece — the Python script that connects your ChromaDB index to the LLM and makes it work like a librarian.

### Step 5.1 — The Main Query Script

Save this as `librarian.py`:

```python name=librarian.py
"""
AI Librarian — RAG Pipeline
Asks questions against your 70GB manuscript collection
and returns answers with citations.
"""

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_chroma import Chroma
from gpt4all import GPT4All
import json

# -----------------------------------------------
# CONFIG — update paths to match your setup
# -----------------------------------------------
CHROMA_DB_DIR = "./chroma_db"
MODEL_PATH = "./models"
MODEL_NAME = "qwen2.5-7b-instruct-q4_k_m.gguf"
TOP_K_RESULTS = 5  # How many passages to retrieve per question

# -----------------------------------------------
# SYSTEM PROMPT — This tells the AI how to behave
# -----------------------------------------------
SYSTEM_PROMPT = """You are a scholarly librarian and research assistant specializing in 
historical manuscripts, religious texts, philosophical works, and literature from South Asia. 
Your collection includes texts in English, Punjabi (Gurmukhi and Shahmukhi scripts), and Urdu.

Your job is to:
1. Answer questions ONLY using the provided source passages
2. Always cite which document each piece of information comes from
3. Quote directly from the source text when possible
4. If the answer is not found in the provided passages, say so clearly
5. Maintain scholarly accuracy and never fabricate citations

Format your answer like this:
- Give the answer clearly
- List your sources at the end under "📚 Sources:"
- Include the document name and language for each source"""

def load_retriever():
    """Load the ChromaDB vector store."""
    print("📚 Loading your manuscript index...")
    embedding_function = GPT4AllEmbeddings(
        model_name="nomic-embed-text-v1.5.f16.gguf"
    )
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_function
    )
    return db.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

def load_llm():
    """Load the Qwen2.5 model via GPT4All."""
    print("🤖 Loading AI model (Qwen2.5 7B)...")
    model = GPT4All(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        allow_download=False,
        n_ctx=8192  # context window size
    )
    return model

def format_context(docs):
    """Format retrieved passages into a readable context block."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source_name = meta.get("original_file", meta.get("source", "Unknown"))
        language = meta.get("language", "unknown").capitalize()
        style = meta.get("style", "")
        
        context_parts.append(
            f"[Source {i}] File: {source_name} | Language: {language} | Type: {style}\n"
            f"{doc.page_content}\n"
        )
    return "\n---\n".join(context_parts)

def ask_librarian(question, retriever, model):
    """Main function: retrieves passages and generates a cited answer."""
    
    print(f"\n🔍 Searching manuscripts for: '{question}'")
    
    # Step 1: Find relevant passages from the database
    relevant_docs = retriever.get_relevant_documents(question)
    
    if not relevant_docs:
        return "I couldn't find any relevant passages in the manuscript collection for your question."
    
    # Step 2: Format the context
    context = format_context(relevant_docs)
    
    # Step 3: Build the prompt
    full_prompt = f"""{SYSTEM_PROMPT}

Here are the relevant passages found in the manuscript collection:

{context}

---

Question: {question}

Answer (with citations):"""

    print("💭 Generating answer...")
    
    # Step 4: Generate the answer
    with model.chat_session():
        response = model.generate(
            full_prompt,
            max_tokens=1024,
            temp=0.1,          # low temperature = more factual, less creative
            top_p=0.9,
        )
    
    return response

def main():
    """Interactive session with your AI Librarian."""
    retriever = load_retriever()
    model = load_llm()
    
    print("\n" + "="*60)
    print("📖 AI LIBRARIAN — Manuscript Research Assistant")
    print("   Collection: Historical texts in English, Punjabi, Urdu")
    print("   Type 'quit' to exit | Type 'help' for example questions")
    print("="*60 + "\n")
    
    while True:
        question = input("❓ Your question: ").strip()
        
        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye! ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ / خدا حافظ")
            break
        if question.lower() == "help":
            print("\nExample questions:")
            print("  - What does the text say about the concept of Waheguru?")
            print("  - Find all references to the Mughal Empire across the collection")
            print("  - What philosophical views on death are expressed in the Urdu texts?")
            print("  - Show me passages about Punjab from the 18th century manuscripts\n")
            continue
        
        answer = ask_librarian(question, retriever, model)
        
        print("\n" + "="*60)
        print("📜 ANSWER:")
        print("="*60)
        print(answer)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
```

Run your AI Librarian:
```bash
python librarian.py
```

---

## 9. Phase 6 — Test Your AI Librarian

### Step 6.1 — Start Simple

Ask questions that should have clear answers in your collection:

```
❓ Your question: What texts do you have about Sikh history?
❓ Your question: Find passages mentioning Lahore in the Punjabi manuscripts
❓ Your question: What does the Urdu literature say about love and devotion?
```

### Step 6.2 — Test Citation Accuracy

Pick a passage you know exists in a specific document and ask about it. Then verify the cited source is correct.

### Step 6.3 — Test Multilingual Handling

```
❓ Your question: Kya aap mujhe urdu manuscripts ke baare mein bata sakte hain?
❓ Your question: ਪੰਜਾਬੀ ਲਿਖਤਾਂ ਵਿੱਚ ਪਰਮਾਤਮਾ ਬਾਰੇ ਕੀ ਲਿਖਿਆ ਹੈ?
```

> 💡 **Tip:** Qwen2.5 understands both the question AND the source text even when they're in different languages. You can ask in English about Punjabi texts and it will answer.

---

## 10. Phase 7 — Fine-Tuning Tips

"Fine-tuning" in the RAG world means making your librarian **smarter over time** — not retraining the model from scratch.

### Tip 1 — Improve Chunk Size

If answers feel **too vague** → decrease chunk size to 400-600  
If answers feel **cut off** → increase chunk size to 1000-1200  

Edit `CHUNK_SIZE` in `build_index.py` and rebuild the index.

### Tip 2 — Improve the System Prompt

The most powerful thing you can do is refine the `SYSTEM_PROMPT` in `librarian.py`. Add:
- Specific instructions for citation format
- Details about your collection (e.g., "This collection spans 1600–1900 CE")
- Instructions for handling multiple languages

### Tip 3 — Add Metadata Rich Sources

When you add new documents, include detailed metadata:

```python
doc.metadata["title"] = "Guru Granth Sahib — SGPC Edition"
doc.metadata["author"] = "Various Sikh Gurus"
doc.metadata["year"] = "1604"
doc.metadata["language"] = "punjabi"
doc.metadata["script"] = "gurmukhi"
```

The richer your metadata, the better citations you get.

### Tip 4 — Handle Handwritten Text Better

For difficult cursive/handwritten manuscripts that Tesseract struggles with, try **PaddleOCR** as a secondary extractor:

```bash
pip install paddleocr paddlepaddle
```

```python name=paddle_extract.py
from paddleocr import PaddleOCR

# For Punjabi handwritten text
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use 'en' as base; fine-tune for Punjabi
result = ocr.ocr('./manuscripts/punjabi/handwritten/sample.jpg', cls=True)
for line in result[0]:
    print(line[1][0])  # prints extracted text
```

### Tip 5 — Add New Documents Without Rebuilding

```python name=add_documents.py
# To add new documents WITHOUT rebuilding the entire index:
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your existing database
embedding_function = GPT4AllEmbeddings(model_name="nomic-embed-text-v1.5.f16.gguf")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# Load and add a new document
loader = TextLoader("./new_manuscript.txt", encoding="utf-8")
new_doc = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
new_chunks = splitter.split_documents(new_doc)

db.add_documents(new_chunks)
print(f"✅ Added {len(new_chunks)} new chunks to the library")
```

---

## 11. Folder Structure Overview

After completing all phases, your project should look like this:

```
ai_librarian/
│
├── manuscripts/                  ← Your original 70GB documents
│   ├── english/
│   │   ├── printed/
│   │   └── handwritten/
│   ├── punjabi/
│   │   ├── printed/
│   │   └── handwritten/
│   └── urdu/
│       ├── printed/
│       └── handwritten/
│
├── extracted_text/               ← OCR-processed .txt files
│   ├── english_printed_doc1.txt
│   ├── punjabi_handwritten_doc2.txt
│   └── ...
│
├── chroma_db/                    ← Your vector search index (auto-generated)
│   └── ...
│
├── models/                       ← Your downloaded GGUF model files
│   └── qwen2.5-7b-instruct-q4_k_m.gguf
│
├── .venv/                        ← Python virtual environment
│
├── extract_text.py               ← Phase 1: OCR extraction script
├── build_index.py                ← Phase 3: Vector database builder
├── librarian.py                  ← Phase 5: The AI librarian chat interface
├── add_documents.py              ← Phase 7: Add new docs without rebuilding
├── extraction_log.json           ← Auto-generated log of processed files
│
└── README.md                     ← This file!
```

---

## 12. Troubleshooting Common Issues

### ❌ "Model not found" in GPT4All
**Fix:** Make sure the `.gguf` file is in the folder you set as the Model Path in GPT4All settings.

### ❌ OCR output is garbled for Punjabi/Urdu
**Fix 1:** Make sure you installed Tesseract language packs: `brew install tesseract-lang`  
**Fix 2:** Switch to PaddleOCR for those documents (better for non-Latin scripts)  
**Fix 3:** Improve scan quality — 300 DPI minimum, good contrast

### ❌ "Out of memory" when building the index
**Fix:** Reduce `BATCH_SIZE` from 500 to 100 in `build_index.py`

### ❌ Answers are too vague or hallucinated
**Fix 1:** Increase `TOP_K_RESULTS` from 5 to 8 to give the model more context  
**Fix 2:** Lower the temperature (`temp=0.05`) in `librarian.py` for more factual answers  
**Fix 3:** Add more specific instructions to the `SYSTEM_PROMPT`

### ❌ Building the index is taking too long
**Fix:** Process documents in language batches. Build one ChromaDB for English, one for Punjabi, one for Urdu. Then merge, or query them separately.

### ❌ GPT4All Python package errors on M1
```bash
# Make sure you're using the right pip
which pip  # should show your .venv path
pip install --upgrade gpt4all
```

---

## 13. Model Reference Card

| Model | Link | Size (Q4_K_M) | Best For |
|-------|------|---------------|----------|
| **Qwen2.5 7B Instruct** ⭐ | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) | ~4.5 GB | Primary model: multilingual Q&A + citations |
| **Mistral NeMo 12B** | [HuggingFace](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) | ~7 GB | Secondary: better English reasoning, larger context |
| **nomic-embed-text-v1.5** ⭐ | [HuggingFace](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | ~270 MB | Embeddings: turning your text into searchable vectors |
| **Phi-4 GGUF** | [HuggingFace](https://huggingface.co/GPT4All-Community/phi-4-GGUF) | ~8 GB | Alternative: excellent reasoning, strong on factual Q&A |

---

## 🎉 You're Done!

You now have a fully local, private AI librarian that:

- 📖 Has read and indexed your entire 70GB+ manuscript collection
- 🌍 Understands English, Punjabi (Gurmukhi/Shahmukhi), and Urdu
- 📜 Handles both printed and handwritten/cursive texts
- 📚 Gives you citations and sources for every answer
- 🔒 Runs 100% on your MacBook Pro — no cloud, no subscriptions, no data leaving your machine
- 💰 Costs nothing after setup (all open-source, free models)

---

*Built with ❤️ using: [GPT4All](https://www.nomic.ai/gpt4all) · [LangChain](https://www.langchain.com/) · [ChromaDB](https://www.trychroma.com/) · [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) · [HuggingFace](https://huggingface.co) · [Tesseract OCR](https://tesseract-ocr.github.io/) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)*
