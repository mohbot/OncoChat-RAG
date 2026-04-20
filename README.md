# Cancer Drugs RAG Chat System

A Retrieval-Augmented Generation (RAG) system designed to provide precise, cited answers to questions regarding FDA-approved cancer drugs. This system ingests official PDF drug labels, sections them by canonical headers, and allows for interactive querying via a local Large Language Model.

---

## 🚀 Key Features

* **Intelligent Sectioning:** Automatically identifies and parses 15+ canonical FDA label sections (e.g., *Indications and Usage*, *Adverse Reactions*, *Clinical Studies*).
* **Vector Search:** Uses **FAISS** and **Sentence-Transformers** for high-performance semantic retrieval.
* **Local LLM:** Powered by **Ollama** (Gemma 3) for private, local inference without data leaving your machine.
* **Source Citation:** Every answer includes citations to specific drug names and label sections.
* **Interactive CLI:** Includes a streaming chat interface with source inspection and index management.

---

## 🛠️ Components

| Component | Technology |
| :--- | :--- |
| **PDF Ingestion** | `pypdf` |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Vector Store** | `FAISS` (CPU) |
| **LLM Engine** | `Ollama` |
| **Default Model** | `gemma3:4b` |

---

## 📋 Prerequisites

1.  **Python 3.8+**
2.  **Ollama:** [Download and install Ollama](https://ollama.ai/).
3.  **Model:** Pull the required model:
    ```bash
    ollama pull gemma3:4b
    ```

---

## 🔧 Installation

1.  **Clone the repository** (or save the script).
2.  **Install dependencies:**
    ```bash
    pip install pypdf sentence-transformers faiss-cpu numpy
    ```

---

## 📖 Usage

### 1. Prepare Data
Place your FDA drug label PDFs in a directory named `drug_reports/`. The system expects filenames to follow a pattern including the drug name (e.g., `osimertinib_2022_label.pdf`).

### 2. Run the System
Start the interactive chat session:
```bash
python OncoChat-RAG.py.py
```

### 3. CLI Arguments
You can customize the directory paths and model settings:
```bash
python OncoChat-RAG.py --pdf-dir ./my_pdfs --model llama3 --top-k 10
```

* `--pdf-dir`: Path to folder containing PDFs.
* `--index-dir`: Where to save the FAISS index.
* `--model`: Name of the Ollama model to use.
* `--rebuild`: Force the system to re-parse PDFs and rebuild the index.

---

## 💬 Chat Commands

Inside the chat loop, you can use:
* `sources`: Display the specific chunks and relevance scores used for the last answer.
* `rebuild`: Trigger a fresh scan of the PDF directory.
* `help`: Show the command menu.
* `quit`: Exit the application.

---

## ⚠️ Disclaimer
> This tool is for informational and research purposes only. It parses raw FDA data but is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a healthcare professional.