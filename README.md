# Capstone-RAG

Measuring how well pre-processing and post-processing measures in a Retrieval-Augmented Generation (RAG) pipeline reduce **hallucinations** and **indirect prompt injections**.

Four pipeline variants (a baseline and three mitigations) are implemented in two RAG frameworks, **LangChain** and **LlamaIndex**. Each can be run against four LLMs from the command line, so every measure can be compared across models and frameworks under the same test set.

---

## Pipeline variants

Each variant lives in its own folder under `RAG/<framework>/` and is selected with `--pipeline`.

| Pipeline | Pre-processing (before generation) | Post-processing (after generation) | Scoring |
|---|---|---|---|
| `baseline` | None: PDF split into 1000-char chunks (200 overlap), top-16 FAISS similarity search | None | Heuristic |
| `pre_processing` | 800-char chunks (150 overlap) with custom separators and source metadata; query normalisation; hybrid retrieval (MMR + FAISS with distance threshold + BM25); numeric-signal boosting (ABNs, phone numbers, long IDs); deduplication and re-ranking | None | Heuristic |
| `post_processing` | None (same retrieval as `baseline`) | Answer split into sentence-level claims (NLTK); each claim judged 0–100 by an LLM; scores averaged | LLM judge |
| `combined` | Same as `pre_processing` | Same as `post_processing` | LLM judge |

All variants embed chunks with OpenAI `text-embedding-3-small` and store them in a FAISS `IndexFlatL2` index.

**Scoring methods**
- **Heuristic** (`evaluator.py` in `baseline` and `pre_processing`): weighted mix of exact match (40%), fuzzy match (20%), embedding similarity using `all-MiniLM-L6-v2` (30%) and keyword overlap (10%), on a 0–100 scale.
- **LLM judge** (`evaluator.py` in `post_processing` and `combined`): the judge model compares each claim against the ground-truth answer and returns 0–100; the claim scores are averaged.

---

## Frameworks

Selected with `--framework`. Both frameworks implement every variant with the same parameters (chunk sizes, `k` values, MMR settings, FAISS distance threshold, prompts and scoring), so differences in results come from the frameworks themselves.

| Component | LangChain (`RAG/langchain/`) | LlamaIndex (`RAG/llamaindex/`) |
|---|---|---|
| PDF loading | `PyPDFLoader` | `PDFReader` |
| Chunking | `RecursiveCharacterTextSplitter` | `SentenceSplitter`, sized in characters |
| Embeddings | `OpenAIEmbeddings` | `OpenAIEmbedding` |
| Vector store | `FAISS` | `VectorStoreIndex` on `FaissVectorStore` |
| MMR | `max_marginal_relevance_search` | FAISS top-12 re-ranked with LlamaIndex's MMR |
| BM25 | `BM25Retriever` (`rank_bm25`) | `BM25Retriever` (`bm25s`) |
| Prompts and LLM calls | `ChatPromptTemplate` piped into LangChain chat models | `ChatPromptTemplate` with `llm.predict()` on LlamaIndex LLMs |
| Pipeline class | `LCPipeline` | `LIPipeline` |

Differences that remain by design, because each framework uses its own components:
- **BM25 tokenisation**: LangChain splits on whitespace and matches case-sensitively (against a lower-cased query); LlamaIndex lower-cases, stems and removes stopwords.
- **Embedding input**: LlamaIndex's OpenAI embeddings replace newlines with spaces before embedding; LangChain sends the text unchanged.
- **MMR**: LlamaIndex penalises similarity to the most recently selected chunk, LangChain to all selected chunks. Because `fetch_k` (12) is below `k` (16), both return the same 12 chunks, in a different order.
- **Chunk boundaries**: the splitters use different rules for where to break text, so chunks of the same size can start and end in different places.

---

## Models

Selected with `--model` (the answering model) and `--judge` (the scoring model). Defined in `MODELS` in `config.py`, where `get_llm()` builds each model with the selected framework's classes.

| CLI name | Model | Runs |
|---|---|---|
| `gpt` | `gpt-5-mini` | OpenAI API |
| `gemini` | `gemini-2.5-flash` | Google Gemini API |
| `qwen` | `Qwen/Qwen2.5-1.5B-Instruct` | Locally via HuggingFace Transformers |
| `deepseek` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Locally via HuggingFace Transformers |

Local models are downloaded from HuggingFace the first time they are selected.

---

## Directory structure

```
Capstone-RAG/
├── main.py                   # CLI entry point: runs the hallucination test set
├── config.py                 # API keys (.env), the model registry and get_llm() for both frameworks
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not committed, see Setup)
├── RAG/
│   ├── langchain/            # LangChain pipelines (LCPipeline)
│   │   ├── baseline/
│   │   ├── pre_processing/
│   │   ├── post_processing/
│   │   └── combined/
│   │       ├── indexer.py    # loads PDFs and splits them into chunks
│   │       ├── datastore.py  # builds / loads the FAISS vector store
│   │       ├── retriever.py  # retrieves relevant chunks for a question
│   │       ├── generator.py  # prompts the selected LLM with the retrieved context
│   │       ├── evaluator.py  # scores the answer against the ground truth
│   │       └── pipeline.py   # LCPipeline: wires the modules together
│   └── llamaindex/           # LlamaIndex pipelines (LIPipeline): same variants and modules
├── data/                     # test data (gitignored, add locally)
│   ├── hallucination/        # Swan Lagoon PDFs + swan_lagoon.json (25 question/answer pairs)
│   └── injection/            # indirect prompt injection test data
└── faiss_index/              # generated vector stores, one folder per framework and pipeline
```

Every pipeline folder contains the same six modules. The differences between variants are described in [Pipeline variants](#pipeline-variants), and the differences between frameworks in [Frameworks](#frameworks).

---

## Setup

1. **Create a virtual environment and install dependencies** (installs both LangChain and LlamaIndex)

   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add API keys.** Create a `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-openai-key
   GEMINI_API_KEY=your-gemini-key
   ```

   `OPENAI_API_KEY` is always required, because every pipeline uses OpenAI embeddings. `GEMINI_API_KEY` is only needed when `gemini` is used as `--model` or `--judge`.

3. **Add the test data.** `data/` is gitignored. Place the files in `data/hallucination/`:
   - `sl_booklet.pdf`
   - `sl_business_directory.pdf`
   - `sl_service_guide.pdf`
   - `sl_tourist_guide.pdf`
   - `swan_lagoon.json`

4. **(Optional) GPU for local models.** The default `torch` from pip is CPU-only on Windows. To run `qwen` or `deepseek` on an NVIDIA GPU, install the CUDA build using the command from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Commands

```powershell
python main.py [--framework FRAMEWORK] [--pipeline PIPELINE] [--model MODEL] [--judge MODEL]
```

| Flag | Choices | Default | Purpose |
|---|---|---|---|
| `--framework` | `langchain`, `llamaindex` | `langchain` | RAG framework the pipeline is built with |
| `--pipeline` | `baseline`, `pre_processing`, `post_processing`, `combined` | `combined` | RAG variant to test |
| `--model` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that answers the questions |
| `--judge` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that scores the answers (`post_processing` and `combined` only; ignored by the heuristic pipelines) |

### Examples

```powershell
# Show all options
python main.py --help

# Run with defaults (LangChain, combined pipeline, GPT answering, GPT judging)
python main.py

# One run per pipeline
python main.py --pipeline baseline --model gpt
python main.py --pipeline pre_processing --model gemini
python main.py --pipeline post_processing --model gpt --judge gemini
python main.py --pipeline combined --model gemini --judge gpt

# The same pipelines built with LlamaIndex
python main.py --framework llamaindex --pipeline baseline --model gpt
python main.py --framework llamaindex --pipeline combined --model gemini --judge gpt

# Local models
python main.py --pipeline baseline --model qwen
python main.py --framework llamaindex --pipeline combined --model deepseek --judge gpt

# Run every framework, pipeline and model, saving each run's output
foreach ($f in 'langchain','llamaindex') {
    foreach ($p in 'baseline','pre_processing','post_processing','combined') {
        foreach ($m in 'gpt','gemini','qwen','deepseek') {
            python main.py --framework $f --pipeline $p --model $m --judge gpt | Tee-Object "results_${f}_${p}_${m}.txt"
        }
    }
}
```

---

## Output

Each run first prints the selected framework, pipeline, model and judge, ingests the PDFs into `faiss_index/<framework>/<pipeline>/`, then prints for each of the 25 test cases:

```
Question 1: Who discovered gold in Swan Lagoon and founded the town?
Actual Answer: Jedediah 'Jed' Stone.
Response: <model answer>
Evaluation Result: <score, 0–100>
Total Time: <seconds>
```

Both scoring methods use a 0–100 scale, but they measure different things (surface and semantic similarity vs. LLM-judged correctness), so compare scores within the same scoring method.

---

## Known limitations

- In both frameworks, `baseline` and `post_processing` rebuild the vector store on every `ingest()` call, so only the last ingested PDF ends up indexed.
- Heuristic and LLM-judge scores are not directly comparable across pipelines.
- Retrieval differs slightly between frameworks for the reasons listed under [Frameworks](#frameworks).
- `deepseek` (DeepSeek-R1) includes `<think>` reasoning text in its answers, which affects scoring.
- Indirect prompt injection tests are not implemented yet (placeholder at the end of `main.py`).
