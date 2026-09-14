# Capstone-RAG

Measuring how well pre-processing and post-processing measures in a Retrieval-Augmented Generation (RAG) pipeline reduce **hallucinations** and **indirect prompt injections**.

Four pipeline variants (a baseline and three mitigations) can be run against four LLMs from the command line, so each measure can be compared across models under the same test set.

---

## Pipeline variants

Each variant lives in its own folder under `RAG/` and is selected with `--pipeline`.

| Pipeline | Pre-processing (before generation) | Post-processing (after generation) | Scoring |
|---|---|---|---|
| `baseline` | None: PDF split into 500-char chunks (50 overlap), top-16 FAISS similarity search | None | Heuristic |
| `pre_processing` | 800-char chunks (150 overlap) with custom separators and source metadata; query normalisation; hybrid retrieval (MMR + FAISS with distance threshold + BM25); numeric-signal boosting (ABNs, phone numbers, long IDs); deduplication and re-ranking | None | Heuristic |
| `post_processing` | None (same retrieval as `baseline`) | Answer split into sentence-level claims (NLTK); each claim judged 0–100 by an LLM; scores averaged | LLM judge |
| `combined` | Same as `pre_processing` | Same as `post_processing` | LLM judge |

All variants embed chunks with OpenAI `text-embedding-3-small` and store them in FAISS.

**Scoring methods**
- **Heuristic** (`RAG/baseline/evaluator.py`, `RAG/pre_processing/evaluator.py`): weighted mix of exact match (40%), fuzzy match (20%), embedding similarity using `all-MiniLM-L6-v2` (30%) and keyword overlap (10%), on a 0–100 scale.
- **LLM judge** (`RAG/post_processing/evaluator.py`, `RAG/combined/evaluator.py`): the judge model compares each claim against the ground-truth answer and returns 0–100; the claim scores are averaged.

---

## Models

Selected with `--model` (the answering model) and `--judge` (the scoring model). Defined in `MODELS` in `config.py`.

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
├── main.py               # CLI entry point: runs the hallucination test set
├── config.py             # API keys (.env) and the model registry / get_llm() factory
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed, see Setup)
├── RAG/
│   ├── baseline/
│   ├── pre_processing/
│   ├── post_processing/
│   └── combined/
│       ├── indexer.py    # loads PDFs and splits them into chunks
│       ├── datastore.py  # builds / loads the FAISS vector store
│       ├── retriever.py  # retrieves relevant chunks for a question
│       ├── generator.py  # prompts the selected LLM with the retrieved context
│       ├── evaluator.py  # scores the answer against the ground truth
│       └── pipeline.py   # LCPipeline: wires the modules together
├── data/                 # test data (gitignored, add locally)
│   ├── hallucination/    # Swan Lagoon PDFs + swan_lagoon.json (25 question/answer pairs)
│   └── injection/        # indirect prompt injection test data
└── faiss_index/          # generated vector stores, one folder per pipeline
```

Every folder in `RAG/` contains the same six modules; the differences between variants are described in [Pipeline variants](#pipeline-variants).

---

## Setup

1. **Create a virtual environment and install dependencies**

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
python main.py [--pipeline PIPELINE] [--model MODEL] [--judge MODEL]
```

| Flag | Choices | Default | Purpose |
|---|---|---|---|
| `--pipeline` | `baseline`, `pre_processing`, `post_processing`, `combined` | `combined` | RAG variant to test |
| `--model` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that answers the questions |
| `--judge` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that scores the answers (`post_processing` and `combined` only; ignored by the heuristic pipelines) |

### Examples

```powershell
# Show all options
python main.py --help

# Run with defaults (combined pipeline, GPT answering, GPT judging)
python main.py

# One run per pipeline
python main.py --pipeline baseline --model gpt
python main.py --pipeline pre_processing --model gemini
python main.py --pipeline post_processing --model gpt --judge gemini
python main.py --pipeline combined --model gemini --judge gpt

# Local models
python main.py --pipeline baseline --model qwen
python main.py --pipeline combined --model deepseek --judge gpt

# Run every pipeline against every model, saving each run's output
foreach ($p in 'baseline','pre_processing','post_processing','combined') {
    foreach ($m in 'gpt','gemini','qwen','deepseek') {
        python main.py --pipeline $p --model $m --judge gpt | Tee-Object "results_${p}_${m}.txt"
    }
}
```

---

## Output

Each run first prints the selected pipeline, model and judge, ingests the PDFs into `faiss_index/<pipeline>/`, then prints for each of the 25 test cases:

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

- `baseline` and `post_processing` rebuild the vector store on every `ingest()` call, so only the last ingested PDF ends up indexed.
- Heuristic and LLM-judge scores are not directly comparable across pipelines.
- `deepseek` (DeepSeek-R1) includes `<think>` reasoning text in its answers, which affects scoring.
- Indirect prompt injection tests are not implemented yet (placeholder at the end of `main.py`).
