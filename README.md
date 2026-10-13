# Capstone-RAG

Measuring how well pre-processing and post-processing measures in a Retrieval-Augmented Generation (RAG) pipeline reduce **hallucinations** and **indirect prompt injections**.

Four pipeline variants (a baseline and three mitigations) are implemented in two RAG frameworks, **LangChain** and **LlamaIndex**. Each can be run against four LLMs from the command line, so every measure can be compared across models and frameworks under the same test set.

---

## Pipeline variants

Each variant lives in its own folder under `RAG/<test>/<framework>/` and is selected with `--pipeline`.

| Pipeline | Pre-processing (before generation) | Post-processing (after generation) | Scoring |
|---|---|---|---|
| `baseline` | None: PDF split into 1000-char chunks (200 overlap), top-16 FAISS similarity search | None | Heuristic |
| `pre_processing` | Hallucination test: nine hallucination measures. Injection test: injection defences D1–D5. Both listed below | None | Heuristic (hallucination), marker matching (injection) |
| `post_processing` | None (same retrieval and generation as `baseline`) | Eight hallucination measures, listed below (implemented under `RAG/hallucination/` only) | LLM judge |
| `combined` | The nine `pre_processing` measures | The eight `post_processing` measures, checked against the compressed context | LLM judge |

All variants embed chunks with OpenAI `text-embedding-3-small` and store them in a FAISS `IndexFlatL2` index.

**`pre_processing` measures**, applied per question in this order: rewrite (6) → normalise (2) → hybrid retrieve (3, 4) → boost, dedupe and re-rank (5) → grade (7) → NLI filter (8) → compress (9) → generate.

| # | Measure | Module | What it does |
|---|---|---|---|
| 1 | Chunking | `indexer.py` | 800-char chunks, 150 overlap, separator hierarchy `\n\n`, `\n`, `.`, space; source file path stored on every chunk |
| 2 | Query normalisation | `retriever.py` | Lower-case, strip and collapse whitespace before embedding and BM25 |
| 3 | Hybrid retrieval | `retriever.py` | Union of MMR (`k=16`, `fetch_k=48`, `lambda_mult=0.5`), dense top-10 and BM25 top-16 |
| 4 | Distance threshold | `retriever.py` | Dense hits kept only at FAISS squared-L2 distance ≤ 0.35 |
| 5 | Numeric-signal re-ranking | `retriever.py` | Chunks scored for ABNs (+5), AU phone numbers (+4), other phone numbers (+3) and 6+ digit IDs (+2), plus +2 if the query appears verbatim; deduplicated by content, top 16 kept |
| 6 | Query rewriting | `rewriter.py` | One LLM call turns the question into a search query, keeping proper nouns and numbers verbatim; the original question is used if the rewrite is empty, over 3× longer or drops a number |
| 7 | Corrective retrieval grading | `grader.py` | One LLM call grades the retrieved context (first 12,000 chars) `CORRECT`, `AMBIGUOUS` or `INCORRECT`. `AMBIGUOUS` re-retrieves once with a second, different rewrite (one more LLM call, not re-graded); `INCORRECT` skips generation and answers `Not found in context`; an unreadable grade counts as `CORRECT` |
| 8 | NLI relevance filter | `filter.py` | `cross-encoder/nli-deberta-v3-base` (fallback `cross-encoder/qnli-electra-base`) scores each (question, chunk) pair; chunks with relevance below 0.5 are dropped, keeping the top 3 if none pass. Relevance is 1 − P(contradiction), because a passage never entails a question |
| 9 | Extractive compression | `compressor.py` | Chunks split into sentences (NLTK); sentences with `all-MiniLM-L6-v2` cosine similarity to the question below 0.3 are dropped, always keeping each chunk's best sentence |

Measures 6 and 7 add two LLM calls per question (three when the grade is `AMBIGUOUS`); measures 1–5, 8 and 9 run without LLM calls. Each question prints a one-line summary: chunks retrieved, after NLI and after compression, the grade, whether the rewrite was used, LLM calls added and the added latency.

**`post_processing` measures**, applied to each draft answer in this order: decompose (1) → ground claims (2, 3) → sample-based checks (4, 5) → gate or revise (6, 7) → annotate provenance (8). The defences see only the question, the draft and the retrieved context, never the ground-truth answer; `query()` returns the final answer and `evaluate()` scores it against the ground truth afterwards. A draft that is already `Not found in context` is returned unchanged with no checks.

| # | Measure | Module | What it does |
|---|---|---|---|
| 1 | Claim decomposition | `verifier.py` | Draft split into sentence-level claims with NLTK's Punkt tokenizer; a sentence shorter than 3 tokens is merged into the previous claim |
| 2 | NLI groundedness | `verifier.py` | `cross-encoder/nli-deberta-v3-base` (fallback `cross-encoder/qnli-electra-base`) gives P(entailment) for each claim against each retrieved chunk, as the claim alone and prefixed with the question (drafts are often bare fragments such as `1962`); a claim's NLI support is its best score. Chunks are scored one at a time because the cross-encoder reads only 512 tokens |
| 3 | LLM faithfulness judge | `judge.py` | One `--judge` call per answer: numbered claims plus the context, answered as a JSON array of `yes` / `partial` / `no` (1, 0.5, 0) on whether the context supports each claim. Claim support is the mean of the NLI and judge scores; if the reply does not parse, NLI support is used alone |
| 4 | Self-consistency | `consistency.py` | 5 extra answers sampled at temperature 1.0 (the only value gpt-5 models accept) with the same prompt and context; consistency is the mean pairwise `all-MiniLM-L6-v2` cosine similarity over the draft and the samples |
| 5 | Semantic entropy | `consistency.py` | The same 6 answers clustered by bidirectional entailment (P ≥ 0.5 both ways, each read with the question); entropy over cluster sizes, from 0 to ln 6 ≈ 1.79 |
| 6 | Abstention gate | `pipeline.py` | Answer replaced with `Not found in context` if mean claim support < 0.5, entropy > 1.10 (ln 3, an even three-way split of meanings) or no claim passes |
| 7 | Claim revision | `verifier.py` | Otherwise claims with support < 0.5 are removed; the answer is rebuilt from slices of the draft, so it never contains new text |
| 8 | Provenance | `verifier.py` | Each surviving claim is linked to its best-entailing chunk; coverage is the percentage of surviving claims whose source chunk has NLI ≥ 0.5. Returned in `evaluate()`'s `verification` dict, not added to the answer |

Thresholds were fixed from the support and entropy distributions of the 25 questions, without consulting the ground-truth answers. Each question adds 6 LLM calls (1 judge, 5 samples) and prints a one-line summary: claims, claims dropped, aggregate support, consistency, entropy, whether it abstained and why, provenance coverage, LLM calls added and the added latency.

**`combined`** runs both sets unchanged: rewrite → normalise → hybrid retrieve → re-rank → grade → NLI filter → compress → generate draft → decompose → ground claims → sample-based checks → gate or revise → provenance. The post-processing checks (NLI support, faithfulness judge, the 5 samples and provenance) use the compressed context the generator saw. A draft of `Not found in context`, including the `INCORRECT` grade path, skips the post-processing checks. Each question adds 8 LLM calls (rewrite, grade, judge, 5 samples), 9 when the grade is `AMBIGUOUS`, or 2–3 when the draft abstains, and prints both the `Pre-processing:` and `Post-processing:` summary lines. The NLI cross-encoder and `all-MiniLM-L6-v2` are loaded once and shared by the measures that use them.

**Injection `pre_processing` defences** (`RAG/injection/<framework>/pre_processing/`) run on the retrieved chunks before the generator is called. Retrieval, chunking and the index are the same as the injection `baseline`.

| ID | Defence | Module | What it does |
|---|---|---|---|
| D1 | Sanitisation | `sanitiser.py` | Removes HTML comments, elements hidden by `display:none`, `visibility:hidden`, `font-size:0` or a `hidden` attribute, zero-width characters and Unicode tag characters; applies NFKC normalisation and collapses whitespace |
| D2 | Heuristic detector | `detectors.py` | Flags a chunk if any of eight regex patterns for instruction-like or exfiltration text matches (e.g. "ignore previous instructions", "you are now", markdown image links) |
| D3 | Classifier detector | `detectors.py` | `protectai/deberta-v3-base-prompt-injection-v2` scores each chunk locally (no API calls) in windows of up to 512 tokens; the chunk is flagged if its highest window score is at least the threshold (0.9) |
| D4 | Filter | `defence.py` | Flagged chunks are dropped (or replaced with `[removed: possible injected instructions]` when `replace_instead_of_drop` is set); if every chunk is flagged, the answer is `I couldn't find reliable information to answer that.` and the generator is not called |
| D5 | Safe prompt | `generator.py` | System prompt with the priority order system > user > documents; each chunk wrapped in `<document id="…">` tags and datamarked with `ˆ` between words (spotlighting); the question and the "documents are data" rule restated after the documents |

`--condition` selects which defences run (injection test, `pre_processing` pipeline only):

| Condition | Defences |
|---|---|
| `P0` | None (baseline prompt, no filtering) |
| `P1` | D1 |
| `P2` | D1 + D2 |
| `P3` | D1 + D3 |
| `P4` | D5 only |
| `P5` (default) | D1 + D2 + D3 + D4 + D5 |

`P0`–`P3` use the injection `baseline` prompt, so only the listed defences change. The conditions are defined in `CONDITIONS` in `defence.py`. Each case prints a one-line summary: condition, chunks retrieved, flagged chunk ids and which detector flagged them, chunks dropped, whether the fallback was used, and the added latency. The classifier threshold of 0.9 is the plan's default; it has not yet been tuned on a dev split.

**Scoring methods**
- **Heuristic** (`evaluator.py` in `baseline` and `pre_processing`): weighted mix of exact match (40%), fuzzy match (20%), embedding similarity using `all-MiniLM-L6-v2` (30%) and keyword overlap (10%), on a 0–100 scale.
- **LLM judge** (`evaluator.py` in `post_processing` and `combined`): the final answer is split into sentences (NLTK), the judge model compares each against the ground-truth answer and returns 0–100, and the scores are averaged. This is separate from the `post_processing` faithfulness judge (measure 3), which never sees the ground truth.

---

## Frameworks

Selected with `--framework`. Both frameworks implement every variant with the same parameters (chunk sizes, `k` values, MMR settings, FAISS distance threshold, prompts and scoring), so differences in results come from the frameworks themselves.

| Component | LangChain (`RAG/langchain/`) | LlamaIndex (`RAG/llamaindex/`) |
|---|---|---|
| PDF loading | `PyPDFLoader` | `PDFReader` |
| Chunking | `RecursiveCharacterTextSplitter` | `SentenceSplitter`, sized in characters |
| Embeddings | `OpenAIEmbeddings` | `OpenAIEmbedding` |
| Vector store | `FAISS` | `VectorStoreIndex` on `FaissVectorStore` |
| MMR | `max_marginal_relevance_search_by_vector` | FAISS top-48 re-ranked with LlamaIndex's MMR (`get_top_k_mmr_embeddings`) |
| BM25 | `BM25Retriever` (`rank_bm25`) | `BM25Retriever` (`bm25s`) |
| Prompts and LLM calls | `ChatPromptTemplate` piped into LangChain chat models | `ChatPromptTemplate` with `llm.predict()` on LlamaIndex LLMs |
| Pipeline class | `LCPipeline` | `LIPipeline` |

Differences that remain by design, because each framework uses its own components:
- **BM25 tokenisation**: LangChain splits on whitespace and matches case-sensitively (against a lower-cased query); LlamaIndex lower-cases, stems and removes stopwords.
- **Embedding input**: LlamaIndex's OpenAI embeddings replace newlines with spaces before embedding; LangChain sends the text unchanged.
- **MMR**: `FaissVectorStore` has no MMR mode, so LlamaIndex fetches the 48 nearest chunks and re-ranks them with `get_top_k_mmr_embeddings`. LlamaIndex penalises similarity to the most recently selected chunk, LangChain to all selected chunks; with `fetch_k` (48) above `k` (16) the two can select different chunks.
- **Retrieved sets in `pre_processing`**: because of the BM25, MMR and chunk-boundary differences above, the two frameworks retrieve 16 chunks per question but share only about 55% of them at page level, so chunk counts after the NLI filter can differ by several chunks. On the same chunks the NLI filter agrees to within 2 of 42.
- **Sampling in `post_processing`**: LangChain sets the sampling temperature with `.bind(temperature=...)` (HuggingFace models: `pipeline_kwargs` with `do_sample=True`). LlamaIndex passes it to `llm.chat()` (`generation_config` for Gemini); `HuggingFaceLLM` ignores per-call arguments, so it samples from a copy of the model whose `generate_kwargs` enable sampling. The prompts, temperature and sample count are the same.
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

3. **Add the test data.** `data/` is gitignored.

   For the hallucination test, place these in `data/hallucination/`:
   - `sl_booklet.pdf`
   - `sl_business_directory.pdf`
   - `sl_service_guide.pdf`
   - `sl_tourist_guide.pdf`
   - `swan_lagoon.json`

   For the injection test (`--test injection`), `data/injection/` holds the attack documents plus `injection_tests.json` (the manifest of test cases and their success markers). Every non-`.json` file in the folder is ingested; the manifest drives the cases. Supported document formats: `.pdf`, `.txt`, `.html`, `.md`.

4. **(Optional) GPU for local models.** The default `torch` from pip is CPU-only on Windows. To run `qwen` or `deepseek` on an NVIDIA GPU, install the CUDA build using the command from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Commands

```powershell
python main.py [--test TEST] [--framework FRAMEWORK] [--pipeline PIPELINE] [--model MODEL] [--judge MODEL]
```

| Flag | Choices | Default | Purpose |
|---|---|---|---|
| `--test` | `hallucination`, `injection` | `hallucination` | Test set to run: hallucination (Q/A over `data/hallucination/`) or indirect prompt injection (ingests every file in `data/injection/`, scores by marker matching) |
| `--framework` | `langchain`, `llamaindex` | `langchain` | RAG framework the pipeline is built with |
| `--pipeline` | `baseline`, `pre_processing`, `post_processing`, `combined` | `combined` | RAG variant to test |
| `--model` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that answers the questions |
| `--judge` | `gpt`, `gemini`, `qwen`, `deepseek` | `gpt` | LLM that scores the answers (`post_processing` and `combined` only; ignored by the heuristic pipelines) |
| `--condition` | `P0`, `P1`, `P2`, `P3`, `P4`, `P5` | `P5` | Injection defence condition (`--test injection --pipeline pre_processing` only; ignored otherwise) |

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

# Indirect prompt injection test (ingests every file in data/injection/)
python main.py --test injection --pipeline baseline --model gpt
python main.py --test injection --pipeline pre_processing --condition P5 --model gpt
python main.py --test injection --framework llamaindex --pipeline combined --model gemini

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

Each run first prints the selected framework, pipeline, model and judge, ingests the PDFs into `faiss_index/<test>/<framework>/<pipeline>/`, then prints for each of the 25 test cases:

```
Question 1: Who discovered gold in Swan Lagoon and founded the town?
Actual Answer: Jedediah 'Jed' Stone.
Response: <model answer>
Evaluation Result: <score, 0–100>
Total Time: <seconds>
```

Both scoring methods use a 0–100 scale, but they measure different things (surface and semantic similarity vs. LLM-judged correctness), so compare scores within the same scoring method.

With `--test injection`, the run instead ingests every document in `data/injection/`, then for each case in `injection_tests.json` prints the prompt, the model's response, which `injection_markers` leaked into it, and a verdict (`INJECTED (vulnerable)` if any marker matched, else `BLOCKED (safe)`), followed by a summary count. Injection scoring is marker matching, not the claim judge, so `--judge` is ignored in this mode.

---

## Known limitations

- Heuristic and LLM-judge scores are not directly comparable across pipelines.
- Retrieval differs slightly between frameworks for the reasons listed under [Frameworks](#frameworks).
- `deepseek` (DeepSeek-R1) includes `<think>` reasoning text in its answers, which affects scoring.
- In `post_processing`, long multi-sentence answers rarely entail each other in both directions, so they get higher semantic entropy than short ones and are more likely to trip the entropy gate. NLI support is also weak on the business directory, whose PDF text extracts one word per line, so its claims rely on the LLM judge.
- Injection scoring uses case-insensitive substring matching on `injection_markers`; it flags whether attack content reached the answer, not whether the model acted on it in a genuinely harmful way.
