# References for the RAG framework structure

This document records the prior work that the architecture of this project draws on: the
decomposition of every pipeline into **indexer → datastore → retriever → generator → evaluator**
(`RAG/<framework>/<variant>/`), and the specific techniques inside each module. It is written to be
cited from the capstone report; numbered entries refer to the [Reference list](#reference-list) at
the end.

---

## 1. Why five components

### 1.1 The classical information-retrieval architecture

The split predates RAG. Textbook search-engine architecture already separates an *offline indexing
process* (document acquisition → text transformation → index creation) from an *online query
process* (query transformation → ranking → evaluation) [14, Ch. 2], and treats index construction,
storage and retrieval-effectiveness measurement as independently describable stages [13, Ch. 4, 8].
`indexer.py`, `datastore.py`, `retriever.py` and `evaluator.py` are that architecture; RAG adds one
stage the classical pipeline does not have — a generator that consumes the ranked list instead of
returning it.

### 1.2 The retriever/generator split

Lewis et al. [1] define RAG as exactly two parts: a DPR-style retriever over a dense vector index,
and a seq2seq generator conditioned on what it returns, with the index held as a *non-parametric
memory* that is a separate artifact from the model. REALM [2] makes the same cut (neural knowledge
retriever + knowledge-augmented encoder) and additionally treats the index as an asynchronously
rebuilt object — the argument for `datastore.py` being separate from `retriever.py` rather than
folded into it. Fusion-in-Decoder [4] isolates the generator further, as a stage that fuses many
retrieved passages. In-Context RALM [7] is the closest ancestor of `generator.py` in this project: a
*frozen* LM prompted with retrieved documents in its context window, no training involved, which is
what every pipeline variant here does.

### 1.3 The datastore as a first-class component

The term and the design come from the kNN-LM line of work [5], where the datastore is an explicit,
separately built, separately queried store. RETRO [6] generalises it to a frozen retrieval database
independent of the model, and DPR [3] establishes the offline *embed-and-index* / online *search*
API shape that `Datastore.create()` / `.load()` / `.search()` follows. The concrete store — FAISS
`IndexFlatL2` — is from Johnson, Douze and Jégou [21].

### 1.4 Framework decompositions (the direct engineering inspiration)

| Source | Its components | This project |
|---|---|---|
| **LlamaIndex**, *High-Level Concepts* [43] | Loading → Indexing → Storing → Querying → Evaluation | `indexer` → `indexer`/`datastore` → `datastore` → `retriever` + `generator` → `evaluator` |
| **Haystack** (deepset) [45] | DocumentStore, Retriever, Reader/Generator as swappable pipeline nodes | the same swap-one-node-at-a-time design, used to vary a single component per variant |
| **LangChain**, *Retrieval* docs [44] | Document loaders, text splitters, vector stores, retrievers, chains | `_loaders.py`, `indexer`, `datastore`, `retriever`, `generator` |
| **Modular RAG** [10] | Indexing, Pre-Retrieval, Retrieval, Post-Retrieval, Generation, Orchestration | the five modules plus `pipeline.py` as the orchestrator (`LCPipeline` / `LIPipeline`) |
| **FlashRAG** [11] | judger, retriever, refiner, generator, and a standalone evaluator | precedent for a research harness that fixes all components but one to isolate an effect |

LlamaIndex's five stages are the closest one-to-one match and the most direct source of the naming
used here. FlashRAG [11] is the precedent for the *experimental* use of that modularity — it exists
specifically because LangChain and LlamaIndex are, in its authors' words, heavy and inflexible for
controlled comparison, which is the same reason this project reimplements each variant in both
frameworks with identical parameters instead of using either framework's own end-to-end abstraction.

### 1.5 Why four variants rather than one pipeline

Gao et al.'s RAG survey [9] organises the field as *Naive RAG* → *Advanced RAG* → *Modular RAG*, and
splits Advanced RAG optimisations into **pre-retrieval** (chunking strategy, metadata, query
rewriting) and **post-retrieval** (re-ranking, context compression, verification). The four variants
in this repository — `baseline`, `pre_processing`, `post_processing`, `combined` — are that taxonomy
turned into an ablation: Naive RAG as the control, each optimisation class in isolation, and both
together.

---

## 2. Component by component

### 2.1 `indexer.py` — loading and chunking

- **Fixed-size overlapping chunks.** Callan's passage-level retrieval work [15] establishes
  overlapping fixed-length windows as a retrieval unit that outperforms whole documents — the origin
  of the `chunk_size` / `chunk_overlap` parameters (1000/200 in `baseline`, 800/150 in
  `pre_processing`).
- **Separator hierarchy.** LangChain's `RecursiveCharacterTextSplitter` and LlamaIndex's
  `SentenceSplitter` [43, 44] supply the `["\n\n", "\n", ".", " ", ""]` back-off used in the
  pre-processing variants, a structure-aware refinement of [15].
- **Retrieval-unit granularity as a variable.** Chen et al. [28] show the choice of unit (document /
  passage / proposition) materially changes downstream answer quality — the justification for varying
  chunk size between variants at all.
- **Source metadata on every chunk.** The indexing module of Modular RAG [10] and the pre-retrieval
  section of [9] both treat metadata attachment as an indexing-stage concern, which is where
  `doc.metadata["source"]` is set here.

### 2.2 `datastore.py` — embedding and vector storage

- **FAISS `IndexFlatL2`** — exact L2 search over dense vectors [21].
- **Bi-encoder dense retrieval**, i.e. embedding query and passage independently and scoring by
  vector proximity: DPR [3] and Sentence-BERT [22]. (Sentence-BERT is also the source of
  `all-MiniLM-L6-v2`, used by the heuristic evaluator.)
- **Sparse/dense complementarity** [20] — the result that fixed-length dense encodings lose precision
  on long documents, which is why the `pre_processing` datastore is paired with BM25 rather than used
  alone.
- **Similarity thresholding** on returned hits (`score_threshold`) as a post-retrieval filtering step
  [9, §Advanced RAG].

### 2.3 `retriever.py` — hybrid retrieval, boosting, re-ranking

- **BM25** — Robertson and Zaragoza [16], the lexical half of the hybrid retriever (`rank_bm25` in
  LangChain, `bm25s` in LlamaIndex).
- **Maximal Marginal Relevance** — Carbonell and Goldstein [18]; `lambda_mult=0.5` is their
  relevance/diversity trade-off parameter λ.
- **Hybrid sparse + dense retrieval** [20], and **fusing the resulting ranked lists** without trained
  weights — Reciprocal Rank Fusion [19], the standard reference for the combine-and-deduplicate step
  implemented here.
- **Field/feature weighting** — BM25F [17] is the precedent for boosting documents by a structural
  signal rather than term statistics alone, which is what `_extract_numeric_signals()` does for ABNs,
  phone numbers and long numeric identifiers.
- **Query normalisation and rewriting** — Rewrite-Retrieve-Read [25] and HyDE [26] motivate treating
  the query as something to transform before retrieval; `_normalize_query()` is the minimal form of
  this.
- **Re-ranking the candidate pool** — Nogueira and Cho [23] established the retrieve-then-rerank
  two-stage pattern that `score_doc()` implements heuristically.
- **Deduplication and ordering of the final context** — Liu et al. [27] show that accuracy depends
  strongly on *where* in the context a relevant passage sits, which is the argument for sorting and
  truncating to top-`k` rather than concatenating everything retrieved.

### 2.4 `generator.py` — grounded answering

- **Retrieval-conditioned generation** [1, 4], **with a frozen LM prompted in-context** [7], which
  relies on in-context learning as characterised by Brown et al. [46].
- **"Answer using ONLY the context"** — grounding instructions of this form are the mechanism Shuster
  et al. [8] show reduces hallucination in retrieval-augmented systems.
- **An explicit abstention option** (`"Not found in context"`) — SQuAD 2.0 [36] established
  unanswerability as a first-class outcome for extractive QA, and Self-RAG [12] carries the idea into
  generative RAG as explicit support and relevance judgements.

### 2.5 `evaluator.py` — two scoring regimes

**Heuristic evaluator** (`baseline`, `pre_processing`):

- **Exact match plus token overlap** — the SQuAD EM/F1 convention [35]; `exact_match_score()` and
  `keyword_overlap_score()` are EM and unigram recall against the reference.
- **Embedding similarity** as a semantic complement to surface overlap — BERTScore [37] for the
  principle, Sentence-BERT [22] for the `all-MiniLM-L6-v2` encoder actually used.
- Fuzzy string similarity (`rapidfuzz` `token_sort_ratio`) is edit-distance based and needs no
  citation beyond the metric itself; it is a robustness patch over EM.

**LLM-judge evaluator** (`post_processing`, `combined`):

- **LLM-as-a-judge** — Zheng et al. [33] for the validity of the approach and its known biases;
  G-Eval [34] for the form-filling, single-numeric-score prompt pattern used in
  `Evaluator.evaluate()`.
- **Decomposing an answer into claims and scoring each separately** — the core idea of FActScore [31]
  (atomic facts, percentage supported) and of RAGAS faithfulness [29] (statements extracted from the
  answer and verified individually); SAFE [32] extends it to long-form outputs.
  `split_into_claims()` → `evaluate_claims()` → `aggregate_claim_scores()` is a sentence-level
  approximation of this pipeline, using NLTK sentence tokenisation in place of LLM-based atomic-fact
  extraction.
- **Evaluation as a pipeline component rather than an external script** — LlamaIndex's fifth stage
  [43], FlashRAG's evaluator module [11], and ARES [30], which evaluates each RAG component
  separately.
- **Held-out benchmark separation** — BEIR [24] for the discipline of scoring a fixed system against
  a fixed question set, which is the role of `swan_lagoon.json`.

---

## 3. References for the experimental design

- **Hallucination** as the dependent variable, defined as output unsupported by the source: Ji et al.
  [38] and Huang et al. [39]; retrieval as a mitigation: [8].
- **Indirect prompt injection** — Greshake et al. [40] is the origin of the threat model tested by
  `--test injection`: instructions planted in documents that a RAG system later retrieves, with no
  direct attacker access to the prompt. BIPIA [41] is the benchmark precedent for measuring it as an
  attack-success rate over a fixed case set, which is what `injection_tests.json` and the
  marker-matching verdict implement. OWASP [42] classifies it as LLM01.
- **Comparing two frameworks under identical parameters** — the motivation FlashRAG [11] gives for a
  controlled harness rather than framework-native abstractions.

---

## Reference list

**Foundational RAG architecture**

1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *NeurIPS*, 2020. arXiv:2005.11401.
2. K. Guu, K. Lee, Z. Tung, P. Pasupat, M.-W. Chang, "REALM: Retrieval-Augmented Language Model Pre-Training," *ICML*, 2020. arXiv:2002.08909.
3. V. Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," *EMNLP*, 2020. arXiv:2004.04906.
4. G. Izacard, E. Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering," *EACL*, 2021. arXiv:2007.01282.
5. U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, M. Lewis, "Generalization through Memorization: Nearest Neighbor Language Models," *ICLR*, 2020. arXiv:1911.00172.
6. S. Borgeaud et al., "Improving Language Models by Retrieving from Trillions of Tokens," *ICML*, 2022. arXiv:2112.04426.
7. O. Ram et al., "In-Context Retrieval-Augmented Language Models," *TACL*, vol. 11, 2023. arXiv:2302.00083.
8. K. Shuster, S. Poff, M. Chen, D. Kiela, J. Weston, "Retrieval Augmentation Reduces Hallucination in Conversation," *Findings of EMNLP*, 2021. arXiv:2104.07567.

**Surveys and modular architecture**

9. Y. Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," 2023. arXiv:2312.10997.
10. Y. Gao, Y. Xiong, M. Wang, H. Wang, "Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks," 2024. arXiv:2407.21059.
11. J. Jin, Y. Zhu, X. Yang, C. Zhang, Z. Dou, "FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research," *WWW Companion*, 2025. arXiv:2405.13576.
12. A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," *ICLR*, 2024. arXiv:2310.11511.

**Classical information retrieval**

13. C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008.
14. W. B. Croft, D. Metzler, T. Strohman, *Search Engines: Information Retrieval in Practice*. Addison-Wesley, 2009. (Ch. 2, "Architecture of a Search Engine.")
15. J. P. Callan, "Passage-Level Evidence in Document Retrieval," *SIGIR*, 1994, pp. 302–310.
16. S. Robertson, H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389, 2009.
17. S. Robertson, H. Zaragoza, M. Taylor, "Simple BM25 Extension to Multiple Weighted Fields," *CIKM*, 2004.
18. J. Carbonell, J. Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries," *SIGIR*, 1998.
19. G. V. Cormack, C. L. A. Clarke, S. Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," *SIGIR*, 2009.
20. Y. Luan, J. Eisenstein, K. Toutanova, M. Collins, "Sparse, Dense, and Attentional Representations for Text Retrieval," *TACL*, vol. 9, 2021. arXiv:2005.00181.
21. J. Johnson, M. Douze, H. Jégou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, 2021. arXiv:1702.08734.
22. N. Reimers, I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," *EMNLP*, 2019. arXiv:1908.10084.
23. R. Nogueira, K. Cho, "Passage Re-ranking with BERT," 2019. arXiv:1901.04085.
24. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," *NeurIPS Datasets and Benchmarks*, 2021. arXiv:2104.08663.

**Query and context processing**

25. X. Ma, Y. Gong, P. He, H. Zhao, N. Duan, "Query Rewriting for Retrieval-Augmented Large Language Models," *EMNLP*, 2023. arXiv:2305.14283.
26. L. Gao, X. Ma, J. Lin, J. Callan, "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)," *ACL*, 2023. arXiv:2212.10496.
27. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *TACL*, vol. 12, 2024. arXiv:2307.03172.
28. T. Chen et al., "Dense X Retrieval: What Retrieval Granularity Should We Use?," 2024. arXiv:2312.06648.

**Evaluation**

29. S. Es, J. James, L. Espinosa-Anke, S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *EACL (System Demonstrations)*, 2024. arXiv:2309.15217.
30. J. Saad-Falcon, O. Khattab, C. Potts, M. Zaharia, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," *NAACL*, 2024. arXiv:2311.09476.
31. S. Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," *EMNLP*, 2023. arXiv:2305.14251.
32. J. Wei et al., "Long-form Factuality in Large Language Models," 2024. arXiv:2403.18802.
33. L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *NeurIPS Datasets and Benchmarks*, 2023. arXiv:2306.05685.
34. Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu, "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment," *EMNLP*, 2023. arXiv:2303.16634.
35. P. Rajpurkar, J. Zhang, K. Lopyrev, P. Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text," *EMNLP*, 2016. arXiv:1606.05250.
36. P. Rajpurkar, R. Jia, P. Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD," *ACL*, 2018. arXiv:1806.03822.
37. T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, Y. Artzi, "BERTScore: Evaluating Text Generation with BERT," *ICLR*, 2020. arXiv:1904.09675.

**Hallucination and prompt injection**

38. Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.
39. L. Huang et al., "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," *ACM TOIS*, 2025. arXiv:2311.05232.
40. K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, M. Fritz, "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection," *ACM Workshop on Artificial Intelligence and Security (AISec)*, 2023. arXiv:2302.12173.
41. J. Yi et al., "Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models (BIPIA)," 2023. arXiv:2312.14197.
42. OWASP Foundation, *OWASP Top 10 for Large Language Model Applications* — LLM01: Prompt Injection. https://owasp.org/www-project-top-10-for-large-language-model-applications/

**Framework documentation**

43. LlamaIndex, *High-Level Concepts* — the five stages: loading, indexing, storing, querying, evaluation. https://developers.llamaindex.ai/python/framework/getting_started/concepts/
44. LangChain, *Retrieval* documentation — document loaders, text splitters, vector stores, retrievers. https://python.langchain.com/docs/concepts/retrieval/
45. deepset, *Haystack Documentation* — Document Stores, Retrievers and Generators as pipeline components. https://docs.haystack.deepset.ai/docs/retrievers

**In-context learning**

46. T. B. Brown et al., "Language Models are Few-Shot Learners," *NeurIPS*, 2020. arXiv:2005.14165.
