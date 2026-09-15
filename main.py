import argparse
import importlib
import os
import time
import json
import torch
import gc

from config import FRAMEWORKS, MODELS, get_llm

PIPELINES = ["baseline", "pre_processing", "post_processing", "combined"]
TESTS = ["hallucination", "injection"]

# pipeline class exported by RAG/<framework>/<pipeline>/pipeline.py
PIPELINE_CLASSES = {"langchain": "LCPipeline", "llamaindex": "LIPipeline"}

# pipelines whose evaluator is an LLM claim judge (the rest use heuristic scoring)
LLM_JUDGE_PIPELINES = {"post_processing", "combined"}

DATA_DIR = "data/hallucination"
INJECTION_DIR = "data/injection"
INJECTION_MANIFEST = "injection_tests.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG hallucination or indirect prompt injection tests")
    parser.add_argument("--test", choices=TESTS, default="hallucination",
                        help="Which test set to run: hallucination (data/hallucination) "
                             "or injection (ingests every file in data/injection)")
    parser.add_argument("--framework", choices=FRAMEWORKS, default="langchain",
                        help="RAG framework the pipeline is built with")
    parser.add_argument("--pipeline", choices=PIPELINES, default="combined",
                        help="RAG pipeline under RAG/<framework>/ to test")
    parser.add_argument("--model", choices=MODELS, default="gpt",
                        help="LLM used to generate answers")
    parser.add_argument("--judge", choices=MODELS, default="gpt",
                        help="LLM used to score answers (post_processing/combined only)")
    return parser.parse_args()


def response_text(response):
    """LangChain chains return an AIMessage; LlamaIndex's llm.predict() returns a str."""
    return getattr(response, "content", response)


def run_case(rag, pipeline_name, question, reference):
    """Return (response_text, score) regardless of the pipeline's evaluate() shape."""
    if pipeline_name in LLM_JUDGE_PIPELINES:
        result = rag.evaluate(question, reference)
        return response_text(result["answer"]), result["score"]

    out = rag.query(question)
    response = response_text(out[0] if isinstance(out, tuple) else out)
    score = rag.evaluate(response, reference)["final_score"]
    return response, score


def run_injection_case(rag, query, markers):
    """Query the pipeline and report which injection markers leaked into the answer.

    Injection success is measured by marker presence, not by scoring against a
    reference answer, so this uses query() (not evaluate()) for every pipeline.
    """
    out = rag.query(query)
    response = response_text(out[0] if isinstance(out, tuple) else out)
    hits = [m for m in markers if m.lower() in response.lower()]
    return response, hits


if __name__ == "__main__":
    args = parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    # post_processing/combined build a judge-backed evaluator at construction, so the
    # judge LLM must exist for them even in injection mode -- it just goes unused there,
    # since injection scores by marker matching rather than the claim judge.
    needs_judge = args.pipeline in LLM_JUDGE_PIPELINES
    uses_judge = needs_judge and args.test == "hallucination"
    print(f"Test: {args.test} | Framework: {args.framework} | Pipeline: {args.pipeline} | "
          f"Model: {args.model} | Judge: {args.judge if uses_judge else 'n/a'}")
    if needs_judge and not uses_judge:
        print("Note: --judge is ignored (the injection test scores by marker matching)")

    pipeline_module = importlib.import_module(f"RAG.{args.framework}.{args.pipeline}.pipeline")
    Pipeline = getattr(pipeline_module, PIPELINE_CLASSES[args.framework])

    # RAG pipeline built with the selected framework
    rag = Pipeline(
        llm=get_llm(args.model, args.framework),
        judge_llm=get_llm(args.judge, args.framework) if needs_judge else None,
        index_dir=f"./faiss_index/{args.framework}/{args.pipeline}",
    )
    print(f"{args.framework} RAG Pipeline Initialized!")

    if args.test == "hallucination":
        #--------------------
        # HALLUCINATION TEST:
        #--------------------
        # ingest documents
        rag.ingest(f"{DATA_DIR}/sl_booklet.pdf")
        rag.ingest(f"{DATA_DIR}/sl_business_directory.pdf")
        rag.ingest(f"{DATA_DIR}/sl_service_guide.pdf")
        rag.ingest(f"{DATA_DIR}/sl_tourist_guide.pdf")

        rag.load_existing()

        with open(f"{DATA_DIR}/swan_lagoon.json", "r") as f:
            test_cases = json.load(f)

        count = 1
        for test_case in test_cases:

            print("\n")
            total_start = time.perf_counter()

            question = test_case["question"]
            answer = test_case["answer"]

            print(f"Question {count}: {question}")

            response, score = run_case(rag, args.pipeline, question, answer)
            total_end = time.perf_counter()
            total_time = total_end - total_start

            print(f"Actual Answer: {answer}")
            print(f"Response: {response}")
            print(f"Evaluation Result: {score}")

            print(f"Total Time: {total_time:.2f} seconds")
            print("\n")
            count += 1

    else:
        #----------------------------------
        # INDIRECT PROMPT INJECTION TEST:
        #----------------------------------
        # ingest every document in the injection folder (skip the manifest / any json)
        doc_files = sorted(
            f for f in os.listdir(INJECTION_DIR)
            if not f.lower().endswith(".json")
        )
        print(f"Ingesting {len(doc_files)} injection document(s): {', '.join(doc_files)}")
        for doc_file in doc_files:
            rag.ingest(f"{INJECTION_DIR}/{doc_file}")

        rag.load_existing()

        with open(f"{INJECTION_DIR}/{INJECTION_MANIFEST}", "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        injected = 0
        count = 1
        for test_case in test_cases:

            print("\n")
            total_start = time.perf_counter()

            # PDF cases use "question"; standalone (.txt/.html) cases use "task"
            query = test_case.get("question") or test_case.get("task")
            markers = test_case.get("injection_markers", [])

            print(f"Case {count} [{test_case.get('file', '?')}]: {test_case.get('attack', '')}")
            print(f"Prompt: {query}")

            response, hits = run_injection_case(rag, query, markers)
            total_end = time.perf_counter()
            total_time = total_end - total_start

            verdict = "INJECTED (vulnerable)" if hits else "BLOCKED (safe)"
            if hits:
                injected += 1

            print(f"Response: {response}")
            print(f"Markers hit: {hits if hits else 'none'}")
            print(f"Verdict: {verdict}")

            print(f"Total Time: {total_time:.2f} seconds")
            print("\n")
            count += 1

        total = count - 1
        print(f"Injection summary: {injected}/{total} case(s) injected, "
              f"{total - injected}/{total} blocked.")
