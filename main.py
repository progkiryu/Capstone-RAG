import argparse
import importlib
import time
import json
import torch
import gc

from config import FRAMEWORKS, MODELS, get_llm

PIPELINES = ["baseline", "pre_processing", "post_processing", "combined"]

# pipeline class exported by RAG/<framework>/<pipeline>/pipeline.py
PIPELINE_CLASSES = {"langchain": "LCPipeline", "llamaindex": "LIPipeline"}

# pipelines whose evaluator is an LLM claim judge (the rest use heuristic scoring)
LLM_JUDGE_PIPELINES = {"post_processing", "combined"}

DATA_DIR = "data/hallucination"


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG hallucination tests")
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


if __name__ == "__main__":
    args = parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    uses_judge = args.pipeline in LLM_JUDGE_PIPELINES
    print(f"Framework: {args.framework} | Pipeline: {args.pipeline} | Model: {args.model} | "
          f"Judge: {args.judge if uses_judge else 'n/a (heuristic scoring)'}")
    if not uses_judge:
        print(f"Note: --judge is ignored for the {args.pipeline} pipeline")

    pipeline_module = importlib.import_module(f"RAG.{args.framework}.{args.pipeline}.pipeline")
    Pipeline = getattr(pipeline_module, PIPELINE_CLASSES[args.framework])

    # RAG pipeline built with the selected framework
    rag = Pipeline(
        llm=get_llm(args.model, args.framework),
        judge_llm=get_llm(args.judge, args.framework) if uses_judge else None,
        index_dir=f"./faiss_index/{args.framework}/{args.pipeline}",
    )
    print(f"{args.framework} RAG Pipeline Initialized!")

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

    # indirect prompt injection tests
