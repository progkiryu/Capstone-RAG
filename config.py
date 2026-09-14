import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# CLI name -> (provider, model id)
MODELS = {
    "gpt": ("openai", "gpt-5-mini"),
    "gemini": ("google", "gemini-2.5-flash"),
    "qwen": ("hf", "Qwen/Qwen2.5-1.5B-Instruct"),
    "deepseek": ("hf", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
}

# RAG frameworks under RAG/ that the models can be built for
FRAMEWORKS = ["langchain", "llamaindex"]

_llm_cache = {}


def get_llm(name, framework="langchain"):
    """Build (or reuse) the chat model registered under `name` for the given RAG framework."""
    if (framework, name) in _llm_cache:
        return _llm_cache[(framework, name)]

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")

    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework '{framework}'. Choose from: {', '.join(FRAMEWORKS)}")

    provider, model_id = MODELS[name]

    if framework == "llamaindex":
        llm = _build_llamaindex_llm(provider, model_id)
    else:
        llm = _build_langchain_llm(provider, model_id)

    _llm_cache[(framework, name)] = llm
    return llm


def _build_langchain_llm(provider, model_id):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=OPENAI_API_KEY
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            api_key=GEMINI_API_KEY
        )

    import torch
    from transformers import pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    tokenizer, model = _load_hf_model(model_id)

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        do_sample=False,
        max_new_tokens=1024,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))


def _build_llamaindex_llm(provider, model_id):
    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=model_id,
            api_key=OPENAI_API_KEY,
            # LlamaIndex defaults to a 60 s timeout; LangChain uses the OpenAI SDK's 600 s
            timeout=600
        )

    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(
            model=model_id,
            temperature=0,
            api_key=GEMINI_API_KEY
        )

    import torch
    from llama_index.llms.huggingface import HuggingFaceLLM

    tokenizer, model = _load_hf_model(model_id)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # same generation settings as the LangChain pipeline above
    return HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        model_name=model_id,
        tokenizer_name=model_id,
        max_new_tokens=1024,
        generate_kwargs={
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
        },
        is_chat_model=True,
    )


def _load_hf_model(model_id):
    # local HuggingFace models are only loaded when selected
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model
