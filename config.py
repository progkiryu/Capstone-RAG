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

_llm_cache = {}


def get_llm(name):
    """Build (or reuse) the LangChain chat model registered under `name`."""
    if name in _llm_cache:
        return _llm_cache[name]

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")

    provider, model_id = MODELS[name]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_id,
            api_key=OPENAI_API_KEY
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            api_key=GEMINI_API_KEY
        )

    else:
        # local HuggingFace models are only loaded when selected
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

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

        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=hf_pipeline))

    _llm_cache[name] = llm
    return llm
