import math
from itertools import combinations
from langchain_huggingface import ChatHuggingFace
from sentence_transformers import SentenceTransformer
from .verifier import answer_text, entailment

SAMPLE_COUNT = 5
# gpt-5 models accept only temperature=1, so every provider samples at 1.0 for comparability
SAMPLE_TEMPERATURE = 1.0
ENTAILMENT_THRESHOLD = 0.5

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def sampling_llm(llm):
    if isinstance(llm, ChatHuggingFace):
        # the HF pipeline is built with do_sample=False, so sampling is switched on per call
        return llm.bind(pipeline_kwargs={"do_sample": True, "temperature": SAMPLE_TEMPERATURE})
    return llm.bind(temperature=SAMPLE_TEMPERATURE)


def consistency_score(answers):
    embeddings = _embedder.encode(answers, normalize_embeddings=True)
    pairs = list(combinations(range(len(answers)), 2))
    return float(sum(embeddings[i] @ embeddings[j] for i, j in pairs) / len(pairs))


def semantic_entropy(question, answers):
    def equivalent(a, b):
        if a.strip() == b.strip():
            return True
        # answers are read with their question so bare fragments ("1962") can entail each other
        first, second = f"{question} {a}", f"{question} {b}"
        return bool((entailment([(first, second), (second, first)]) >= ENTAILMENT_THRESHOLD).all())

    clusters = []
    for answer in answers:
        for cluster in clusters:
            if equivalent(cluster[0], answer):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])

    total = len(answers)
    entropy = sum(len(c) / total * math.log(total / len(c)) for c in clusters)
    return entropy, len(clusters)


class Consistency:
    def __init__(self, llm, generator):
        self.generator = generator
        self.chain = generator.prompt | sampling_llm(llm)

    def sample(self, question, chunks):
        # same safe prompt and inputs as the draft, so the samples see what the generator saw
        inputs = self.generator.inputs(question, chunks)
        return [answer_text(self.chain.invoke(inputs)) for _ in range(SAMPLE_COUNT)]
