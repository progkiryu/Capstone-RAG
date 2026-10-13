import re
import secrets
from functools import cache
from urllib.parse import urlparse
from sentence_transformers import CrossEncoder, SentenceTransformer, util

URL = re.compile(r"https?://[^\s)\]>\"']+")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
ENTAILMENT_THRESHOLD = 0.5


@cache
def _embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@cache
def _nli():
    return CrossEncoder(NLI_MODEL)


def new_canary():
    return f"CNRY-{secrets.token_hex(6)}"


def canary_leaked(response, canary):
    return canary in response


def exfil_findings(response, question, allowed_domains):
    found = [url for url in URL.findall(response)
             if urlparse(url).netloc not in allowed_domains and url not in question]
    found += MD_IMAGE.findall(response)
    found += [email for email in EMAIL.findall(response) if email not in question]
    return found


def strip_findings(response, findings):
    for finding in findings:
        response = response.replace(finding, "")
    # newlines are kept so bulleted answers still split into separate sentences
    return re.sub(r"[ \t]+", " ", response).strip()


def split_sentences(text):
    # bulleted answers carry no sentence punctuation, so lines are split first
    return [
        sentence
        for line in text.splitlines()
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip())
        if sentence
    ]


def off_task_sentences(question, response, threshold):
    sentences = split_sentences(response)
    if not sentences:
        return []
    embedder = _embedder()
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, sentence_embeddings)[0].tolist()
    return [sentence for sentence, similarity in zip(sentences, similarities) if similarity < threshold]


def ungrounded_sentences(response, chunks):
    sentences = split_sentences(response)
    if not chunks:
        return sentences
    nli = _nli()
    entailment = nli.model.config.label2id["entailment"]
    flagged = []
    for sentence in sentences:
        scores = nli.predict([(chunk, sentence) for chunk in chunks], apply_softmax=True)
        if scores[:, entailment].max() < ENTAILMENT_THRESHOLD:
            flagged.append(sentence)
    return flagged


def most_similar_chunk(text, chunks):
    embedder = _embedder()
    similarities = util.cos_sim(
        embedder.encode(text, convert_to_tensor=True),
        embedder.encode(chunks, convert_to_tensor=True),
    )[0]
    return int(similarities.argmax())
